import flwr as fl
import pandas as pd
import numpy as np
from datetime import datetime

CLIENT_RESULTS = []

def weighted_average(metrics):
    total = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    total_examples = 0
    for num_examples, m in metrics:
        for k in total.keys():
            total[k] += m.get(k, 0.0) * num_examples
        total_examples += num_examples
    if total_examples == 0:
        return {k: 0.0 for k in total.keys()}
    return {k: total[k] / total_examples for k in total.keys()}


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.final_weights = None

    def aggregate_fit(self, rnd, results, failures):
        # pick model from client with most data
        best_result = max(results, key=lambda x: x[1].num_examples)
        parameters = best_result[1].parameters
        self.final_weights = parameters

        total_bytes = sum(
            sum(len(t) for t in result.parameters.tensors)
            for _, result in results
        )
        print(f'Round {rnd} - Overhead: {total_bytes} bytes ({total_bytes/1024:.2f} KB)')

        return parameters, {}

    def aggregate_evaluate(self, rnd, results, failures):
        for client, evaluate_res in results:
            metrics = evaluate_res.metrics
            CLIENT_RESULTS.append([
                rnd,
                metrics.get('tenant', 'unknown'),
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1', 0.0),
            ])
        return super().aggregate_evaluate(rnd, results, failures)


def main():
    strategy = Strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address='0.0.0.0:8081',
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    df = pd.DataFrame(
        CLIENT_RESULTS,
        columns=['round', 'tenant', 'accuracy', 'precision', 'recall', 'f1'],
    )

    final_round = df['round'].max()
    final_df = df[df['round'] == final_round]

    summary = pd.DataFrame([{
        'Method': 'Proposed FL XGBoost',
        'Accuracy': float(f'{final_df["accuracy"].mean():.6f}'),
        'Precision': float(f'{final_df["precision"].mean():.6f}'),
        'Recall': float(f'{final_df["recall"].mean():.6f}'),
        'F1-Score': float(f'{final_df["f1"].mean():.6f}'),
    }])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'fl_xgb_results_detailed_{timestamp}.csv', index=False)
    summary.to_csv(f'fl_xgb_results_summary_{timestamp}.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()