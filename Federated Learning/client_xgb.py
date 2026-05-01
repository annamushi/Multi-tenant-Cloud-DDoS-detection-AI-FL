import flwr as fl
import argparse
import numpy as np
from utils_xgb import load_local_data, load_model, train_local, evaluate_local

class XGBClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.x_train, self.y_train, self.x_test, self.y_test = load_local_data(cid)
        self.model = load_model()
        self.model.fit(self.x_train, self.y_train)

    def get_parameters(self, config):
        return [self.model.get_booster().save_raw('json')]

    def fit(self, parameters, config):
        self.model.get_booster().load_model(bytearray(parameters[0]))
        train_local(self.model, self.x_train, self.y_train)
        return [self.model.get_booster().save_raw('json')], len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.get_booster().load_model(bytearray(parameters[0]))
        loss, metrics = evaluate_local(self.model, self.x_test, self.y_test)
        metrics['tenant'] = self.cid
        return loss, len(self.x_test), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=str, required=True)
    parser.add_argument('--server', type=str, required=True)
    args = parser.parse_args()

    fl.client.start_client(
        server_address=args.server,
        client=XGBClient(args.cid).to_client(),
    )


if __name__ == '__main__':
    main()