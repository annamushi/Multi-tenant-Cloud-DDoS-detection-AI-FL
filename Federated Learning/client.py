import flwr as fl
import argparse
from utils import load_local_data, load_model, train_local, evaluate_local

#load the tenant data and create local LR model
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.x_train, self.y_train, self.x_test, self.y_test = load_local_data(cid)
        input_dim = self.x_train.shape[1]
        self.model = load_model(input_dim)

    #return current local model weights to the server-1
    def get_parameters(self, config):
        return self.model.get_weights()


    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        train_local(self.model, self.x_train, self.y_train)
        return self.model.get_weights(), len(self.x_train), {}


    #tests the global model on local test data and returns metrics
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, metrics = evaluate_local(self.model, self.x_test, self.y_test)
        metrics['tenant'] = self.cid
        return loss, len(self.x_test), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=str, required=True)
    parser.add_argument('--server', type=str, required=True)
    args = parser.parse_args()

    #start the Flw client
    fl.client.start_client(
        server_address=args.server,
        client=FlowerClient(args.cid).to_client(),
    )


if __name__ == '__main__':
    main()
