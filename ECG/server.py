from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from models import *

import flwr as fl

from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.strategy.fedopt import FedOpt

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd: int, results, failures,) -> Optional[fl.common.Weights]:
        weights = super().aggregate_fit(rnd, results, failures)
        if weights is not None:
            # Save weights
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""

    if rnd > 20:
        lr = 5e-6
    elif rnd > 15:
        lr = 1e-5
    elif rnd > 10:
        lr = 5e-5
    else:
        lr = 1e-4

    config = {
        "learning_rate": str(lr),
        "batch_size": str(8),
        "round": str(rnd),
    }
    return config


if __name__ == "__main__":
    # Select strategy for server side operations. Default is FedAvg.
    strategy = fl.server.strategy.FedAvg()

    # Choose number of rounds of training(One round consists of client side training, sending weights to server,
    # aggregating weights at server and sending updated weights back to clients) and IP address to run server on. 
    fl.server.start_server("10.9.9.2:7100",config={"num_rounds": 10}, strategy=strategy)
