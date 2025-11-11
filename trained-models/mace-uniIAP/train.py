import json
import os
import pickle
import random
import sys

from typing import Callable, Dict, List, Optional
import haiku as hk
import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml


from phonax.datasets import datasets
from phonax.utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
)
from phonax.data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)
from phonax.predictors import predict_energy_forces_stress
from phonax.optimizer import optimizer
from phonax.energy_force_train import energy_force_train
from phonax.loss import WeightedEnergyFrocesStressLoss


from phonax.mace_model import MACE_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    save_dir_name = create_directory_with_random_name(
        os.path.splitext(os.path.basename(sys.argv[1]))[0]
    )
    # Save config and code
    with open(f"{save_dir_name}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{save_dir_name}/train.py", "w") as f:
        with open(sys.argv[0]) as g:
            f.write(g.read())
    
    train_loader, valid_loader,test_loader, r_max = datasets(
        r_max = config["cutoff"],
        config_dataset = config["dataset"],
    )   

    print(len(train_loader.graphs))
    print(len(valid_loader.graphs))
    #for tdata in train_loader:
    #    print(tdata.nodes.species)
    model_fn, params, num_message_passing = MACE_model(
        r_max=r_max,
        atomic_energies_dict={},
        train_graphs=train_loader.graphs,
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        hidden_irreps = "128x0e + 128x1o",
        readout_mlp_irreps = "32x0e",
        max_ell = 3,
        num_interactions = 2,
        interaction_irreps = "o3_restricted",
        epsilon = 0.4,
        correlation = 3,
        path_normalization = "path",
        gradient_normalization = "path",
        # gate = jax.nn.silu, 
        #nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = config["initialization"]['reload'] if 'reload' in config["initialization"] else None,
    )
    
    print("num_params:", sum(p.size for p in jax.tree_util.tree_leaves(params)))
    
    
    predictor = jax.jit(
        lambda w, g: predict_energy_forces_stress(lambda *x: model_fn(w, *x), g)
    )
    
    gradient_transform, steps_per_interval, max_num_intervals = optimizer(
        lr = config["training"]["learning_rate"],
        max_num_intervals = config["training"]["max_num_intervals"],
        steps_per_interval = config["training"]["steps_per_interval"],
        # weight_decay = config["training"]["weight_decay"],
    )
    optimizer_state = gradient_transform.init(params)
    print("optimizer num_params:", sum(p.size for p in jax.tree_util.tree_leaves(optimizer_state)))
    
    loss_fn = WeightedEnergyFrocesStressLoss(
        energy_weight = config["training"]["energy_weight"],
        forces_weight = config["training"]["forces_weight"],
        stress_weight = config["training"]["stress_weight"],
    )
    
    energy_force_train(
        predictor,
        params,
        optimizer_state,
        train_loader,
        valid_loader,
        test_loader,
        gradient_transform,
        loss_fn,
        max_num_intervals,
        steps_per_interval,
        save_dir_name,
        #logger,
        #directory,
        #tag,
        ema_decay = config["training"]["ema_decay"],
        patience = config["training"]["patience"],
    )
    print('done!')
    
if __name__ == "__main__":
    main()

