import torch 
import pandas as pd 
import numpy as np 
import networkx as nx 

import sys 
sys.path.append('.') 

import matplotlib.pyplot as plt 

from experiments.utils import (
    get_argument_parser, 
    train, 
    simulate_gaussian_data, 
    create_agfn, 
    create_log_reward, 
    create_mag, 
    sample_many_mags    
)  
from agfn.gym import LogReward 

torch.set_default_dtype(torch.float64) 

def main(config):
    if config.simulate_data: 
        data = simulate_gaussian_data(
            config.num_nodes, config.max_degree, config.num_samples, device=config.device, seed=config.seed
        )  
        node_labels = [f'{i + 1}' for i in range(config.num_nodes)] 

        # Save simulated data 
        torch.save(
            data, f'simulated-{config.num_nodes}-{config.num_samples}.pt'
        )
    else: 
        data = pd.read_csv(config.filename) 
        node_labels = data.columns.tolist()  
        data = torch.tensor(data.values, device=config.device, dtype=torch.float64)    
        
        config.num_nodes = data.shape[1] 

    # Create an AGFN 
    agfn = create_agfn(config.num_nodes, config.device, config.hidden_dim)  

    if config.load_model_path is not None: 
        agfn.load_state_dict(
            torch.load(config.load_model_path) 
        )

    # Create LogReward 
    log_reward = create_log_reward(data, config.temperature, config.use_simple_reward)
    if isinstance(log_reward, LogReward): 
        log_reward.temperature = 1. 
        log_reward.warmup(agfn, lambda: create_mag(config, log_reward), 32) 
        log_reward.temperature = config.temperature 

    # Train the model 
    if not config.only_inference: 
        losses = train(
            agfn, 
            lambda: create_mag(config, log_reward), 
            config.epochs, 
            config.lr, 
            config.lr_logz, 
            config.use_scheduler 
        )

        # Plot the loss function throughout training 
        plt.plot(losses) 
        plt.savefig('losses_agfn_without_guidance.pdf', bbox_inches='tight') 
        plt.clf() 

    if config.save_model_path is not None: 
        torch.save(
            agfn.state_dict(), config.save_model_path 
        )

    # Generate graphs from the learned distribution 
    agfn.eval() 
    mag, log_rewards = sample_many_mags(
        config, agfn=agfn, initial_mag=create_mag(config, log_reward)
    ) 

    plt.figure(figsize=(9*5, 4*5)) 
    indices = torch.topk(log_rewards, k=10).indices 
    for e, i in enumerate(indices): 
        plt.subplot(2, 5, e + 1)  
        mag.draw(i, labels=node_labels) 
        plt.title(log_rewards[i]) 

    plt.savefig('samples_agfn.pdf', bbox_inches='tight') 

if __name__ == '__main__': 
    parser = get_argument_parser() 
    config = parser.parse_args() 
    main(config) 