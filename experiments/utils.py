import torch 
import argparse 
import tqdm 
import networkx as nx 

from itertools import chain 
from copy import deepcopy 

from agfn.gflownet import GFlowNet 
from agfn.models import ForwardPolicy, BackwardPolicy 
from agfn.gym import MAG, LogReward, LogRewardSlightlyComplex  

def get_argument_parser(): 
    parser = argparse.ArgumentParser('description') 

    parser.add_argument(
        '--filename', type=str, default=None, help='path to the data file' 
    )
    parser.add_argument(
        '--use_simple_reward', action='store_true', help='use simple reward' 
    )
    parser.add_argument(
        '--only_inference', action='store_true', help='only perform inference' 
    )
    parser.add_argument(
        '--device', type=str, default='cpu', help='device to train the model' 
    )
    
    # Artifacts 
    parser.add_argument(
        '--save_model_path', type=str, default=None, help='path to save the model' 
    )
    parser.add_argument(
        '--load_model_path', type=str, default=None, help='load model path' 
    )

    # Simulated data 
    parser.add_argument_group(
        'Simulated data', description='parameters for simulating data' 
    ) 
    parser.add_argument(
        '--simulate_data', action='store_true', help='simulate the data from a true graph'  
    )

    parser.add_argument(
        '--num_samples', type=int, default=500, help='dataset size' 
    )
    parser.add_argument(
        '--num_nodes', type=int, default=5, help='number of variables'  
    )
    parser.add_argument(
        '--max_degree', type=int, default=2, help='maximum degree of the simulated DAG' 
    )

    # AGFN config
    parser.add_argument_group(
        'AGFN config', description='hyperparameters for the AGFN model'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=128, help='hidden dimension of the neural network' 
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size for training AGFN' 
    )
    parser.add_argument(
        '--temperature', type=float, default=1., help='R_{T}(x) = [ log R(x) ] / T' 
    )
    parser.add_argument(
        '--num_samples_inference', type=int, default=32, help='number of MAGs for inference' 
    )
    
    # Training config 
    parser.add_argument_group(
        'Training config', description='training hyperparameters, e.g., learning rate, etc.' 
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate' 
    )
    parser.add_argument(
        '--lr_logz', type=float, default=1e-1, help='learning rate for log Z' 
    )
    parser.add_argument(
        '--epochs', type=int, default=512, help='number of epochs for training' 
    )
    parser.add_argument(
        '--use_scheduler', action='store_true', help='use learning rate scheduler'  
    )

    return parser 

def create_mag(config, log_reward): 
    return MAG(config.num_nodes, config.batch_size, log_reward, device=config.device) 

def train(gflownet, create_env, epochs, lr=1e-3, lr_logz=1e-1, use_scheduler=False):
    optimizer = torch.optim.Adam([
        {'params': chain(gflownet.pf.parameters(), gflownet.pb.parameters()), 'lr': lr}, 
        {'params': gflownet.log_z, 'lr': lr_logz}]) 
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs) 

    losses = list() 
    pbar = tqdm.tqdm(range(epochs), disable=False) 
    
    for _ in pbar: 
        optimizer.zero_grad() 
        env = create_env() 
        loss = gflownet(env) 
        loss.backward()     
        optimizer.step() 

        pbar.set_postfix(loss=loss.item()) 
        # Update the exploration rate 
        losses.append(loss.item()) 
        if use_scheduler: 
            scheduler.step() 
    return losses  

def create_agfn(num_nodes, device, hidden_dim): 
    pf = ForwardPolicy(num_nodes, num_layers=2, hidden_dim=hidden_dim, device=device) 
    pb = BackwardPolicy(device) 
    return GFlowNet(pf, pb, criterion='tb', device=device) 

def create_log_reward(data, temperature: float = 1., use_simple_reward=False):
    if use_simple_reward: 
        return LogRewardSlightlyComplex(data.shape[1], seed=42, device=data.device)
    return LogReward(data, temperature=temperature, device=data.device)
    
def sample_many_mags(config, agfn: GFlowNet, initial_mag): 
    log_rewards = list() 
    samples = None 

    for _ in tqdm.trange(config.num_samples_inference): 
        sampled_mag = agfn.sample(
            deepcopy(initial_mag) 
        )
        if samples is None: 
            samples = sampled_mag 
        else: 
            samples.merge(sampled_mag) 
        
        log_rewards.append(sampled_mag.log_reward()) 
    
    log_rewards = torch.hstack(log_rewards) 
    return samples, log_rewards 

def sample_from_structured_gaussian(dag, num_samples, seed=42, device='cpu'):
    parents_matrix, spouses_matrix = dag.directed_amat.squeeze(dim=0), dag.bidirected_amat.squeeze(dim=0) 
    num_nodes = parents_matrix.shape[0]
    nodes = torch.arange(num_nodes, device=device)

    g = torch.Generator(device=device) 
    beta_scale = .5
    beta_loc = 2.
    
    g.manual_seed(seed) 
    beta = beta_scale * torch.rand(size=(num_nodes, num_nodes), device=device, generator=g) + beta_loc
    beta = beta * parents_matrix

    # Sample `omega` from the correspondent Cholesky's decomposition
    omega_scale = .1
    omega_loc = .1

    g.manual_seed(2*seed) 
    omega = omega_scale * torch.rand(size=(num_nodes, num_nodes), device=device, generator=g) + omega_loc
    omega = torch.triu(omega).T @ torch.triu(omega) / num_nodes

    # Apply the structural masks
    g.manual_seed(3*seed) 
    omega = omega * spouses_matrix
    omega[nodes, nodes] = omega_scale * torch.rand(size=(num_nodes,), device=device, generator=g) + omega_loc

    # Identify the correspondent covariance matrix
    sigma = torch.linalg.inv(torch.eye(num_nodes, device=device) - beta)
    sigma = (sigma.T @ omega @ sigma) 

    dist = torch.distributions.MultivariateNormal(loc=torch.zeros(num_nodes, device=device),
                                                covariance_matrix=sigma)
    samples = dist.sample(sample_shape=(num_samples,))
    return samples 

def configuration_model(num_nodes: int, max_degree: int, device='cpu', seed=42): 
    # Sample graph configuration model
    g = torch.Generator(device=device) 
    g.manual_seed(seed) 
    in_deg_sequence = torch.randint(max_degree, size=(num_nodes,), device=device, generator=g)
    # randomly permute this sequence to create a list of out degrees 
    rand_indices = torch.randperm(num_nodes) 
    out_deg_sequence = in_deg_sequence[rand_indices]  
    return nx.directed_configuration_model(in_deg_sequence, out_deg_sequence, seed=seed)
    
def simulate_gaussian_data(
        num_nodes: int, max_degree: int, num_samples: int, device='cpu', seed=42
    ):
    G = configuration_model(num_nodes, max_degree, device)    
    dag = nx.DiGraph() 
    dag.add_nodes_from(G.nodes()) 
    dag.add_edges_from([(u, v) for (u, v) in G.edges() if u < v])
    assert nx.is_directed_acyclic_graph(dag) 

    # Create a MAG with the sampled structure 
    mag = MAG(num_nodes, batch_size=1, log_reward=None, device=device) 
    if len(dag.edges) > 0: 
        mag.edge_list = torch.hstack([
            torch.zeros((len(dag.edges),1), device=device), 
            torch.tensor(list(dag.edges), device=device), 
            torch.ones((len(dag.edges),1), device=device) 
        ]).long() 

    mag.stopped[:] = 1. 
    samples = sample_from_structured_gaussian(mag, num_samples, seed, device)
    return dag, samples 
