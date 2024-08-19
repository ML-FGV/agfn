import torch 
import numpy as np 
import tqdm 
import scipy.stats 
import matplotlib.pyplot as plt 

import torch 
import numpy as np 
import tqdm 
import itertools 

from enum import Enum 

from agfn.ricf import maximum_likelihood_batch_ricf

@torch.no_grad() 
def sample_massive_batch(gflownet, create_env, num_batches,  **expert_params): 
    env = create_env() 
    env = gflownet.sample(env) 
    log_rewards, loglik = env.log_reward(return_loglik=True) 

    for _ in tqdm.tqdm(range(num_batches - 1)): 
        env_i = create_env()
        env_i = gflownet.sample(env_i, **expert_params) 
        env.merge(env_i) 
        log_rewards_i, loglik_i = env_i.log_reward(return_loglik=True)
        log_rewards = torch.hstack([log_rewards, log_rewards_i]) 
        loglik = torch.hstack([loglik, loglik_i])  

    return env, loglik, log_rewards   
 
class EdgeType: 

    FALSE = 0 
    DIRECTED = 1 
    BACK_DIRECTED = 2 
    BIDIRECTED = 3 

def adj_to_edge_marks(adj): 
    ''' 
    Converts an adjacency matrix (B, N, N) to an element of 
    the form (B, N * (N - 1) / 2, 4) indicating the type of edge. 
    '''
    num_nodes = adj.shape[-1] 
    num_edges_complete_graph = (num_nodes) * (num_nodes - 1) // 2 
    edges = torch.triu_indices(num_nodes, num_nodes, offset=1, device=adj.device).t() 

    adj_edge_marks = torch.zeros((adj.shape[0], num_edges_complete_graph, 4), dtype=torch.int32, device=adj.device) 

    for edge_inx in range(num_edges_complete_graph): 
        edge = edges[edge_inx] 
        rev_edge = [edge[1], edge[0]] 

        edgeless = (adj[:, edge[0], edge[1]] != 1) & (adj[:, edge[1], edge[0]] != 1)
        adj_edge_marks[edgeless, edge_inx, EdgeType.FALSE] = 1 

        directed = (adj[:, edge[0], edge[1]] == 1) & (adj[:, edge[1], edge[0]] != 1)
        adj_edge_marks[directed, edge_inx, EdgeType.DIRECTED] = 1 

        back_directed = (adj[:, edge[0], edge[1]] != 1) & (adj[:, edge[1], edge[0]] == 1)
        adj_edge_marks[back_directed, edge_inx, EdgeType.BACK_DIRECTED] = 1 

        bidirected = (adj[:, edge[0], edge[1]] == 1) & (adj[:, edge[1], edge[0]] == 1)
        adj_edge_marks[bidirected, edge_inx, EdgeType.BIDIRECTED] = 1 
    
    return adj_edge_marks 
    
def edge_marks_to_adj(samples_edge_marks): 
    ''' 
    Convert an object of type `edge_marks` (B, E, 4) into an adjacency matrix 
    (B, N, N), E = N (N - 1) / 2 (see previous function) 
    '''
    sample_size = samples_edge_marks.shape[0] 
    num_edges_complete_graph = samples_edge_marks.shape[1] 
    num_nodes = int(.5 * (1 + np.sqrt(1 + 8 * num_edges_complete_graph))) 
    edges = torch.triu_indices(num_nodes, num_nodes, offset=1, device=samples_edge_marks.device).t()

    adj_matrices = torch.zeros((sample_size, num_nodes, num_nodes), device=samples_edge_marks.device) 
    sample_idxs = torch.arange(sample_size, device=samples_edge_marks.device) 
    for edge_inx, edge in enumerate(edges): 
        is_directed_ij = (samples_edge_marks[:, edge_inx, EdgeType.DIRECTED] == 1)
        is_directed_ji = (samples_edge_marks[:, edge_inx, EdgeType.BACK_DIRECTED] == 1) 
        is_bidirected = (samples_edge_marks[:, edge_inx, EdgeType.BIDIRECTED] == 1)  

        rev_edge = [edge[1], edge[0]] 
        adj_matrices[sample_idxs[is_directed_ij], edge[0], edge[1]] = 1 
        adj_matrices[sample_idxs[is_directed_ji], edge[1], edge[0]] = 1 
        
        adj_matrices[sample_idxs[is_bidirected], edge[0], edge[1]] = 1 
        adj_matrices[sample_idxs[is_bidirected], edge[1], edge[0]] = 1 

    return adj_matrices 

def impsmp(f, w, dim=0): # mixed importance sampling estimator for E[f]  
    return (f * w).sum(dim) / w.sum(dim) 

def setdiff1d(t1, t2):
    mask = torch.ones_like(t1, dtype=torch.bool)
    mask[torch.isin(t1, t2)] = False
    return t1[mask]

# methods 
def fpost(edges, feedbacks, priors, queries, pi):
    edges_q = edges[:, queries, :] 
    priors_q = priors[queries, :] 
    fdbkcs = torch.zeros((edges.shape[1], 4), dtype=torch.float32, device=priors.device) 
    fdbkcs[queries, feedbacks] = 1 
    fdbkcs = fdbkcs[queries] 
    # posterior distribution 
    posteriors = priors_q * (pi * fdbkcs + (1 - pi) / 3 * (1 - fdbkcs)) 
    posteriors = posteriors / posteriors.sum(dim=-1, keepdims=True) 
    return (edges_q * posteriors[None, ...]).sum(dim=-1) 

def is_mag(adj): 
    num_nodes = adj.shape[1] 
    symmetric_adj = (adj + torch.transpose(adj, 1, 2)) / 2 
    parents_matrix = (symmetric_adj == .5).int() * adj
    spouses_matrix = (symmetric_adj == 1.).int() * adj 
    adj_exp = torch.matrix_exp(parents_matrix) # (batch_size, num_nodes, num_nodes) 
    threshold = torch.abs((torch.diagonal(adj_exp, dim1=-1, dim2=-2).sum(dim=-1) + torch.sum(adj_exp * spouses_matrix, dim=(-1, -2))) - num_nodes) 
    return threshold < 1e-8 

def rewards_bic_from_adj(adj, samples, gflownet, log_reward, gamma=.5):
    num_nodes = adj.shape[-1] 
    symmetric_adj = (adj + torch.transpose(adj, 1, 2)) / 2 
    parents_matrix = (symmetric_adj == .5).int() * adj
    spouses_matrix = (symmetric_adj == 1.).int() * adj 
    num_edges = parents_matrix.sum(dim=(-1, -2)) + (spouses_matrix.sum(dim=(-1, -2))) / 2 
    num_nodes = parents_matrix.shape[1] 
    num_samples = samples.shape[0] 
    max_loglikhood = maximum_likelihood_batch_ricf(samples, parents_matrix, spouses_matrix) 
    bic = - 2 * max_loglikhood + num_edges * np.log(num_samples) + 4 * gamma * num_edges * np.log(num_nodes) 
    return ((- bic - log_reward.mu) / (log_reward.temperature * log_reward.std)), max_loglikhood 

def sample_from_expert(feedbacks, queries, priors, batch_size, epochs, gflownet, pi, samples, log_reward): 
    priors_q = priors[queries, :] 
    fdbkcs = torch.zeros_like(priors) 
    fdbkcs[queries, feedbacks] = 1 
    fdbkcs = fdbkcs[queries] 
    # Posterior distribution 
    posteriors = priors_q * (pi * fdbkcs + (1 - pi) / 3 * (1 - fdbkcs)) 
    posteriors = posteriors / posteriors.sum(dim=-1, keepdims=True) 
    upriors = priors.clone() 
    upriors[queries, :] = posteriors 

    # Sample from the distribution over graphs induced by the updated priors 
    m = torch.distributions.OneHotCategorical(upriors) 
    sgraphs = torch.zeros((0, priors.shape[0], 4), dtype=torch.int8, device=priors.device) 
    rewards = torch.zeros((0,), dtype=priors.dtype, device=priors.device) 
    logliks = torch.zeros((0,), dtype=priors.dtype, device=priors.device)
    for _ in range(epochs): 
        nsgraphs = m.sample(sample_shape=(batch_size,))
        madjacen = edge_marks_to_adj(nsgraphs) 
        ismagidx = is_mag(madjacen) 
        nsgraphs = nsgraphs[ismagidx] 
        nrewards, nlogliks = rewards_bic_from_adj(madjacen[ismagidx].to(dtype=samples.dtype), samples, gflownet, log_reward) 
        sgraphs = torch.cat([sgraphs, nsgraphs], dim=0)  
        rewards = torch.hstack([rewards, nrewards]) 
        logliks = torch.hstack([logliks, nlogliks])
    return sgraphs, rewards, logliks 

def evaluate_utility(edges, candidate_post, current_post, query, priors, log_rewards): 
    weights = torch.exp(torch.log(candidate_post).sum(dim=-1)) 
    ereward = impsmp(log_rewards, weights) 
    logcp = torch.log(current_post).sum(dim=-1) 
    if current_post.shape[1] >= 1: 
        elog = impsmp(logcp, weights) 
        metric = ereward + elog 
    else: 
        metric = ereward 
    return metric 

def evaluate_utility_from_expert(edges, candidate_post, current_post, queries, query, priors, log_rewards): 
    nonqueries = setdiff1d(torch.arange(edges.shape[1], device=priors.device), queries) 
    epriors = (edges * priors[None, ...]).sum(dim=-1) 
    weights = torch.exp(log_rewards + torch.log(candidate_post[:, -1]) - torch.log(epriors[:, nonqueries]).sum(dim=-1)) 
    if current_post.shape[1] >= 1: 
        ereward = impsmp(log_rewards, weights) 
        logcp = torch.log(current_post).sum(dim=-1) 
        elog = impsmp(logcp, weights) 
        metric = ereward + elog 
    else: 
        ereward = impsmp(log_rewards, weights) 
        metric = ereward 
        # pass 
    # pass 
    return metric 

def nquery(
        edges, log_rewards, feedbacks, priors, queries, pi, 
        alpha=None, expert_edges=None, expert_log_rewards=None):
    viable_edges = setdiff1d(torch.arange(edges.shape[1], device=priors.device), queries) 
    criterion = list() 
    # marginal distribution over the feedbacks 
    fmarg = priors * pi + (1 - priors) * (1 - pi) / 3 
    assert torch.isclose(fmarg.sum(dim=-1), torch.tensor(1., dtype=fmarg.dtype)).all(), fmarg.sum(dim=-1)

    # compute the expert's model posteriors 
    if expert_edges is not None: 
        expert_current_post = fpost(expert_edges, feedbacks, priors, queries, pi) 
    current_post = fpost(edges, feedbacks, priors, queries, pi)

    for query in viable_edges: 
        expected_metric = torch.zeros((4,), dtype=torch.float32, device=log_rewards.device)  
        for feedback_idx in torch.arange(4, device=log_rewards.device): 
            # append 
            feedbacks_cl = torch.hstack([feedbacks, feedback_idx.unsqueeze(0)]) 
            queries_cl = torch.hstack([queries, query.unsqueeze(0)]) 
            candidate_post = fpost(edges, feedbacks_cl, priors, queries_cl, pi) 

            utility = evaluate_utility(edges, candidate_post, current_post, query, priors, log_rewards) 

            # expert 
            if expert_edges is not None: 
                expert_candidate_post = fpost(expert_edges, feedbacks_cl, priors, queries_cl, pi)  
                expert_utility = evaluate_utility_from_expert(expert_edges, expert_candidate_post, expert_current_post, 
                        queries, query, priors, expert_log_rewards) 
                # a convex combination of consistent estimators is consistent 
                utility = alpha * utility + (1 - alpha) * expert_utility 
            expected_metric[feedback_idx] += utility 
        criterion.append((expected_metric * fmarg[query, :]).sum()) 
    criterion = torch.tensor(criterion) 
    idx = torch.argmax(criterion) 
    query = viable_edges[idx] 
    return query