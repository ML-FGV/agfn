import torch 
import torch.nn as nn 
import tqdm 
import wandb 
from itertools import chain 
from abc import ABCMeta, abstractmethod  

class Environment(metaclass=ABCMeta):

    def __init__(self, batch_size: int, max_trajectory_length: int, log_reward: nn.Module, device: torch.device):
        self.batch_size = batch_size
        self.device = device 
        self.max_trajectory_length = max_trajectory_length
        self.batch_ids = torch.arange(self.batch_size, device=self.device) 
        self.traj_size = torch.ones((self.batch_size,), device=self.device)         
        self.stopped = torch.zeros((self.batch_size), device=self.device)
        self.is_initial = torch.ones((self.batch_size,), device=self.device)
        self._log_reward = log_reward

    @abstractmethod
    def apply(self, actions: torch.Tensor):
        pass

    @abstractmethod
    def backward(self, actions: torch.Tensor):
        pass

    @torch.no_grad()
    def log_reward(self):
        return self._log_reward(self)
    
    @torch.no_grad() 
    def merge(self, batch_state): 
        self.batch_ids = torch.hstack([self.batch_ids, batch_state.batch_ids]) 
        self.batch_size += batch_state.batch_size 
        self.stopped = torch.hstack([self.stopped, batch_state.stopped]) 
        self.is_initial = torch.hstack([self.is_initial, batch_state.is_initial]) 
    
    @property 
    def unique_input(self): 
        raise NotImplementedError 

class ForwardPolicyMeta(nn.Module, metaclass=ABCMeta): 
    
    masked_value = -1e5 

    def __init__(self, eps=.3): 
        super(ForwardPolicyMeta, self).__init__() 
        self.eps = eps 

    @abstractmethod 
    def get_latent_emb(self): 
        pass 

    @abstractmethod 
    def get_pol(self): 
        pass 

    def get_actions(self, pol, mask=None): 
        if mask is None: 
            uniform_pol = torch.ones_like(pol) 
        else: 
            uniform_pol = torch.where(mask==1., 1., self.masked_value) 
        uniform_pol = torch.softmax(uniform_pol, dim=-1) 
        eps = 0. if not self.training else self.eps 
        exp_pol = pol * (1 - eps) + eps * uniform_pol 
        actions = torch.multinomial(exp_pol, num_samples=1, replacement=True) 
        actions = actions.squeeze(dim=-1) 
        return actions     

class ForwardPolicyProduct(ForwardPolicyMeta): 

    def __init__(self, gflownets, eps=.3, device='cpu'): 
        super(ForwardPolicyProduct, self).__init__(eps=eps)  
        self.device = device 
        self.policies = [gflownet.pf for gflownet in gflownets] 

    def get_latent_emb(self, batch_state): 
        latent_emb_lst = list() 
        for policy in self.policies: 
            latent_emb = policy.get_latent_emb(batch_state) 
            latent_emb_lst.append(latent_emb) 
        return latent_emb_lst 
    
    def get_pol(self, latent_emb_lst, mask): 
        pol_lst = list()  
        for policy, latent_emb in zip(self.policies, latent_emb_lst): 
            pol, _ = policy.get_pol(latent_emb, mask) 
            pol_lst.append(pol.unsqueeze(-1)) 
        pol = torch.cat(pol_lst, dim=-1).prod(dim=-1)
        pol /= pol.sum(dim=1, keepdims=True) 
        return pol   

    def forward(self, batch_state, actions=None): 
        latent_emb_lst = self.get_latent_emb(batch_state) 
        pol = self.get_pol(latent_emb_lst, batch_state.forward_mask) 
        if actions is None: actions = self.get_actions(pol, batch_state.forward_mask) 
        return actions, torch.log(pol[batch_state.batch_ids, actions]) 
        
def train(gflownet, create_env, epochs, batch_size, lr=1e-3, lr_logz=1e-1, lr_r=1e-5,  
            min_eps=1e-2, use_scheduler=False, use_wandb=False):
    optimizer = torch.optim.NAdam([
        {'params': chain(gflownet.pf.parameters(), gflownet.pb.parameters()), 'lr': lr}, 
        {'params': gflownet.log_z, 'lr': lr_logz}]) 
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs) 

    initial_eps = gflownet.pf.eps 
    eps_schedule = 2 * (initial_eps - min_eps) / epochs

    losses = list() 
    pbar = tqdm.tqdm(range(epochs), disable=False) 
    for epoch in pbar: 
        if (epoch > epochs // 3): 
            gflownet.pb.requires_grad_(False) 
        optimizer.zero_grad() 
        env = create_env(batch_size=batch_size) 
        loss = gflownet(env) 
        loss.backward()     

        optimizer.step() 

        pbar.set_postfix(loss=loss.item()) 
        # Update the exploration rate 
        gflownet.pf.eps = max(min_eps, gflownet.pf.eps - eps_schedule * epoch) 
        losses.append(loss.item()) 
        if use_wandb: 
            wandb.log({'loss': loss.item()}) 
        
        if use_scheduler: 
            scheduler.step() 
        
    return gflownet, losses  

@torch.no_grad() 
def init_logz(gflownet, create_env, num_batches, batch_size): 
    log_rewards = list() 
    marginal_log_dist = list() 
    for _ in tqdm.tqdm(range(num_batches)): 
        env = create_env(batch_size) 
        env = gflownet.sample(env) 
        log_rewards.append(env.log_reward()) 
        marginal_log_prob = gflownet.sample_many_backward(env, num_trajectories=8)
        marginal_prob = torch.logsumexp(marginal_log_prob, dim=1) 
        marginal_prob -= torch.log(torch.tensor(marginal_log_prob.shape[1]))  
        marginal_log_dist.append(marginal_prob) 
    log_rewards = torch.hstack(log_rewards) 
    marginal_log_dist = torch.hstack(marginal_log_dist) 
    gflownet.log_z.fill_(torch.logsumexp(log_rewards - marginal_log_dist, dim=0))
    gflownet.log_z.add_(-torch.log(torch.tensor(len(log_rewards))))  

    assert gflownet.log_z.requires_grad 

@torch.no_grad() 
def sample_massive_batch(gflownet, create_env, num_batches, num_back_traj=1): 
    env = create_env() 
    env = gflownet.sample(env) 
    if num_back_traj is not None:
        marginal_log = gflownet.sample_many_backward(env, num_trajectories=num_back_traj) 
    log_rewards = env.log_reward() 

    for _ in tqdm.tqdm(range(num_batches - 1)): 
        env_i = create_env()
        env_i = gflownet.sample(env_i) 
        env.merge(env_i) 
        if num_back_traj is not None: 
            marginal_log = torch.vstack([marginal_log, gflownet.sample_many_backward(
                            env_i, num_trajectories=num_back_traj)])
        log_rewards = torch.hstack([log_rewards, env_i.log_reward()]) 

    if num_back_traj is not None: 
        return env, marginal_log, log_rewards   
    else: 
        return env, log_rewards 
 
def unique(x, dim=-1):
    values, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=values.device)
    # inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return values, inverse, counts, inverse.new_empty(values.size(dim)).scatter_(dim, inverse, perm)

def marginal_dist(env, marginal_log, log_rewards, dim=-1, return_indices=False):
    if not torch.is_tensor(env): 
        values, inverse, counts, indices = unique(env.unique_input, dim=dim)
    else: 
        values, inverse, counts, indices = unique(env, dim=dim) 
    # Compute learned distribution 
    marginal_log = marginal_log - marginal_log.max() # numerical stability 
    marginal_log_batch = torch.zeros((values.size(0), marginal_log.shape[1]), device=values.device)
    marginal_log_batch.scatter_add_(dim=0, index=inverse.view(-1, 1), src=marginal_log.exp())
    marginal_log_batch = marginal_log_batch.sum(dim=-1)
    marginal_log_batch /= (counts * marginal_log.shape[1]) 
    learned_dist = marginal_log_batch / marginal_log_batch.sum()  
    # Compute the target distribution 
    target_dist = (log_rewards[indices] - torch.logsumexp(log_rewards[indices], dim=0)).exp() 
    if return_indices: 
        return learned_dist, target_dist, indices     
    return learned_dist, target_dist 


def submarginal_dist(env, marginal_log, log_rewards, dim=-1):
    values, inverse, counts, indices = unique(env.unique_input, dim=dim)
    # Compute learned distribution 
    marginal_log_batch = torch.zeros((values.size(0), marginal_log.shape[1]), device=values.device)
    marginal_log_batch.scatter_add_(dim=0, index=inverse.view(-1, 1), src=marginal_log.exp())
    marginal_log_batch = marginal_log_batch.sum(dim=-1)
    marginal_log_batch /= (counts * marginal_log.shape[1]) 
    learned_dist = marginal_log_batch / marginal_log_batch.sum()  
    # Compute the target distribution 
    target_dist = (log_rewards[indices] - torch.logsumexp(log_rewards[indices], dim=0)).exp() 
    return values, log_rewards[indices], learned_dist, target_dist 

@torch.no_grad() 
def eval_step_l1(gflownet, create_env, config, untrained_gfn=None): 
    with gflownet.off_policy(): 
        env, marginal_log, log_rewards = sample_massive_batch(gflownet, create_env, 
                                                        num_batches=config.num_batches, 
                                                        sampler=untrained_gfn) 
    learned_dist, target_dist = marginal_dist(env, marginal_log, 
                                        log_rewards, dim=0) 
    l1 = (learned_dist - target_dist).abs().sum() 
    wandb.log({'l1': l1}) 

# Organize the experiments 
