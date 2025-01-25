import torch 
import numpy as np 

def batch_masked_regression(covariates, targets, mask, lamb=1e-3):
    # Generate Penrose's matrix 
    num_nodes = covariates.shape[2]
    batch_size = covariates.shape[0]
    X = covariates * mask[:, None, :] 
    # regularize 
    reg = lamb * torch.eye(num_nodes, device=covariates.device).expand(batch_size, num_nodes, num_nodes) 

    reg_params, _, _, _ = torch.linalg.lstsq(
            torch.bmm(torch.transpose(X, 1, 2), X) + reg,
            torch.transpose(X, 1, 2) @ targets
    ) 
    reg_estimates = torch.bmm(X, reg_params[..., None]).squeeze() 
    reg_residuals = (targets - reg_estimates) * (targets - reg_estimates) 
    return reg_params, reg_residuals.sum(dim=-1) 
    
def batch_ricf(data, parents_matrix, spouses_matrix): 
    # parents_matrix has shape `(num_batch, num_nodes, num_nodes)`
    num_nodes = parents_matrix.shape[1] 
    batch_size = parents_matrix.shape[0] 
    omega = torch.eye(num_nodes, device=data.device).repeat((batch_size, 1, 1)) 
    beta = torch.zeros((batch_size, num_nodes, num_nodes), device=data.device) 
    nodes = torch.arange(num_nodes, device=data.device) 
    for i in nodes: 
        complementary_vars = (nodes != i) 
        inx_mask = torch.logical_and(complementary_vars.unsqueeze(-1).expand(num_nodes, num_nodes), 
                                     complementary_vars.unsqueeze(0).expand(num_nodes, num_nodes))  
        omega_mimi = omega[:, inx_mask].reshape(batch_size, num_nodes - 1, num_nodes - 1) 
        # Compute the residual variables 
        residuals = torch.transpose(data - data @ beta, 1, 2)
        pseudo_vars, _, _, _ = torch.linalg.lstsq(omega_mimi, residuals[:, complementary_vars]) 
        pseudo_vars = torch.transpose(pseudo_vars, 1, 2) 
        # pseudo_vars = torch.transpose(torch.linalg.solve(omega_mimi, residuals[:, complementary_vars]), 1, 2)
        # Initially regress the observed variables for this feature on both the other features and the pseudo variables
        expanded_data = data.expand((batch_size, *data.shape)) 
        covariates = torch.cat([expanded_data[:, :, complementary_vars], pseudo_vars], dim=-1) 
        targets = data[:, i] 
        reg_mask = torch.cat([parents_matrix[:, complementary_vars, i], spouses_matrix[:, i, complementary_vars]], dim=-1) 
        reg_mask = reg_mask.to(dtype=covariates.dtype) 
        reg_params, reg_residuals = batch_masked_regression(covariates=covariates, targets=targets, mask=reg_mask)
        # Update the variables' using the regression's estimates 
        beta_i, omega_i = reg_params[:, :num_nodes - 1], reg_params[:, num_nodes - 1:] 
        omega[:, i, complementary_vars] = omega_i
        omega[:, complementary_vars, i] = omega_i 
        beta[:, complementary_vars, i] = beta_i 
        # Then compute the optimal adjusted correlation 
        adjusted_corr = reg_residuals / data.shape[0] 
        # And update the correlation matrix 
        omega_mi_inv = torch.linalg.solve(omega_mimi, omega[:, complementary_vars, i].unsqueeze(-1))
        omega[:, i, i] = adjusted_corr + torch.bmm(omega[:, i, complementary_vars].unsqueeze(1), omega_mi_inv).squeeze() 
    return beta, omega 


def fix_positive_definite_matrix(matrix, lamb=1e-10): 
    eigen = torch.linalg.eig(matrix) 
    eigenvalues = torch.real(eigen.eigenvalues) 
    eigenvectors = torch.real(eigen.eigenvectors) 
              
    eigenvalues = torch.where(eigenvalues < 0, lamb, eigenvalues) 
    return torch.bmm(
                eigenvectors, 
                torch.bmm(torch.diag_embed(eigenvalues, offset=0, dim1=-2, dim2=-1), 
                torch.transpose(eigenvectors, 1, 2))) 

def maximum_likelihood_batch_ricf(data, parents_matrix, spouses_matrix): 
    # `parents_matrix` and `spouses_matrix` have shape (batch_size, num_nodes, num_nodes) 
    beta, omega = batch_ricf(data, parents_matrix, spouses_matrix) 
    num_samples = data.shape[0] 
    batch_size = beta.shape[0] 
    num_nodes = beta.shape[1] 
    sigma = torch.linalg.inv(torch.eye(num_nodes, device=data.device).expand((batch_size, num_nodes, num_nodes)) - beta) 
    omega = fix_positive_definite_matrix(omega)  
    sigma = torch.transpose(sigma, 1, 2) @ omega @ sigma 
    
    expanded_data = data.expand((batch_size, *data.shape)) 
    loglikelihood = -torch.diagonal(torch.bmm(torch.bmm(torch.transpose(expanded_data, 1, 2), expanded_data), 
                                              torch.linalg.inv(sigma)), dim1=-2, dim2=-1).sum(dim=-1) - \
                    num_samples * torch.logdet(sigma) - num_samples * num_nodes * np.log(2 * torch.pi)
    return loglikelihood / 2

@torch.no_grad() 
def reward_log_likelihood(batch_states): 
    parents_matrix, spouses_matrix = batch_states.parents_and_spouses 
    return maximum_likelihood_batch_ricf(batch_states.data, parents_matrix, spouses_matrix) / batch_states.data.shape[0] 

@torch.no_grad() 
def reward_bic(batch_states, data, gamma=.5): 
    parents_matrix, spouses_matrix = batch_states.parents_and_spouses  
    num_edges = parents_matrix.sum(dim=(-1, -2)) + (spouses_matrix.sum(dim=(-1, -2))) / 2 
    num_nodes = parents_matrix.shape[1] 
    num_samples = data.shape[0] 
    max_loglikhood = maximum_likelihood_batch_ricf(data, parents_matrix, spouses_matrix) 
    bic = - 2 * max_loglikhood + num_edges * np.log(num_samples) + 4 * gamma * num_edges * np.log(num_nodes) 
    return - bic 