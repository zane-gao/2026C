"""
PyTorch-based Bayesian model for DWTS elimination prediction.
Uses variational inference with reparameterization trick.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..types import TensorPack, WeekObs, RuleParams
from ..config import Config
from .priors import PriorConfig
from ..rules.torch_rules import compute_total_loglik


class DWTSModel(nn.Module):
    """
    Variational Bayesian model for DWTS elimination prediction.
    
    Latent variables:
    - beta: feature coefficients [D]
    - u_pro: professional dancer effects [n_pro]
    - v_celeb: celebrity effects [n_celeb]  
    - w_season: season effects [S]
    - gamma: judge score coefficient (scalar)
    - delta: time-varying ability [S, N, T]
    - sigma_delta: noise scale for delta
    
    Each latent variable has a mean and log-std (for reparameterization trick).
    """
    
    def __init__(
        self,
        tensors: TensorPack,
        prior_cfg: PriorConfig,
        device: torch.device = None
    ):
        super().__init__()
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prior_cfg = prior_cfg
        
        # Dimensions
        S, N, D = tensors.X.shape
        T = tensors.T_max
        n_pro = int(tensors.pro_id.max()) + 1
        n_celeb = int(tensors.celeb_id.max()) + 1
        
        self.S, self.N, self.D, self.T = S, N, D, T
        self.n_pro = n_pro
        self.n_celeb = n_celeb
        
        # Store data tensors
        self.register_buffer("X", torch.tensor(tensors.X, dtype=torch.float32))
        self.register_buffer("J_z", torch.tensor(tensors.J_z, dtype=torch.float32))
        self.register_buffer("active_mask", torch.tensor(tensors.active_mask, dtype=torch.float32))
        self.register_buffer("pro_id", torch.tensor(tensors.pro_id, dtype=torch.long))
        self.register_buffer("celeb_id", torch.tensor(tensors.celeb_id, dtype=torch.long))
        
        # Variational parameters (mean and log-std for each latent variable)
        # beta: [D]
        self.beta_mu = nn.Parameter(torch.zeros(D))
        self.beta_logstd = nn.Parameter(torch.zeros(D) - 1.0)
        
        # u_pro: [n_pro]
        self.u_pro_mu = nn.Parameter(torch.zeros(n_pro))
        self.u_pro_logstd = nn.Parameter(torch.zeros(n_pro) - 1.0)
        
        # v_celeb: [n_celeb]
        self.v_celeb_mu = nn.Parameter(torch.zeros(n_celeb))
        self.v_celeb_logstd = nn.Parameter(torch.zeros(n_celeb) - 1.0)
        
        # w_season: [S]
        self.w_season_mu = nn.Parameter(torch.zeros(S))
        self.w_season_logstd = nn.Parameter(torch.zeros(S) - 1.0)
        
        # gamma: scalar
        self.gamma_mu = nn.Parameter(torch.zeros(1))
        self.gamma_logstd = nn.Parameter(torch.zeros(1) - 1.0)
        
        # delta: [S, N, T] - time-varying effects
        self.delta_mu = nn.Parameter(torch.zeros(S, N, T) * 0.01)
        self.delta_logstd = nn.Parameter(torch.zeros(S, N, T) - 2.0)
        
        # sigma_delta (log-scale for positivity)
        self.log_sigma_delta = nn.Parameter(torch.zeros(1))
        
        self.to(self.device)
    
    def sample_latents(self, n_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Sample latent variables using reparameterization trick."""
        samples = {}
        
        # Beta
        eps = torch.randn(n_samples, self.D, device=self.device)
        samples["beta"] = self.beta_mu + torch.exp(self.beta_logstd) * eps
        
        # u_pro
        eps = torch.randn(n_samples, self.n_pro, device=self.device)
        samples["u_pro"] = self.u_pro_mu + torch.exp(self.u_pro_logstd) * eps
        
        # v_celeb
        eps = torch.randn(n_samples, self.n_celeb, device=self.device)
        samples["v_celeb"] = self.v_celeb_mu + torch.exp(self.v_celeb_logstd) * eps
        
        # w_season
        eps = torch.randn(n_samples, self.S, device=self.device)
        samples["w_season"] = self.w_season_mu + torch.exp(self.w_season_logstd) * eps
        
        # gamma
        eps = torch.randn(n_samples, 1, device=self.device)
        samples["gamma"] = self.gamma_mu + torch.exp(self.gamma_logstd) * eps
        
        # delta
        eps = torch.randn(n_samples, self.S, self.N, self.T, device=self.device)
        samples["delta"] = self.delta_mu + torch.exp(self.delta_logstd) * eps
        
        # sigma_delta
        samples["sigma_delta"] = torch.exp(self.log_sigma_delta)
        
        return samples
    
    def compute_theta(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute theta (latent ability) from sampled latent variables.
        
        Returns: [n_samples, S, T, N] tensor
        """
        n_samples = samples["beta"].shape[0]
        
        # mu: [n_samples, S, N]
        # X: [S, N, D], beta: [n_samples, D]
        mu = torch.einsum("snd,bd->bsn", self.X, samples["beta"])
        
        # Add random effects
        # u_pro: [n_samples, n_pro], pro_id: [S, N]
        u_effect = samples["u_pro"][:, self.pro_id]  # [n_samples, S, N]
        v_effect = samples["v_celeb"][:, self.celeb_id]  # [n_samples, S, N]
        w_effect = samples["w_season"][:, :, None]  # [n_samples, S, 1]
        
        mu = mu + u_effect + v_effect + w_effect
        
        # gamma * J_z: [n_samples, S, T, N]
        # J_z: [S, T, N], gamma: [n_samples, 1]
        gamma_effect = samples["gamma"][:, :, None, None] * self.J_z[None, :, :, :]
        
        # delta: [n_samples, S, N, T] -> [n_samples, S, T, N]
        delta = samples["delta"].permute(0, 1, 3, 2)
        
        # theta before centering: [n_samples, S, T, N]
        theta = mu[:, :, None, :] + gamma_effect + delta
        
        # Center per (s, t) on active set
        # active_mask: [S, T, N]
        active = self.active_mask[None, :, :, :]  # [1, S, T, N]
        sum_theta = (theta * active).sum(dim=-1, keepdim=True)
        count = active.sum(dim=-1, keepdim=True) + 1e-12
        mean_theta = sum_theta / count
        theta_centered = theta - mean_theta
        
        return theta_centered  # [n_samples, S, T, N]
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational posterior and prior.
        KL(q||p) for Gaussian: 0.5 * (sigma_q^2/sigma_p^2 + (mu_p-mu_q)^2/sigma_p^2 - 1 + log(sigma_p^2/sigma_q^2))
        
        For standard normal prior (mu_p=0, sigma_p=prior_sigma):
        KL = 0.5 * (exp(2*logstd)/prior_sigma^2 + mu^2/prior_sigma^2 - 1 + log(prior_sigma^2) - 2*logstd)
        """
        kl = 0.0
        
        # Helper function for KL of single Gaussian
        def kl_normal(mu, logstd, prior_sigma):
            var_q = torch.exp(2 * logstd)
            var_p = prior_sigma ** 2
            return 0.5 * (var_q / var_p + mu ** 2 / var_p - 1 + np.log(var_p) - 2 * logstd).sum()
        
        kl += kl_normal(self.beta_mu, self.beta_logstd, self.prior_cfg.beta_sigma)
        kl += kl_normal(self.u_pro_mu, self.u_pro_logstd, self.prior_cfg.u_sigma)
        kl += kl_normal(self.v_celeb_mu, self.v_celeb_logstd, self.prior_cfg.v_sigma)
        kl += kl_normal(self.w_season_mu, self.w_season_logstd, self.prior_cfg.w_sigma)
        kl += kl_normal(self.gamma_mu, self.gamma_logstd, self.prior_cfg.gamma_sigma)
        
        # delta: use sigma_delta as prior std
        sigma_delta = torch.exp(self.log_sigma_delta).item()
        kl += kl_normal(self.delta_mu, self.delta_logstd, sigma_delta)
        
        # KL for sigma_delta (HalfNormal prior approximated as LogNormal)
        # Simplified: just add small regularization
        kl += 0.5 * (self.log_sigma_delta ** 2).sum()
        
        return kl
    
    def elbo(
        self,
        week_obs: List[WeekObs],
        season_idx: Dict[int, int],
        params: RuleParams,
        n_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO).
        
        ELBO = E_q[log p(data|z)] - KL(q||p)
        
        Returns: (elbo, log_likelihood, kl_divergence)
        """
        # Sample latents
        samples = self.sample_latents(n_samples)
        
        # Compute theta
        theta = self.compute_theta(samples)  # [n_samples, S, T, N]
        
        # Compute log-likelihood (average over samples)
        log_lik = 0.0
        for i in range(n_samples):
            theta_i = theta[i]  # [S, T, N]
            log_lik = log_lik + compute_total_loglik(theta_i, week_obs, season_idx, params)
        log_lik = log_lik / n_samples
        
        # Compute KL divergence
        kl = self.kl_divergence()
        
        # ELBO
        elbo = log_lik - kl
        
        return elbo, log_lik, kl
    
    def get_posterior_mean_theta(self) -> torch.Tensor:
        """Get posterior mean of theta (for prediction)."""
        samples = {
            "beta": self.beta_mu.unsqueeze(0),
            "u_pro": self.u_pro_mu.unsqueeze(0),
            "v_celeb": self.v_celeb_mu.unsqueeze(0),
            "w_season": self.w_season_mu.unsqueeze(0),
            "gamma": self.gamma_mu.unsqueeze(0),
            "delta": self.delta_mu.unsqueeze(0),
            "sigma_delta": torch.exp(self.log_sigma_delta),
        }
        return self.compute_theta(samples).squeeze(0)  # [S, T, N]


def build_torch_model(tensors: TensorPack, cfg: Config, device: torch.device = None) -> DWTSModel:
    """Build PyTorch model from tensors and config."""
    prior_cfg = PriorConfig()
    return DWTSModel(tensors, prior_cfg, device=device)
