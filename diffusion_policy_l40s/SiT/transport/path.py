import torch as th
import numpy as np
from functools import partial

def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


#################### Coupling Plans ####################

class ICPlan:
    """Linear Coupling Plan"""
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1
    
    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": th.tensor(norm), # ryan added
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }
        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        
        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0
    
    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt
    
    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0
    
    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut
    

class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min 
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min


    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = th.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t
    
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2
    

class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
    
    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t
    
    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t
    
    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * th.tan(t * np.pi / 2))

import math

class VPCPlan_squaredcos_cap_v2(ICPlan): # ryan added
    """class for VP path flow matching using squaredcos_cap_v2 schedule"""
    def __init__(self):
        super().__init__()
        # Precompute bar_alpha(0)
        self.alpha0 = self.bar_alpha(th.tensor(0.0))

    def bar_alpha(self, t):
        # t is a tensor
        u = (math.pi/2.0)*((t + 0.008)/1.008)
        return th.cos(u)**2

    def compute_alpha_t(self, t):
        """
        alpha_t = sqrt(bar_alpha(t)/bar_alpha(0))
        d_alpha_t/dt is computed via chain rule
        """
        bar_alpha_t = self.bar_alpha(t)
        ratio = bar_alpha_t / self.alpha0
        alpha_t = th.sqrt(ratio)

        # Compute derivative of bar_alpha wrt t
        u = (math.pi/2.0)*((t + 0.008)/1.008)
        d_bar_alpha_dt = -th.sin(2*u)*(math.pi/(2*1.008))

        # d_alpha_t = (1/(2 sqrt(ratio))) * (1/alpha0)*d_bar_alpha_dt
        d_alpha_t = d_bar_alpha_dt/(2*self.alpha0*th.sqrt(ratio))
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """
        sigma_t = sqrt(1 - bar_alpha(t)/bar_alpha(0))
        d_sigma_t/dt similarly
        """
        bar_alpha_t = self.bar_alpha(t)
        ratio = bar_alpha_t / self.alpha0
        sigma_t = th.sqrt(1.0 - ratio)

        u = (math.pi/2.0)*((t + 0.008)/1.008)
        d_bar_alpha_dt = -th.sin(2*u)*(math.pi/(2*1.008))

        # d_sigma_t = -(d_bar_alpha_dt/(2 * alpha0 * sqrt(1 - ratio)))
        d_sigma_t = -(d_bar_alpha_dt/(2*self.alpha0*th.sqrt(1 - ratio)))
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        # d_alpha_t/alpha_t = derivative of log(alpha_t)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion


import math
import torch as th
import numpy as np

class GVPCPlan2(ICPlan): # ryan added
    def __init__(self, sigma=0.0):
        super().__init__(sigma)
        self.offset = 0.008
        self.scale = 1.008

    def transform_t(self, t):
        # Transform t into (t+0.008)/1.008
        return (t + self.offset) / self.scale

    def compute_alpha_t(self, t):
        """
        Compute coefficient of x1 with offset and scaling:
        alpha_t = sin(( (t+0.008)/1.008 ) * pi/2)
        
        d_alpha_t/dt = (d/dt'[alpha_t(t')]) * (dt'/dt)
                     = (pi/2 * cos(t'*pi/2)) * (1/scale)
        """
        t_transformed = self.transform_t(t)
        alpha_t = th.sin(t_transformed * np.pi / 2)
        d_alpha_t = (np.pi / 2) * th.cos(t_transformed * np.pi / 2) / self.scale
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """
        Compute coefficient of x0 with offset and scaling:
        sigma_t = cos(( (t+0.008)/1.008 ) * pi/2)
        
        d_sigma_t/dt = (d/dt'[sigma_t(t')]) * (dt'/dt)
                     = (-pi/2 * sin(t'*pi/2)) * (1/scale)
        """
        t_transformed = self.transform_t(t)
        sigma_t = th.cos(t_transformed * np.pi / 2)
        d_sigma_t = (-np.pi / 2) * th.sin(t_transformed * np.pi / 2) / self.scale
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        """
        Special purposed function for computing numerically stable d_alpha_t / alpha_t.

        Originally (without offset/scale):
        d_alpha_t / alpha_t = pi/(2 * tan(t*pi/2))

        With t' = (t+0.008)/1.008:
        d_alpha_t/dt = [pi/(2 * cos(t'*pi/2)/sin(t'*pi/2))] * (1/scale)
                     = (pi/(2 * tan(t'*pi/2))) * (1/scale)

        So we just plug in t_transformed and multiply by 1/scale.
        """
        t_transformed = self.transform_t(t)
        return (np.pi / (2 * th.tan(t_transformed * np.pi / 2))) * (1/self.scale)