import numpy as np
import torch as th
import torch.nn as nn
from torchdiffeq import odeint
from functools import partial
from tqdm import tqdm

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        # if below is jsut float, make it tensor # ryan added
        # diffusion=th.tensor(diffusion).to(x) # ryan added
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""
        x = init
        mean_x = init 
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)

        return samples


# class sde_t_minus_one:
#     """SDE solver class"""
#     def __init__(
#         self, 
#         drift,
#         diffusion,
#         *,
#         t0,
#         t1,
#         num_steps,
#         sampler_type,
#     ):
#         assert t0 < t1, "SDE sampler has to be in forward time"

#         self.num_timesteps = num_steps
#         self.t = th.linspace(t0, t1, num_steps)
#         self.dt = self.t[1] - self.t[0]
#         self.drift = drift
#         self.diffusion = diffusion
#         self.sampler_type = sampler_type

#     def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
#         w_cur = th.randn(x.size()).to(x)
#         t = th.ones(x.size(0)).to(x) * (t-self.dt)
#         dw = w_cur * th.sqrt(self.dt)
#         drift = self.drift(x, t, model, **model_kwargs)
#         diffusion = self.diffusion(x, t)
#         mean_x = x + drift * self.dt
#         x = mean_x + th.sqrt(2 * diffusion) * dw
#         return x, mean_x
    
#     def __Heun_step(self, x, _, t, model, **model_kwargs):
#         w_cur = th.randn(x.size()).to(x)
#         dw = w_cur * th.sqrt(self.dt)
#         t_cur = th.ones(x.size(0)).to(x) * t
#         diffusion = self.diffusion(x, t_cur)
#         xhat = x + th.sqrt(2 * diffusion) * dw
#         K1 = self.drift(xhat, t_cur, model, **model_kwargs)
#         xp = xhat + self.dt * K1
#         K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
#         return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

#     def __forward_fn(self):
#         """TODO: generalize here by adding all private functions ending with steps to it"""
#         sampler_dict = {
#             "Euler": self.__Euler_Maruyama_step,
#             "Heun": self.__Heun_step,
#         }

#         try:
#             sampler = sampler_dict[self.sampler_type]
#         except:
#             raise NotImplementedError("Smapler type not implemented.")
    
#         return sampler

#     def sample(self, init, model, **model_kwargs):
#         """forward loop of sde"""
#         x = init
#         mean_x = init 
#         samples = []
#         sampler = self.__forward_fn()
#         for ti in self.t[:-1]:
#             with th.no_grad():
#                 x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
#                 samples.append(x)

#         return samples
from torch.distributions.beta import Beta

# class ode:
#     """ODE solver class"""
#     def __init__(
#         self,
#         drift,
#         *,
#         t0,
#         t1,
#         sampler_type,
#         num_steps,
#         atol,
#         rtol,
#         Ssam=None, # ryan added #"uniform",  # "uniform" or "beta"
#         beta_params=None, # ryan added #(1.5, 1.0),    # (alpha, beta_param)
#         beta_scaling=None, #ryan added 0.999,        # Scaling factor for Beta samples
#     ):

#         #assert Ssam is not None in below. how
#         assert Ssam is not None, "Ssam has to be specified"
#         assert beta_params is not None, "beta_params has to be specified"
#         assert beta_scaling is not None, "beta_scaling has to be specified"
        
#         assert t0 < t1, "ODE sampler has to be in forward time"
#         # beta_params = tuple(map(float, beta_params.split(',')))

#         self.Ssam = Ssam
#         self.beta_params = beta_params
#         self.beta_scaling = beta_scaling
#         self.drift = drift

#         if Ssam == "uniform":
#             self.t = th.linspace(t0, t1, num_steps)
#         elif Ssam == "beta":
#             self.t = self._generate_beta_grid(t0, t1, num_steps)

#         self.atol = atol
#         self.rtol = rtol
#         self.sampler_type = sampler_type

#     def sample(self, x, model, **model_kwargs):
        
#         device = x[0].device if isinstance(x, tuple) else x.device
#         def _fn(t, x):
#             t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
#             model_output = self.drift(x, t, model, **model_kwargs)
#             return model_output

#         t = self.t.to(device)
#         atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
#         rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
#         samples = odeint(
#             _fn,
#             x,
#             t,
#             method=self.sampler_type,
#             atol=atol,
#             rtol=rtol
#         )
#         return samples


#     def _generate_beta_grid(self, t0, t1, num_steps):
#         """Generate a Beta-distributed time grid."""
#         alpha, beta_param = self.beta_params
#         beta_dist = Beta(alpha, beta_param)
#         # Sample Beta-distributed values
#         beta_samples = beta_dist.sample((num_steps,))
#         # Scale and transform to the range [t0, t1]
#         scaled_samples = self.beta_scaling * (1 - beta_samples)
#         return t0 + (t1 - t0) * scaled_samples.sort().values  # Sort for increasing time

class ode:
    """ODE solver class supporting 'uniform', 'beta', 'cosine', 'exponential', and 'karras' schedules."""
    def __init__(
        self,
        drift,
        *,
        t0,
        t1,
        sampler_type,
        num_steps,
        atol,
        rtol,
        Ssam=None,         # e.g. "uniform", "beta", "cosine", "exponential", "karras"
        beta_params=None,  # used for 'beta' schedule or as optional param store
        beta_scaling=None, # used for 'beta' schedule or as optional param store
    ):

        # We'll reuse beta_params/beta_scaling to store or pass schedule parameters
        # or require that they're not None (to satisfy original checks).
        assert Ssam is not None, "Ssam has to be specified (uniform/beta/cosine/exponential/karras)."
        assert beta_params is not None, "beta_params must be specified (can store custom schedule params)."
        assert beta_scaling is not None, "beta_scaling must be specified (can store custom schedule params)."

        assert t0 < t1, "ODE sampler must have t0 < t1."
        
        self.Ssam = Ssam
        self.beta_params = beta_params    # could hold e.g. (alpha, beta) for 'beta' or (steepness,) for exponential
        self.beta_scaling = beta_scaling  # used in 'beta' or can hold e.g. rho for karras
        self.drift = drift
        
        # Decide how to create the time grid self.t:
        if Ssam == "uniform":
            self.t = th.linspace(t0, t1, num_steps)
        elif Ssam == "beta":
            self.t = self._generate_beta_grid(t0, t1, num_steps)
        elif Ssam == "cosine":
            self.t = self._generate_cosine_grid(t0, t1, num_steps)
        elif Ssam == "exponential":
            self.t = self._generate_exponential_grid(t0, t1, num_steps)
        elif Ssam == "karras":
            self.t = self._generate_karras_grid(t0, t1, num_steps)
        else:
            raise ValueError(f"Unknown Ssam mode: {Ssam}")

        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        """Integrate x from self.t[0] to self.t[-1] using odeint."""
        device = x[0].device if isinstance(x, tuple) else x.device
        
        def _fn(t, x):
            # Make time t a batch of identical scalars
            t_batch = (th.ones(x[0].size(0)) * t).to(device) if isinstance(x, tuple) \
                    else th.ones(x.size(0)).to(device) * t
                    #   else (th.ones(x.size(0)) * t).to(device)
                    
            return self.drift(x, t_batch, model, **model_kwargs)

        t = self.t.to(device)
        # Handle the possibility x is a tuple of states
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]

        samples = odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)
        return samples

    # def _generate_beta_grid(self, t0, t1, num_steps):
    #     """
    #     Generate a Beta-distributed time grid:
    #       alpha, beta_param = self.beta_params
    #       scaled by self.beta_scaling,
    #       then sorted so that it goes from t0 -> t1.
    #     """
    #     alpha, beta_param = self.beta_params
    #     beta_dist = Beta(alpha, beta_param)
    #     beta_samples = beta_dist.sample((num_steps,))  # in [0,1]
    #     scaled_samples = self.beta_scaling * (1 - beta_samples)  # emphasize near t0
    #     # Sorted ascending
    #     sorted_vals = scaled_samples.sort().values  
    #     return th.linspace(t0, t1, 1) if num_steps == 1 else t0 + (t1 - t0) * sorted_vals
    def _generate_beta_grid(self, t0, t1, num_steps):
        """
        Generate a Beta-distributed time grid that:
        1) Always starts at t0
        2) Always ends at t1
        3) Places (num_steps-2) points in-between, drawn from Beta(alpha, beta_param)
            and transformed by beta_scaling * (1 - sample)
        4) Sorts them, so t always goes from t0 to t1 in ascending order.
        """
        # Special cases:
        if num_steps == 1:
            # Only one point: just t0 (which is also t1).
            return th.tensor([t0])
        elif num_steps == 2:
            # Exactly two points: [t0, t1]
            return th.tensor([t0, t1])

        alpha, beta_param = self.beta_params
        beta_dist = Beta(alpha, beta_param)

        # Draw (num_steps-2) Beta samples in [0, 1].
        beta_samples = beta_dist.sample((num_steps - 2,))  # shape: [num_steps-2]

        # Transform them: scaled_samples in [0, self.beta_scaling]
        # because (1 - x) is in [0, 1], multiplied by beta_scaling
        scaled_samples = self.beta_scaling * (1.0 - beta_samples)

        # Sort ascending
        sorted_vals = scaled_samples.sort().values  # shape: [num_steps-2]

        # Concatenate [0.0, sorted_vals, 1.0] so we always include endpoints in [0,1].
        # This final_01 is length (num_steps).
        final_01 = th.cat([th.tensor([0.0]), sorted_vals, th.tensor([1.0])], dim=0)

        # Now map [0,1] -> [t0, t1]
        return t0 + (t1 - t0) * final_01
    def _generate_cosine_grid(self, t0, t1, num_steps):
        """
        A 'cosine' schedule from t0 to t1, starting exactly at t0 and
        ending at t1. We'll do the standard:
          t(i) = 0.5 * (1 - cos(pi * i/(N-1)))
        scaled to [t0, t1].
        """
        if num_steps == 1:
            return th.tensor([t0])
        i = th.linspace(0, num_steps-1, num_steps)
        # values in [0,1]
        vals_01 = 0.5 * (1 - th.cos(np.pi * i/(num_steps - 1)))
        # scale to [t0, t1]
        return t0 + (t1 - t0)*vals_01

    def _generate_exponential_grid(self, t0, t1, num_steps):
        """
        Exponential schedule from t0 to t1.
        We'll interpret beta_params[0] as 'steepness'.
        t(i) = [exp(steepness*i/(N-1)) - 1]/(exp(steepness) - 1),
        then scale to [t0, t1].
        """
        if num_steps == 1:
            return th.tensor([t0])
        steepness = self.beta_params[0]  # e.g. 5.0
        i = th.linspace(0, num_steps-1, num_steps)
        r = i / (num_steps - 1)  # goes from 0..1
        denom = np.exp(steepness) - 1.0
        vals_01 = (th.exp(steepness * r) - 1.0) / denom
        return t0 + (t1 - t0)*vals_01

    def _generate_karras_grid(self, t0, t1, num_steps):
        """
        Karras-like schedule from t0 to t1.
        We'll interpret beta_scaling as 'rho'.
        t(i) = [i/(N-1)]^(1/rho).
        """
        if num_steps == 1:
            return th.tensor([t0])
        rho = self.beta_scaling  # e.g. 7.0
        i = th.linspace(0, num_steps-1, num_steps)
        r = i / (num_steps - 1)
        vals_01 = r**(1.0 / rho)
        return t0 + (t1 - t0)*vals_01

