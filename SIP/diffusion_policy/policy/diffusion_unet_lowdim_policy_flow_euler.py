from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from import_true_basics import *


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
from torch.utils.data import DataLoader, random_split
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models import SiT_models
# from download import find_model
# from transport import create_transport_flow,Sampler_flow
from diffusers.models import AutoencoderKL

from SiT.models import SiT_models
from SiT.download import find_model
from SiT.transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from SiT.train_utils import parse_transport_args
import SiT.wandb_utils

import os
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import getpass

# from import_basics import *
import argparse
import torch,copy,time,datetime

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

# from models import SiT_models
# from download import find_model
# from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
# from train_utils import parse_transport_args
# import wandb_utils
# os.environ["WANDB_KEY"] = "9993390f8cecde45fb561fff55f5701bc969eb2d"
# os.environ["ENTITY"] = "diffusion"
# os.environ["PROJECT"] = "SiT_v2"

# export CUDA_LAUNCH_BLOCKING=1

# wandb_key = os.environ["WANDB_KEY"]
# entity = os.environ["ENTITY"]
# project = os.environ["PROJECT"]

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag




def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def image_encoding(vae,x):

    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    return x


def image_decoding(vae,samples):
    with torch.no_grad():
        samples = vae.decode(samples / 0.18215).sample
    return samples

def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)

def make_continuous_beta_def(input1,WSCALE):
    import torch
    import math

    # Original betas_for_alpha_bar function to generate betas
    def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    # Generate betas for 100 diffusion steps
    num_diffusion_timesteps = WSCALE
    betas = betas_for_alpha_bar(num_diffusion_timesteps)
    betas=reversed(betas)
    # Function to interpolate betas for a continuous x in [0, 1]
    def betas_def(x):
        # Ensure x is in the range [0, 1]
        x = torch.clamp(x, 0, 1)

        # Map x to an index in the range [0, num_diffusion_timesteps - 1]
        index = x * (num_diffusion_timesteps - 1)

        # Get the lower and upper indices for interpolation
        lower_index = torch.floor(index).long()
        upper_index = torch.ceil(index).long()

        # Handle boundary case where index is exactly at the upper limit
        if upper_index >= num_diffusion_timesteps:
            upper_index = num_diffusion_timesteps - 1
        if lower_index < 0:
            lower_index = 0

        # Interpolation between the two beta values
        beta_lower = betas[lower_index]
        beta_upper = betas[upper_index]

        # Linear interpolation based on the fractional part of the index
        fractional_part = index - lower_index
        beta_interp = beta_lower + fractional_part * (beta_upper - beta_lower)
        
        return beta_interp

    # Example: Using betas_def for a value of x in the range [0, 1]
    x = torch.tensor([input1])  # Example input, can be any value in [0, 1]
    beta_value = betas_def(x)
    # print(beta_value)

    import matplotlib.pyplot as plt

    # plt.plot(betas)
    # plt.xlabel('')  
    # plt.ylabel('')  
    # plt.title('')  
    # plt.show()
    return beta_value

from diffusion_policy.model.diffusion.conditional_unet1d_flow import ConditionalUnet1D_Flow

class DiffusionUnetLowdimPolicy_flow_euler(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D_Flow,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            T=None,
            path_type=None,
            TS=None,
            DDPM=None,
            WSCALE=None,
            diffusion_form=None,
            T_CHANGED=None,
            ODE_SDE=None,
            last_step=3,
            last_step_size='default',
            prediction=None,
            train_eps=None,
            sample_eps=None,
            Tsam=None,
            Ssam=None,
            Tdiscrete=None,
            beta_params=None,
            beta_scaling=None,
            TS_list=None,
            # parameters passed to step
            **kwargs):
        super().__init__()
        # self.path_type=''
        self.T=T
        self.T_CHANGED=T_CHANGED
        self.TS=TS
        self.DDPM=DDPM
        self.ODE_SDE=ODE_SDE
        self.WSCALE=WSCALE
        self.path_type=path_type
        self.diffusion_form=diffusion_form
        self.last_step=last_step
        self.last_step_size=last_step_size
        self.prediction=prediction
        self.train_eps=train_eps
        self.sample_eps=sample_eps
        self.Tsam=Tsam
        self.Tdiscrete=Tdiscrete
        self.TS_list=TS_list
        # input(f'policy_flow_euler next? {self.path_type}')
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        # if num_inference_steps is None:
        #     num_inference_steps = noise_scheduler.config.num_train_timesteps
        # self.num_inference_steps = num_inference_steps
    
        self.path_type=path_type
        '''
            
        if (path_type in [PathType.VP]):
            train_eps = 1e-5 if train_eps is None else train_eps
            sample_eps = 1e-3 if train_eps is None else sample_eps
        elif (path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY):
            train_eps = 1e-3 if train_eps is None else train_eps
            sample_eps = 1e-3 if train_eps is None else sample_eps
        else: # velocity & [GVP, LINEAR] is stable everywhere
            train_eps = 0
            sample_eps = 0
        
        '''
        print("self.path_type: ",self.path_type)
        assert self.T is not None, "self.T is not defined. Please set self.T to either True or False."        
        assert self.T_CHANGED is not None, "self.T_CHANGED is not defined. Please set self.T to either True or False."        
        if self.path_type=='VP':
            print("policy flow euler. it is VP: self.path_type: ",self.path_type)
            
            loss_weight=None
            # train_eps=1e-5
            # sample_eps=1e-3 
            self.transport = create_transport(
                    self.path_type,
                    self.prediction,
                    loss_weight,
                    self.train_eps,
                    self.sample_eps
                )  # default: velocity; 
            self.transport_sampler = Sampler(self.transport,Ssam, beta_params, beta_scaling)
        elif self.path_type=='GVP' or self.path_type=='Linear' or self.path_type=='GVP2':
            print("policy flow euler. it is : self.path_type: ",self.path_type)
            loss_weight=None
            # train_eps=0.0
            # sample_eps=0.0
            # sample_eps=1e-3 # ryan added
            self.transport = create_transport(
                    self.path_type,
                    self.prediction,
                    loss_weight,
                    self.train_eps,
                    self.sample_eps
                )  # default: velocity; 
            self.transport_sampler = Sampler(self.transport,Ssam, beta_params, beta_scaling)
            
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,DDPM=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        x = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        if self.ODE_SDE=='ODE':

            # self.transport_sampler.t_minus_one=False
            # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
            sample_fn = self.transport_sampler.sample_ode(
                    sampling_method=DDPM.lower(), #Euler or Heun
                    num_steps=self.TS,
            )

            sample_model_kwargs={
                'global_cond':global_cond
            }
            model_fn=self.model.forward
            x = sample_fn(x, model_fn, **sample_model_kwargs)
            x=x[-1]


        elif self.ODE_SDE=='SDE':
            # increasing-decreasing    decreasing   sigma  linear  SBDM    constant
            # self.transport_sampler.t_minus_one=True
            # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
            # print("self.last_step: ",self.last_step)
            # check below if last step is either Mean, Euler,Tweedie
            if self.last_step not in ['Mean', 'Euler', 'Tweedie']:
                raise ValueError("Invalid value for last_step. Choose from 'Mean', 'Euler', 'Tweedie', or None.")
            if self.diffusion_form not in ['increasing-decreasing',    'decreasing',   'sigma'  ,'linear',  'SBDM'   , 'constant']:
                raise ValueError("Invalid diffusion form . error")

            # input('next?')
            sample_fn = self.transport_sampler.sample_sde(
                    sampling_method=DDPM, #Euler or Heun
                    diffusion_form=self.diffusion_form,
                    diffusion_norm=1.0,
                    last_step=self.last_step, # Mean or None
                    last_step_size=self.last_step_size,
                    num_steps=self.TS,
            )

            sample_model_kwargs={
                'global_cond':global_cond
            }
            model_fn=self.model.forward
            x = sample_fn(x, model_fn, **sample_model_kwargs)
            x=x[-1]
        # finally make sure conditioning is enforced


        x[condition_mask] = condition_data[condition_mask]        

        return x


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True
        # print("shape: ",shape)
        # run sampling
        # print("self.kwargs: ",**self.kwargs)
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            DDPM=self.DDPM,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        # nactions = self.normalizer['action'].normalize(batch['action'])
        nactions=action
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        if self.Tsam=='vp_like':
            # import torch

            # Define the VP beta function
            sigma_min, sigma_max = 0.1, 20.0
            beta_vp = lambda t: sigma_min + (1 - t) * (sigma_max - sigma_min)

            # Define the integral of beta for VP
            def integral_beta_vp(t):
                """Compute the integral of beta_vp from 0 to t."""
                # Using a numerical approximation for integration in PyTorch
                return torch.cumsum(beta_vp(t) * (t[1] - t[0]), dim=0)

            # Define the transformation for sampling
            def torch_rand_vp_like(batch_size, device):
                """Sample t values based on the VP schedule."""
                t_uniform = torch.rand((batch_size,), device=device)  # Uniform [0, 1]
                t_uniform_sorted, _ = torch.sort(t_uniform)  # Sorting for numerical stability

                # Numerically integrate beta_vp to create a mapping
                t_sampled = -0.5 * torch.cumsum(beta_vp(t_uniform_sorted), dim=0)

                return t_sampled  # Match shape with your earlier code

            # Example usage
            # batch_size = 10
            # device = "cuda" if torch.cuda.is_available() else "cpu"

            # Generate VP-like sampled t
            t = torch_rand_vp_like(batch_size, nactions.device)
            # t=t.unsqueeze(-1).unsqueeze(-1)
            if t.dim() == 3:
                # For size = x, x, x
                t = t.unsqueeze(-1).unsqueeze(-1)
            elif t.dim() == 2:
                # For size = x, x
                t = t.unsqueeze(-1)
            # print(t_vp)
        elif self.Tsam=='beta':
            from torch.distributions.beta import Beta

            # Parameters for Beta distribution and scaling
            s = 0.999
            alpha = 1.5
            beta_param = 1.0

            # Function to generate beta-like sampling in PyTorch
            def torch_rand_beta_like(batch_size, device="cpu"):
                # Define Beta distribution
                beta_dist = Beta(alpha, beta_param)
                # Sample from Beta distribution
                sampled_values = beta_dist.sample((batch_size,))
                # Transform to τ space
                tau_samples = s * (1 - sampled_values)
                return tau_samples.to(device)

            # Example usage
            # batch_size = 10000
            t = torch_rand_beta_like(batch_size, nactions.device)
            if self.Tdiscrete=='Tdiscrete':
                t=t*int(self.T-1)
                t=t.long()
                t=t/float(self.T-1) #
            # t=t.unsqueeze(-1).unsqueeze(-1)
            if t.dim() == 3:
                # For size = x, x, x
                t = t.unsqueeze(-1).unsqueeze(-1)
            elif t.dim() == 2:
                # For size = x, x
                t = t.unsqueeze(-1)
        elif self.Tsam=='uniform':
            if self.Tdiscrete=='Tdiscrete':
                t = torch.randint(0, self.T, (batch_size,), device=nactions.device)
                t=t/float(self.T-1) #
            elif self.Tdiscrete=='Tconti':
                t = torch.rand((batch_size,), device=nactions.device)
            # t=t.unsqueeze(-1).unsqueeze(-1)
            if t.dim() == 3:
                # For size = x, x, x
                t = t.unsqueeze(-1).unsqueeze(-1)
            elif t.dim() == 2:
                # For size = x, x
                t = t.unsqueeze(-1)
            

        if not self.T_CHANGED:
            raise ValueError('t change is not true. error.')

        t, xt, ut=self.transport.path_sampler.plan(t, noise, nactions)
        # t=t*int(self.T-1)
        ut_hat=self.model(xt,t,global_cond)
        mse_loss = F.mse_loss(ut_hat, ut, reduction='none')
        mse_loss = reduce(mse_loss, 'b ... -> b (...)', 'mean').mean()
        mse_loss

        return mse_loss
