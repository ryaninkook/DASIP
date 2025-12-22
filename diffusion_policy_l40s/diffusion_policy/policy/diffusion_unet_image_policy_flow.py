from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d_flow import ConditionalUnet1D_Flow
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
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


class DiffusionUnetImagePolicy_flow(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            T=None,
            path_type=None,
            TS=None,
            DDPM=None,
            WSCALE=None,
            diffusion_form=None,
            T_CHANGED=None,
            ODE_SDE=None,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.T=T
        self.T_CHANGED=T_CHANGED
        self.TS=TS
        self.DDPM=DDPM
        self.ODE_SDE=ODE_SDE
        self.WSCALE=WSCALE
        self.path_type=path_type
        self.diffusion_form=diffusion_form
        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D_Flow(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print("self.path_type: ",self.path_type)
        assert self.T is not None, "self.T is not defined. Please set self.T to either True or False."        
        assert self.T_CHANGED is not None, "self.T_CHANGED is not defined. Please set self.T to either True or False."        
        if self.path_type=='VP':
            print("policy flow euler. it is VP: self.path_type: ",self.path_type)
            prediction='velocity'
            loss_weight=None
            train_eps=1e-5
            sample_eps=1e-3 
            self.transport = create_transport(
                    self.path_type,
                    prediction,
                    loss_weight,
                    train_eps,
                    sample_eps
                )  # default: velocity; 
            self.transport_sampler = Sampler(self.transport)
        elif self.path_type=='GVP' or self.path_type=='Linear':
            print("policy flow euler. it is: self.path_type: ",self.path_type)
            prediction='velocity'
            loss_weight=None
            train_eps=0.0
            sample_eps=0.0
            self.transport = create_transport(
                    self.path_type,
                    prediction,
                    loss_weight,
                    train_eps,
                    sample_eps
                )  # default: velocity; 
            self.transport_sampler = Sampler(self.transport)
            

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            DDPM=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        x = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        TS=self.TS
        WSCALE=self.WSCALE
        dt = 1./TS
        eps=1e-6
        T=self.T
        device=self.device
        B=x.shape[0]
        # set step values


        if self.path_type=='Linear':
            if self.ODE_SDE=='ODE':

                # self.transport_sampler.t_minus_one=False
                # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
                sample_fn = self.transport_sampler.sample_ode(
                        sampling_method=DDPM, #Euler or Heun
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
                sample_fn = self.transport_sampler.sample_sde(
                        sampling_method=DDPM, #Euler or Heun
                        diffusion_form=self.diffusion_form,
                        diffusion_norm=1.0,
                        last_step=None, # Mean or None
                        last_step_size=None,
                        num_steps=self.TS,
                )

                sample_model_kwargs={
                    'global_cond':global_cond
                }
                model_fn=self.model.forward
                x = sample_fn(x, model_fn, **sample_model_kwargs)
                x=x[-1]


        elif self.path_type=='VP':
            if self.ODE_SDE=='ODE':

                # self.transport_sampler.t_minus_one=False
                # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
                sample_fn = self.transport_sampler.sample_ode(
                        sampling_method=DDPM, #Euler or Heun
                        num_steps=self.TS,
                )

                sample_model_kwargs={
                    'global_cond':global_cond
                }
                model_fn=self.model.forward
                x = sample_fn(x, model_fn, **sample_model_kwargs)
                x=x[-1]


            elif self.ODE_SDE=='SDE':
                
                # diffusion_form='SBDM' # SBDM # 'linear' 
                sample_fn = self.transport_sampler.sample_sde(
                        sampling_method=DDPM,
                        diffusion_form=self.diffusion_form,
                        diffusion_norm=1.0,
                        last_step=None, #"Mean",
                        last_step_size=None,
                        num_steps=self.TS,
                )

                sample_model_kwargs={
                    'global_cond':global_cond
                }
                model_fn=self.model.forward
                x = sample_fn(x, model_fn, **sample_model_kwargs)
                x=x[-1]

        elif self.path_type=='GVP':
            if self.ODE_SDE=='ODE':

                # self.transport_sampler.t_minus_one=False
                # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
                sample_fn = self.transport_sampler.sample_ode(
                        sampling_method=DDPM, #Euler or Heun
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
                sample_fn = self.transport_sampler.sample_sde(
                        sampling_method=DDPM, #Euler or Heun
                        diffusion_form=self.diffusion_form,
                        diffusion_norm=1.0,
                        last_step=None, # Mean or None
                        last_step_size=None,
                        num_steps=self.TS,
                )

                sample_model_kwargs={
                    'global_cond':global_cond
                }
                model_fn=self.model.forward
                x = sample_fn(x, model_fn, **sample_model_kwargs)
                x=x[-1]
        else:
            raise TypeError(f"error aa11 .. Invalid path_type: {self.path_type}")


        # finally make sure conditioning is enforced

        x[condition_mask] = condition_data[condition_mask]        

        return x

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image



        t = torch.rand((batch_size,), device=nactions.device)
        t=t.unsqueeze(-1).unsqueeze(-1)


        if self.path_type=='Linear':
            xt=noise*(1-t) + nactions*(t)

            # apply conditioning
            xt[condition_mask] = trajectory[condition_mask]

            recon_true=nactions-noise
            if self.T_CHANGED:
                t=t
            else:
                t=t*int(self.T-1)
            

            recon_hat=self.model(xt,t,global_cond)

            mse_loss = F.mse_loss(recon_hat, recon_true, reduction='none')
            mse_loss = reduce(mse_loss, 'b ... -> b (...)', 'mean').mean()
            mse_loss
        elif self.path_type=='VP' or self.path_type=='GVP'  :

            t, xt, ut=self.transport.path_sampler.plan(t, noise, nactions)
            # t=t*int(self.T-1)
            ut_hat=self.model(xt,t,global_cond)
            mse_loss = F.mse_loss(ut_hat, ut, reduction='none')
            mse_loss = reduce(mse_loss, 'b ... -> b (...)', 'mean').mean()
            mse_loss

        return mse_loss
