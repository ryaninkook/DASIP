from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from SI_policy.model.common.normalizer import LinearNormalizer
from SI_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from SI_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from SI_policy.model.diffusion.mask_generator import LowdimMaskGenerator
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

from SI_policy.model.diffusion.conditional_unet1d_flow import ConditionalUnet1D_Flow
from SI_policy.model.diffusion.conditional_unet1d_flow_map import ConditionalUnet1D_Flow_Map

class DiffusionUnetLowdimPolicy_flow_map(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D_Flow_Map,
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
            # parameters passed to step
            **kwargs):
        super().__init__()
        # self.path_type=''
        self.T=T
        self.TS=TS
        self.DDPM=DDPM
        self.WSCALE=WSCALE
        self.path_type=path_type
        self.diffusion_form=diffusion_form
        self.DT=0.001
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

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
        self.path_type=path_type
        if self.path_type=='VP':
            print("policy flow euler. it is VP: self.path_type: ",self.path_type)
            prediction='velocity'
            loss_weight=None
            train_eps=0.001
            sample_eps=0.001
            self.transport = create_transport(
                    self.path_type,
                    prediction,
                    loss_weight,
                    train_eps,
                    sample_eps
                )  # default: velocity; 
            self.transport_sampler = Sampler(self.transport)
        elif self.path_type=='GVP':
            print("policy flow euler. it is GVP: self.path_type: ",self.path_type)
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
            generator=None,DDPM=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        # model = self.model
        # scheduler = self.noise_scheduler

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
        if self.path_type=='direct_distill':
            x=self.sample(x,global_cond)

        elif self.path_type=='VP':
                        
            
            diffusion_form='SBDM' # SBDM # 'linear' 
            sample_fn = self.transport_sampler.sample_sde(
                    sampling_method="Euler",
                    diffusion_form=diffusion_form,
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
                        
            # increasing-decreasing    decreasing   sigma  linear  SBDM    constant
            self.transport_sampler.t_minus_one=True
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

    def compute_loss(self, batch,pretrained_ema_policy=None):
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
        device=nactions.device
        x0 = torch.randn(trajectory.shape, device=trajectory.device).to(device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        self.K=0.25
        # t = torch.rand((batch_size,), device=nactions.device)
        # t=t.unsqueeze(-1).unsqueeze(-1)
        B=batch_size
        
        B=batch_size
        s = torch.rand((B,), device=nactions.device).to(device)
        # lower_bound = torch.clamp(s - self.K, min=0)
        # upper_bound = torch.clamp(s + self.K, max=1)
        t = torch.rand((B,), device=nactions.device).to(device)

        # t = t * (upper_bound - lower_bound) + lower_bound
        t=t.unsqueeze(-1).unsqueeze(-1)
        s=s.unsqueeze(-1).unsqueeze(-1)
        if self.path_type=='direct_distill':
            xs=x0*(1-s) + nactions*(s)
            # loss_mask = ~condition_mask

            # xt[condition_mask] = cond_data[condition_mask]  

            # recon_true=a_true-noise

            # xt=xt.to(device)
            # t=t.to(device)
            # t = t.clone().detach().requires_grad_(True)
            # t=t*int(self.T-1)
            # s=s*int(self.T-1)
            # global map1
            x_st= (1-t+s)*xs +(t-s)*self.model(xs,s,global_cond,t)
            
            self.DT=0.001
            
            dt = torch.tensor(self.DT)

            # xs.requires_grad_(True)
            # t.requires_grad_(True)
            # t = t.clone().detach().requires_grad_(True)

            gradt=1/2*(self.model(xs,s,global_cond,t+dt)-self.model(xs,s,global_cond,t-dt))/self.DT
            # model_output = self.model(xs, s, global_cond, t)
            # gradt = torch.autograd.grad(outputs=model_output, inputs=t, grad_outputs=torch.ones_like(model_out), create_graph=True)[0]
            # gradt = torch.autograd.grad(
            #     outputs=model_output,
            #     inputs=t,
            #     grad_outputs=torch.ones_like(model_output),
            #     create_graph=True,
            #     retain_graph=True
            # )[0]

            dXst_dt=-xs+self.model(xs,s,global_cond,t)+(t-s)*gradt
            # global pretrained_ema_policy
            with torch.no_grad():
                dxst_dt_hat=pretrained_ema_policy.model(x_st,t*int(self.T-1),global_cond)

            mse_loss=F.mse_loss(dxst_dt_hat,dXst_dt)
            mse_loss=mse_loss.mean()
        # elif self.path_type=='map_train':

        #     xt=x0*(1-t) + nactions*(t)

        #     # loss_mask = ~condition_mask

        #     # xt[condition_mask] = cond_data[condition_mask]  

        #     dxt_dt=-x0+nactions
        #     x_ts= (1-s+t)*xt +(s-t)*self.model(xt,t,global_cond,s)
        #     # L2 loss

        #     dt = torch.tensor(self.DT)
        #     A=self.model(xt,t,global_cond,s+dt)-self.model(xt,t,global_cond,s-dt)
        #     grand_t=0.5*(A)/dt

        #     dx_ts_dt=-xt+self.model(xt,t,global_cond,s) +(t-s)*grand_t
        #     # loss1=dx_ts_dt - dxt_dt
        #     loss1 = ((dx_ts_dt - dxt_dt)**2).sum()
        #     x_st=(1-t+s)*x_ts+(t-s)*self.model(x_ts,s,global_cond,t)
        #     # loss2=x_st-xt
        #     loss2 = ((x_st - xt)**2).sum()

        #     mse_loss=loss1+loss2

        elif self.path_type=='VP' or self.path_type=='GVP'  :

            t, xt, ut=self.transport.path_sampler.plan(t, x0, nactions)
            t=t*int(self.T-1)
            ut_hat=self.model(xt,t,global_cond)
            mse_loss = F.mse_loss(ut_hat, ut, reduction='none')
            mse_loss = reduce(mse_loss, 'b ... -> b (...)', 'mean').mean()
            mse_loss

        return mse_loss

    def sample(self,x_start,global_cond):
        
        ts  = torch.linspace(0.0,1.0, self.TS + 1).to(self.device)
        self.eps=0.0001
        ts[0]=self.eps
        x=x_start
        # start=time.time()
        for i,t in enumerate(ts[:-1]):
            s = t.repeat(len(x)).to(self.device)
            t = ts[i+1].repeat(len(x))
            # print(s,t)
            t=t.unsqueeze(-1).unsqueeze(-1)
            s=s.unsqueeze(-1).unsqueeze(-1)
            x=(1-t+s)*x +(t-s)*self.model(x,s,global_cond,t)
            x=x.detach()
        # end=time.time()
        # time_took=end-start
        return x
