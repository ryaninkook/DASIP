from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from SI_policy.model.common.normalizer import LinearNormalizer
from SI_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from SI_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from SI_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from SiT.models import SiT_models
from SiT.download import find_model
from SiT.transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from SiT.train_utils import parse_transport_args
import SiT.wandb_utils
class DiffusionTransformerLowdimPolicy_flow(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # parameters passed to step
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

            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond
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

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
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
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
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
            self.transport_sampler = Sampler(self.transport)
        elif self.path_type=='GVP' or self.path_type=='Linear':
            print("policy flow euler. it is GVP: self.path_type: ",self.path_type)
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
            self.transport_sampler = Sampler(self.transport)
            

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None, DDPM=None, global_cond=None,
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
        
        if self.ODE_SDE=='ODE':

            # self.transport_sampler.t_minus_one=False 
            # self.diffusion_form='increasing-decreasing' # SBDM # 'linear' 
            sample_fn = self.transport_sampler.sample_ode(
                    sampling_method=DDPM, #Euler or Heun
                    num_steps=self.TS,
            )
            sample_model_kwargs={
                'cond':cond
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
                    last_step=self.last_step, # Mean or None
                    last_step_size=self.last_step_size,
                    num_steps=self.TS,
            )

            sample_model_kwargs={
                'cond':cond
            }
            model_fn=self.model.forward
            x = sample_fn(x, model_fn, **sample_model_kwargs)
            x=x[-1]


        # finally make sure conditioning is enforced


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
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
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

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            DDPM=self.DDPM,

            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
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
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
