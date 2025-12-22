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
class Flow_Euler_UnetImagePolicy(BaseImagePolicy):
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
            T=100,
            # parameters passed to step
            **kwargs):
        super().__init__()
        self.T=T
        self.TS=None
        self.DDPM='ODE'
        self.WSCALE=100
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
        self.grid=False
        print("self.dtype: ",self.dtype)
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,DDPM='ODE',
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        B=condition_data.shape[0]
        x = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        # scheduler.set_timesteps(self.num_inference_steps)
        # TS= self.num_inference_steps
        TS=self.TS
        WSCALE=self.WSCALE
        dt = 1./TS
        eps=1e-6
        T=self.T
        device=self.device
                

        def sample_score_ddpm(x,global_cond,cond_mask,cond_data):


            betas = betas_for_alpha_bar(self.T)
            alphas = 1.0 - betas
            one = torch.tensor(1.0)
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alpha_prod_t = alphas_cumprod[t]
            # alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else one
            beta_prod_t = 1 - alpha_prod_t
            # beta_prod_t_prev = 1 - alpha_prod_t_prev
            # current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            # current_beta_t = 1 - current_alpha_t

            # x=torch.randn_like(a_true)
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.TS)
            # for t in reversed(range(TS)):
            for t in scheduler.timesteps:
                t=t.to(device)
                float_t=(t/TS)
                near_integar_t=(t/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = t

                beta=betas[t]
                z=torch.randn_like(x)
                noise_hat=self.model(x,t,global_cond)
                x=1/torch.sqrt(1-beta)*(x+beta*(1/torch.sqrt(1-alpha_prod_t))*noise_hat)+torch.sqrt(beta)*z
                x=x.detach()

            return  x

        def ddpm_sample(x, global_cond,cond_mask,cond_data):


            # Beta schedule (variance schedule for noise)
            beta = torch.linspace(0.0001, 0.02, TS, device=device)
            beta=reversed(beta)
            alpha = 1 - beta
            alpha_cumprod = torch.cumprod(alpha, dim=0)
            sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
            
            # Reverse loop for DDPM sampling (from T-1 to 0)
            for i in range(TS):
                float_t=(i/TS)
                near_integar_t=(i/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = vector_ones * near_integar_t
                
                # Model predicts the noise (eps_theta) in the current noisy image
                eps_theta = self.model(x, input_t, global_cond)
                
                # Compute the mean (mu) for the reverse process
                mu = (1 / torch.sqrt(alpha[i])) * (x - (beta[i] / sqrt_one_minus_alpha_cumprod[i]) * eps_theta)
                
                if i > 0:
                    # Add noise if it's not the final step
                    z = torch.randn_like(x)
                    sigma = torch.sqrt(beta[i])  # Standard deviation for noise
                    x = mu + sigma * z  # DDPM update: deterministic part + noise
                else:
                    # No noise is added on the final step
                    x = mu
                
                x = x.detach()  # Detach to avoid gradients in this sampling process
            
            return x
        def sample_ode(x,global_cond,cond_mask=None,cond_data=None):
            T= self.T
            
            for i in range(TS):
                float_t=(i/TS)
                near_integar_t=(i/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = vector_ones * near_integar_t

                model_output = self.model(x, input_t, global_cond)
                x = x + model_output * dt 
                
                x = x.detach()  # Detach to avoid storing gradients in this sampling process
        
            return x
            
        def sample_sde(x,global_cond,cond_mask=None,cond_data=None):

            # Example of a variance schedule (adjust this as needed)
            beta = torch.linspace(0.02, 0.0001, TS, device=device)  # Adjust based on the model
            # beta=reversed(beta)
            sigma_schedule = torch.sqrt(beta)

            for i in range(TS):
                float_t=(i/TS)
                near_integar_t=(i/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = vector_ones * near_integar_t
                
                # Call your diffusion model for the denoised output (mean)
                model_output = self.model(x, input_t, global_cond)
                
                # Add noise according to the SDE process
                z = torch.randn_like(x)  # Random noise
                noise_scale = sigma_schedule[i] * torch.sqrt(torch.tensor(dt))  # Scale the noise by sigma and dt
                
                # Update step with both the model output and stochastic noise
                x = x + model_output * dt + noise_scale * z  # SDE update
                
                x = x.detach()  # Detach to avoid storing gradients in this sampling process
            
            return x

        def sample_sde_curve(x,global_cond,cond_mask=None,cond_data=None):

            # Example of a variance schedule (adjust this as needed)
            beta = torch.linspace(0.02, 0.0001, TS, device=device)  # Adjust based on the model
            # beta=reversed(beta)
            sigma_schedule = torch.sqrt(beta)

            for i in range(TS):
                float_t=(i/TS)
                near_integar_t=(i/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = vector_ones * near_integar_t

                num_t=(i+eps)/TS
                sigma_t = make_continuous_beta_def(num_t,WSCALE).to(device)
                # Call your diffusion model for the denoised output (mean)
                model_output = self.model(x, input_t, global_cond)
                
                # Add noise according to the SDE process
                z = torch.randn_like(x)  # Random noise
                noise_scale = sigma_t * torch.sqrt(torch.tensor(dt))  # Scale the noise by sigma and dt
                
                # Update step with both the model output and stochastic noise
                x = x + model_output * dt + noise_scale * z  # SDE update
                
                x = x.detach()  # Detach to avoid storing gradients in this sampling process
            
            return x

        def sample_rect_sde(x,global_cond,cond_mask=None,cond_data=None):

            # Example of a variance schedule (adjust this as needed)
            # beta = torch.linspace(0.02, 0.0001, TS, device=device)  # Adjust based on the model
            # beta=reversed(beta)
            # sigma_schedule = torch.sqrt(beta)
            # betas_def=make_continuous_beta_def()
            print("TS: ",TS)
            for i in range(TS):
                float_t=(i/TS)
                near_integar_t=(i/TS)*int(T-1)
                vector_ones = torch.ones(B, device=device) 
                vector_ones = vector_ones.unsqueeze(-1)
                
                input_t = vector_ones * near_integar_t
                
                # Call your diffusion model for the denoised output (mean)
                model_output = self.model(x, input_t, global_cond)
                
                # Add noise according to the SDE process
                # z = torch.randn_like(x)  # Random noise
                # noise_scale = sigma_schedule[i] * torch.sqrt(torch.tensor(dt))  # Scale the noise by sigma and dt
                noise_scale=1
                # # Update step with both the model output and stochastic noise
                # x = x + model_output * dt + noise_scale * z  # SDE update
                num_t=(i+eps)/TS
                # noise_scale=1
                
                sigma_t = make_continuous_beta_def(num_t,WSCALE).to(device)
                num_t=torch.tensor(num_t).to(device)
                VAR=(sigma_t**2)/(2*(noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * model_output - 0.5 * (2.-num_t)*x.detach().clone())
                # print("VAR: ",VAR)
                # print("model_output: ",model_output)
                pred_sigma = model_output + VAR
                # print("pred_sigma: ",pred_sigma)
                x = x.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
            
                x = x.detach()  # Detach to avoid storing gradients in this sampling process
            
            return x
        if DDPM=='DDPM':
            x=ddpm_sample(x,global_cond,condition_mask,condition_data)
        elif DDPM=='ODE':
            x=sample_ode(x,global_cond,condition_mask,condition_data)
        elif DDPM=='SDE':
            x=sample_sde(x,global_cond,condition_mask,condition_data)
        elif DDPM=='SDE_CURVE':
            x=sample_sde_curve(x,global_cond,condition_mask,condition_data)
        elif DDPM=='RECT':
            x=sample_rect_sde(x,global_cond,condition_mask,condition_data)
        elif DDPM=='SCORE_DDPM':
            x=sample_score_ddpm(x,global_cond,condition_mask,condition_data)
            
        

        # finally make sure conditioning is enforced
        x[condition_mask] = condition_data[condition_mask]        

        return x


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],a_true=None) -> Dict[str, torch.Tensor]:
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
            DDPM=self.DDPM,
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
        if self.grid:
            unnormalized_img=obs_dict['image']*255
            # unnormalized_a_true = obs_dict['action']
            if a_true is not None:
                unnormalized_a_true=a_true
            else:
                unnormalized_a_true=action_pred
            unnormalized_a_hat=action_pred
            unnormalized_agent_pos=obs_dict['agent_pos']
            self.draw_grid(unnormalized_img,unnormalized_a_true,unnormalized_a_hat,unnormalized_agent_pos)    
        return result
    def draw_grid(self,unnormalized_img,unnormalized_a_true,unnormalized_a_hat,unnormalized_agent_pos):

        title=''
        script_name='a8_6_euler4_W10_debug'
        epoch=0
        image_mse=1
        save_grid_v2(title,unnormalized_img,unnormalized_a_true,unnormalized_a_hat,unnormalized_agent_pos,script_name,epoch,image_mse)
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

        xt=noise*(1-t) + nactions*(t)

        loss_mask = ~condition_mask

        xt[condition_mask] = cond_data[condition_mask]  

        recon_true=nactions-noise

        t=t*int(self.T-1)
        t

        recon_hat=self.model(xt,t,global_cond)
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (bsz,), device=trajectory.device
        # ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # noisy_trajectory = self.noise_scheduler.add_noise(
            # trajectory, noise, timesteps)
        
        # compute loss mask
        # loss_mask = ~condition_mask

        # apply conditioning
        # noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        # pred = self.model(noisy_trajectory, timesteps, 
            # local_cond=local_cond, global_cond=global_cond)

        # pred_type = self.noise_scheduler.config.prediction_type 
        # if pred_type == 'epsilon':
        #     target = noise
        # elif pred_type == 'sample':
        #     target = trajectory
        # else:
        #     raise ValueError(f"Unsupported prediction type {pred_type}")

        # loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()

        mse_loss = F.mse_loss(recon_hat, recon_true, reduction='none')
        mse_loss = reduce(mse_loss, 'b ... -> b (...)', 'mean').mean()
        mse_loss
        return mse_loss
