from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
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

class RobomimicLowdimPolicy_flow_euler(BaseLowdimPolicy):
    def __init__(self, 
            action_dim, 
            obs_dim,
            algo_name='bc_rnn',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
            T=100
        ):
        super().__init__()
        # key for robomimic obs input
        # previously this is 'object', 'robot0_eef_pos' etc
        obs_key = 'obs'
        self.T=T
        self.TS=None
        self.DDPM='ODE'
        self.WSCALE=100
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)
        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]
        
        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={obs_key: [obs_dim]},
                ac_dim=action_dim,
                device='cpu',
            )
        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.obs_key = obs_key
        self.config = config

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self.normalizer['obs'].normalize(obs_dict['obs'])
        assert obs.shape[1] == 1
        robomimic_obs_dict = {self.obs_key: obs[:,0,:]}



        # naction = self.model.get_action(robomimic_obs_dict)
        cond_data=None
        cond_mask=None
        local_cond=None
        naction = self.conditional_sample(  
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=robomimic_obs_dict,
            DDPM=self.DDPM
            )
        

        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,DDPM='ODE',
            # keyword arguments to scheduler.step
            
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

    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def train_on_batch(self, batch, epoch, validate=False):
        nbatch = self.normalizer.normalize(batch)
        robomimic_batch = {
            'obs': {self.obs_key: nbatch['obs']},
            'actions': nbatch['action']
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info

    def get_optimizer(self):
        return self.model.optimizers['policy']
