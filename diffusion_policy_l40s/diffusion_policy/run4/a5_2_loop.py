#%%
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy_flow_euler
from diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler import DiffusionUnetLowdimPolicy_flow_euler
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy_ddim import DiffusionUnetLowdimPolicy_ddim
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
# from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel


import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from import_true_basics import *

OmegaConf.register_new_resolver("eval", eval, replace=True)

scriptname = os.path.basename(__file__).replace('.py', '')
default_timestamp = timestamp()

def save_video(runner_log,local_epoch_idx,output_dir_media):
    # global local_epoch_idx
    # Loop through the videos and rename them based on the rewards
    for i in range(6):
        # Handle different keys for train and test videos
        if i < 2:
            video_key = f'train/sim_video_{i}'
            reward_key = f'train/sim_max_reward_{i}'
        else:
            video_key = f'test/sim_video_10000{i-2}'
            reward_key = f'test/sim_max_reward_10000{i-2}'
        try:
            # Get the max reward and round it
            max_reward = runner_log[reward_key]
            max_reward = round(max_reward, 2)

            # Get the original video path
            video_path = runner_log[video_key]._path

            # Create the new video name with the reward
            new1 = video_path.replace('.mp4', '') + f'_{max_reward}.mp4'
            new_name = new1.split('/')[-1]
            new_name=f'{local_epoch_idx}_{new_name}'
            # Move the video to the new directory
            shutil.move(video_path, os.path.join(output_dir_media, new_name))

            # Print the new path for debugging
            # print(f"Moved video to: {os.path.join(output_dir_media, new_name)}")
        except Exception as e:
            print('error save video,',i,e)
    # Save the runner_log to a JSON file in the new directory
    log_path = os.path.join(output_dir_media, 'log',f'{local_epoch_idx}_log.json')
    with open(log_path, 'w') as json_file:
        json.dump(runner_log, json_file, default=str)  # default=str to handle non-serializable objects like WandB objects

# %%


config_path = 'diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml'
cfg = OmegaConf.load(config_path)

cfg_task_path='diffusion_policy/config/task/square_lowdim_abs.yaml'
cfg_task=OmegaConf.load(cfg_task_path)
cfg.task=cfg_task
# 2) Apply all the CLI-style overrides in Python.

# Original CLI arguments: 
# task=square_lowdim_abs
# logging.name=square_lowdim_abs
# logging.project=classifier_infer_ode2_v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M
# cfg.task.name = 'square_lowdim_abs'
cfg.logging.name = 'square_lowdim_abs'
cfg.logging.project = 'classifier_infer_ode2_v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M'

# +training.architecture=unet
cfg.training.architecture = 'unet'

# dataloader.num_workers=0
cfg.dataloader.num_workers = 0

# +training.save_50=True
cfg.training.save_50 = True

# +training.save_from=0.0
cfg.training.save_from = 0.0

# policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler
cfg.policy._target_ = 'diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler'

# policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow
cfg.policy.model._target_ = 'diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow'

# +training.save_path=v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M
cfg.training.save_path = 'v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M'

# training.resume=False
cfg.training.resume = False

# +policy.path_type=GVP2
cfg.policy.path_type = 'GVP2'

# +policy.ODE_SDE=ODE
cfg.policy.ODE_SDE = 'ODE'

# +policy.DDPM=Euler
cfg.policy.DDPM = 'Euler'

# +policy.diffusion_form=.
cfg.policy.diffusion_form = '.'

# +policy.WSCALE=1000
cfg.policy.WSCALE = 1000

# +policy.T=100
# +policy.model.T=100
cfg.policy.T = 100
cfg.policy.model.T = 100

# +policy.T_CHANGED=True
# +policy.model.T_CHANGED=True
cfg.policy.T_CHANGED = True
cfg.policy.model.T_CHANGED = True

# +policy.TS=2
cfg.policy.TS = 10

# +policy.last_step=.
# +policy.last_step_size=.
cfg.policy.last_step = '.'
cfg.policy.last_step_size = '.'

# +policy.train_eps=0
# +policy.sample_eps=0
cfg.policy.train_eps = 0
cfg.policy.sample_eps = 0

# +policy.prediction=velocity
cfg.policy.prediction = 'velocity'

# +policy.Tsam=uniform
# +policy.Ssam=uniform
cfg.policy.Tsam = 'uniform'
cfg.policy.Ssam = 'uniform'

# +policy.Tdiscrete=Tdiscrete
cfg.policy.Tdiscrete = 'Tdiscrete'

# +policy.model.t_minus_one=t_minus_one
cfg.policy.model.t_minus_one = True

# training.seed=1000000
cfg.training.seed = 1000000

# training.num_epochs=5000
cfg.training.num_epochs = 5000

# +policy.beta_params=_
# +policy.beta_scaling=_
cfg.policy.beta_params = '_'
cfg.policy.beta_scaling = '_'

# +training.training=False
cfg.training.training = False

# +training.infer_from=0
cfg.training.infer_from = 0


# class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    # include_keys = ['global_step', 'epoch']

# def __init__(self, cfg: OmegaConf, output_dir=None):
if cfg.training.training:
    if cfg.policy.path_type=='DDPM' or cfg.policy.path_type=='DDIM':
        cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}'
    else:
        cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_Teps{cfg.policy.train_eps}_Tsam{cfg.policy.Tsam}_T{cfg.policy.T}'
else:
    if cfg.policy.path_type=='DDPM' or cfg.policy.path_type=='DDIM':
        cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_TS{cfg.policy.TS}'
    else:
        if cfg.policy.ODE_SDE=='ODE':
            cfg.logging.name=f'_TS{cfg.policy.TS}_{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_Teps{cfg.policy.train_eps}_Seps{cfg.policy.sample_eps}_Ssam{cfg.policy.Ssam}_{cfg.policy.beta_params}_{cfg.policy.beta_scaling}_TS{cfg.policy.TS}'
        elif cfg.policy.ODE_SDE=='SDE':
            cfg.logging.name=f'{cfg.logging.name}_{cfg.training.architecture}_{cfg.policy.DDPM}_{cfg.policy.path_type}_{cfg.policy.prediction}_{cfg.policy.ODE_SDE}_Teps{cfg.policy.train_eps}_Seps{cfg.policy.sample_eps}_{cfg.policy.diffusion_form}_{cfg.policy.last_step}_{cfg.policy.last_step_size}_Ssam{cfg.policy.Ssam}_{cfg.policy.beta_params}_{cfg.policy.beta_scaling}_TS{cfg.policy.TS}'
# if output_dir is None:
# if cfg.training.training:
if len(str(cfg.training.save_path))<5:
    raise TypeError('wrong save path')

output_dir = os.path.join('data', 'output4_train', cfg.task.name , cfg.training.save_path)
video_dir=copy.deepcopy(output_dir)
if not cfg.training.training:
    checkpoint_dir=os.path.join(output_dir,'checkpoints')
    timedelta1=0
    timedelta2=100
    today1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1)).strftime("%Y%m%d_%H%M%S"))
    today2=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1))).timestamp()
    past1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta2)).strftime("%Y-%m-%d"))
    past2=datetime.strptime('20210218', "%Y%m%d").timestamp()
    output_dir=os.path.join(output_dir,'media',f'{today1}_{cfg.logging.name}')
print("save path called! output_dir: ",output_dir)
time.sleep(3)

# else:
# output_dir = os.path.join('data', 'output4_infer',scriptname, cfg.task.name, f'{default_timestamp}_{cfg.logging.name}')
os.makedirs(output_dir, exist_ok=True)

# fol1=f'data/output2/{scriptname}/media'
# output_dir
# self.output_dir=output_dir
# if cfg.training.training:
output_dir_media = os.path.join(output_dir,'media')
os.makedirs(output_dir_media, exist_ok=True)
os.makedirs(os.path.join(output_dir_media,'log'), exist_ok=True)

# output_dir_media = os.path.join(output_dir,'media',f'{default_timestamp}_{cfg.logging.name}')
# os.makedirs(output_dir_media, exist_ok=True)

os.makedirs(os.path.join(output_dir_media,'log'), exist_ok=True)
output_dir_media=output_dir_media

# super().__init__(cfg, output_dir=output_dir)

# set seed
seed = cfg.training.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# configure model

# configure model
if cfg.policy.path_type=='DDPM':
    model: DiffusionUnetLowdimPolicy
    ema_model: DiffusionUnetLowdimPolicy = None
elif cfg.policy.path_type=='DDIM':
    model: DiffusionUnetLowdimPolicy_ddim
        
    OmegaConf.set_struct(cfg.policy.noise_scheduler, False)

    if 'variance_type' in cfg.policy.noise_scheduler:
        cfg.policy.noise_scheduler.pop('variance_type')

    # Re-enable struct mode
    OmegaConf.set_struct(cfg.policy.noise_scheduler, True)

    cfg.policy.noise_scheduler._target_= 'diffusers.schedulers.scheduling_ddim.DDIMScheduler'
    cfg.policy.noise_scheduler.num_train_timesteps=100
    cfg.policy.noise_scheduler.beta_start=0.0001
    cfg.policy.noise_scheduler.beta_end=0.02
    cfg.policy.noise_scheduler.beta_schedule='squaredcos_cap_v2'
    cfg.policy.noise_scheduler.clip_sample=True

    model = hydra.utils.instantiate(cfg.policy)
    print("model.noise_scheduler: ",model.noise_scheduler)
    model.num_inference_steps=cfg.policy.TS


    ema_model: DiffusionUnetLowdimPolicy_ddim = None

else:
    model: DiffusionUnetLowdimPolicy_flow_euler
    ema_model: DiffusionUnetLowdimPolicy_flow_euler = None


model = hydra.utils.instantiate(cfg.policy)

if cfg.training.use_ema:
    ema_model = copy.deepcopy(model)

# configure training state
optimizer = hydra.utils.instantiate(
    cfg.optimizer, params=model.parameters())

global_step = 0
epoch = 0
#%%
cfg.task.abs_action
#%%
    # def run(self):
cfg = copy.deepcopy(cfg)

# resume training
# if cfg.training.resume:
#     lastest_ckpt_path = get_checkpoint_path()
#     if lastest_ckpt_path.is_file():
#         print(f"Resuming from checkpoint {lastest_ckpt_path}")
#         load_checkpoint(path=lastest_ckpt_path)
#         # random from 0 to 20 sec sleep below
#         print('loading success')
#         time.sleep(random.randint(0,20))

# configure dataset
dataset: BaseLowdimDataset
dataset = hydra.utils.instantiate(cfg.task.dataset)
assert isinstance(dataset, BaseLowdimDataset)
train_dataloader = DataLoader(dataset, **cfg.dataloader)
normalizer = dataset.get_normalizer()

# configure validation dataset
val_dataset = dataset.get_validation_dataset()
val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

model.set_normalizer(normalizer)
if cfg.training.use_ema:
    ema_model.set_normalizer(normalizer)

# configure lr scheduler
lr_scheduler = get_scheduler(
    cfg.training.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps,
    num_training_steps=(
        len(train_dataloader) * cfg.training.num_epochs) \
            // cfg.training.gradient_accumulate_every,
    # pytorch assumes stepping LRScheduler every epoch
    # however huggingface diffusers steps it every batch
    last_epoch=global_step-1
)

# configure ema
ema: EMAModel = None
if cfg.training.use_ema:
    ema = hydra.utils.instantiate(
        cfg.ema,
        model=ema_model)
print("output_dir: ",output_dir)
time.sleep(3)
# configure env runner
print("cfg.task.env_runner.test_start_seed: ",cfg.task.env_runner.test_start_seed)
cfg.task.env_runner.test_start_seed=cfg.training.seed
print("cfg.task.env_runner.test_start_seed: ",cfg.task.env_runner.test_start_seed)
env_runner: BaseLowdimRunner
from diffusion_policy.env_runner.robomimic_lowdim_runner_classifier import RobomimicLowdimRunner
#%%
cfg.task.env_runner._target_='diffusion_policy.env_runner.robomimic_lowdim_runner_classifier.RobomimicLowdimRunner'

# cfg.task.env_runner.n_envs=1
#%%


#%%
env_runner = hydra.utils.instantiate(
    cfg.task.env_runner,
    # RobomimicLowdimRunner,
    output_dir=output_dir)
assert isinstance(env_runner, BaseLowdimRunner)
#%%


#%%
# configure logging
print("cfg.task: ",cfg.task)
print("cfg.task.name: ",cfg.task.name)
print("cfg.logging.project: ",cfg.logging.project)
print("cfg.logging.name: ",cfg.logging.name)

print("cfg.logging.name: ",cfg.logging.name)
# input('next?')

#%%
# checkpoint_dir=os.path.join('/data/ryan/code/robo/diffusion_policy',checkpoint_dir)
checkpoint_dir
#%%


import joblib
label_path='/data/ryan/code/robo/diffusion_policy/data/flow_label/square_xg_93_c1_learning_rate0.8_max_depth9_n_estimators100_1.pkl'
label_classifier = joblib.load(label_path)

#%%
os.getcwd()
#%%
load_dict={}
if not cfg.training.training:
    data2 = os.listdir(checkpoint_dir)
    data2=sorted(data2)
    print("checkpoint_dir: ",checkpoint_dir)
    print("checkpoint_dir os listdir: ",data2)
    for item in data2:
        fullpath=os.path.join(checkpoint_dir,item)
        if os.path.isfile(fullpath):
            if '.ckpt' in item and '=' in item:
                # data/output2/a15_6_unet_linear_abs_DDPM/20240923_084752_lift_lowdim_abs_unet_DDPM_W100_TS2/checkpoints/epoch=0000-test_mean_score=0.020.ckpt
                
                epoch_num=int(item.split('=')[1].split('-')[0])
                load_dict[epoch_num]=fullpath
                
# #%%
# cfg.multi_run={'run_dir': 'data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}', 'wandb_name_base': '${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'}
# ----> 2 cfg.multi_run.run_dir
# UnsupportedInterpolationType: Unsupported interpolation type now
#     full_key: multi_run.run_dir
#     object_type=dict
#%%
#%%
# cfg.multi_run['wandb_name_base']
#%%
cfg.multi_run
#%%
cfg.logging.project
#%%
cfg.logging.name
#%%
from wandb import Settings

# wandb_run = wandb.init(
    # project=cfg.logging.project,       # Your W&B project
    # name=cfg.logging.name,             # A run name or leave blank for auto-generated
    # config={                        # Any hyperparameters or settings you want to track
    #     "learning_rate": 1e-4,
    #     "batch_size": 64
    # },
    # dir=str(output_dir),
    # settings=Settings(jupyter=False),
    # mode="offline",  # no online sync
    # notebook='disabled'


    
    # ) # Optional: directory for wandb logs



# OmegaConf.to_container(cfg, resolve=True)
#%%
print('hi')
# wandb_run = wandb.init(
#     dir=str(output_dir),
#     config=OmegaConf.to_container(cfg, resolve=True),
#     **cfg.logging
# )
#%%
# wandb.init()
#%%
# wandb.config.update(
#     {
#         "output_dir": output_dir,
#     }
# )

# configure checkpoint
topk_manager = TopKCheckpointManager(
    save_dir=os.path.join(output_dir, 'checkpoints'),
    **cfg.checkpoint.topk
)
#%%
# device transfer
# device = torch.device(cfg.training.device)
device=find_empty_gpu_v2()
# device=torch.device(2)
# cfg.policy.device
#%%
# device=cfg.policy.device
model.to(device)
if ema_model is not None:
    ema_model.to(device)
optimizer_to(optimizer, device)

# save batch for sampling
train_sampling_batch = None

if cfg.training.debug:
    cfg.training.num_epochs = 2
    cfg.training.max_train_steps = 3
    cfg.training.max_val_steps = 3
    cfg.training.rollout_every = 1
    cfg.training.checkpoint_every = 1
    cfg.training.val_every = 1
    cfg.training.sample_every = 1
mean_score_test=0
mean_score_train=0
average_test_success=Average_past(10)
average_train_success=Average_past(10)
# training loop
print("device: ",device)
log_path = os.path.join(output_dir, 'logs.json.txt')
with JsonLogger(log_path) as json_logger:
    for local_epoch_idx in range(cfg.training.num_epochs):
        break

local_epoch_idx=4950

step_log = dict()
# ========= train for this epoch ==========
train_losses = list()
start1=time.time()
#%%

#%%
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
TASK=cfg.task.name.split('_')[0]
#%%
if TASK in ['tool_hang', 'can', 'lift', 'square']:
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
elif TASK in ['transport']:
    obs_keys = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos']

# robomimic_env = RobomimicLowdimWrapper(
#                         env=env_runner,
#                         obs_keys=obs_keys,
#                         init_state=None,
#                         render_hw=(256,256),
#                         render_camera_name='agentview'
#                         )

# env_fn
# env = AsyncVectorEnv(env_runner.env_fns)
#%%

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env
#%%

# if local_epoch_idx % cfg.training.checkpoint_every==0 and local_epoch_idx>=cfg.training.infer_from:
#     try:
        
#     except Exception as e:
#         print('success done- ',e)
#         continue
# else:
#     wandb_run.log(step_log, step=local_epoch_idx)
#     continue
import dill

ckpt_path=load_dict[local_epoch_idx]
model_name=ckpt_path.split('/')[-1]
print(f"Resuming from checkpoint {ckpt_path}")
print("device: ",device)
#%%
ckpt_path
#%%
# states=torch.load(ckpt_path)
# states
#%%

# path = pathlib.Path(ckpt_path)
# payload = torch.load(path.open('rb'), pickle_module=dill)
# # self.load_payload(payload, 
# #     exclude_keys=exclude_keys, 
# #     include_keys=include_keys)
# exclude_keys = tuple()
# include_keys = payload['pickles'].keys()
# include_keys

#%%
# for key, value in payload['state_dicts'].items():
#     print(key)
# #%%
# # print("payload.keys(): ",payload.keys())
# for key, value in payload['state_dicts'].items():
#     # print("key: ",key)
#     # print("value: ",value)
#     if key not in exclude_keys:
#         self.__dict__[key].load_state_dict(value)
# for key in include_keys:
#     if key in payload['pickles']:
#         self.__dict__[key] = dill.loads(payload['pickles'][key])

# return payload
# states = torch.load(ckpt_path, pickle_module=dill)
states = torch.load(ckpt_path)
states
#%%
# my_ema_model = DiffusionUnetLowdimPolicy_flow_euler(**some_kwargs)
#%%
# states = torch.load(ckpt_path, pickle_module=dill, map_location='cpu')
states = torch.load(ckpt_path, map_location='cpu')
states
#%%
# Extract the ema_model part
ema_sd = states['state_dicts']['ema_model']

# Now load it into your newly created model
ema_model.load_state_dict(ema_sd)
#%%
ema_model.eval()  # optional
#%%
model_sd = states['state_dicts']['model']

model.load_state_dict(model_sd)

# RuntimeError: Error(s) in loading state_dict for DiffusionUnetLowdimPolicy_flow_euler:
# 	Missing key(s) in state_dict: "_dummy_variable", "model.s_embeder.1.weight", "model.s_embeder.1.bias", "model.s_embeder.3.weight", "model.s_embeder.3.bias", "model.mid_modules.0.blocks.0.block.0.weight", "model.mid_modules.0.blocks.0.block.0.bias", "model.mid_modules.0.blocks.0.block.1.weight", "model.mid_modules.0.blocks.0.block.1.bias", "model.mid_modules.0.blocks.1.block.0.weight", "model.mid_modules.0.blocks.1.block.0.bias", "model.mid_modules.0.blocks.1.block.1.weight", "model.mid_modules.0.blocks.1.block.1.bias", "model.mid_modules.0.cond_encoder.1.weight", "model.mid_modules.0.cond_encoder.1.bias", "model.mid_modules.1.blocks.0.block.0.weight", "model.mid_modules.1.blocks.0.block.0.bias", "model.mid_modules.1.blocks.0.block.1.weight", "model.mid_modules.1.blocks.0.block.1.bias", "model.mid_modules.1.blocks.1.block.0.weight", "model.mid_modules.1.blocks.1.block.0.bias", "model.mid_modules.1.blocks.1.block.1.weight", "model.mid_modules.1.blocks.1.block.1.bias", "model.mid_modules.1.cond_encoder.1.weight", "model.mid_modules.1.cond_encoder.1.bias", "model.diffusion_step_encoder.1.weight", "model.diffusion_step_encoder.1.bias", "model.diffusion_step_encoder.3.weight", "model.diffusion_step_encoder.3.bias", "model.up_modules.0.0.blocks.0.block.0.weight", "model.up_modules.0.0.blocks.0.block.0.bias", "model.up_modules.0.0.blocks.0.block.1.weight", "model.up_modules.0.0.blocks.0.block.1.bias", "model.up_modules.0.0.blocks.1.block.0.weight", "model.up_modules.0.0.blocks.1.block.0.bias", "model.up_modules.0.0.blocks.1.block.1.weight", "model.up_modules.0.0.blocks.1.block.1.bias", "model.up_modules.0.0.cond_encoder.1.weight", "model.up_modules.0.0.cond_encoder.1.bias", "model.up_modules.0.0.residual_conv.weight", "model.up_modules.0.0.residual_conv.bias", "model.up_modules.0.1.blocks.0.block.0.weight", "model.up_modules.0.1.blocks.0.block.0.bias", "model.up_modules.0.1.blocks.0.block.1.weight", "model.up_modules.0.1.blocks.0.block.1.bias", "model.up_modules.0.1.blocks.1.block.0.weight", "model.up_modules.0.1.blocks.1.block.0.bias", "model.up_modules.0.1.blocks.1.block.1.weight", "model.up_modules.0.1.blocks.1.block.1.bias", "model.up_modules.0.1.cond_encoder.1.weight", "model.up_modules.0.1.cond_encoder.1.bias", "model.up_modules.0.2.conv.weight", "model.up_modules.0.2.conv.bias", "model.up_modules.1.0.blocks.0.block.0.weight", "model.up_modules.1.0.blocks.0.block.0.bias", "model.up_modules.1.0.blocks.0.block.1.weight", "model.up_modules.1.0.blocks.0.block.1.bias", "model.up_modules.1.0.blocks.1.block.0.weight", "model.up_modules.1.0.blocks.1.block.0.bias", "model.up_modules.1.0.blocks.1.block.1.weight", "model.up_modules.1.0.blocks.1.block.1.bias", "model.up_modules.1.0.cond_encoder.1.weight", "model.up_modules.1.0.cond_encoder.1.bias", "model.up_modules.1.0.residual_conv.weight", "model.up_modules.1.0.residual_conv.bias", "model.up_modules.1.1.blocks.0.block.0.weight", "model.up_modules.1.1.blocks.0.block.0.bias", "model.up_modules.1.1.blocks.0.block.1.weight", "model.up_modules.1.1.blocks.0.block.1.bias", "model.up_modules.1.1.blocks.1.block.0.weight", "model.up_modules.1.1.blocks.1.block.0.bias", "model.up_modules.1.1.blocks.1.block.1.weight", "model.up_modules.1.1.blocks.1.block.1.bias", "model.up_modules.1.1.cond_encoder.1.weight", "model.up_modules.1.1.cond_encoder.1.bias", "model.up_modules.1.2.conv.weight", "model.up_modules.1.2.conv.bias", "model.down_modules.0.0.blocks.0.block.0.weight", "model.down_modules.0.0.blocks.0.block.0.bias", "model.down_modules.0.0.blocks.0.block.1.weight", "model.down_modules.0.0.blocks.0.block.1.bias", "model.down_modules.0.0.blocks.1.block.0.weight", "model.down_modules.0.0.blocks.1.block.0.bias", "model.down_modules.0.0.blocks.1.block.1.weight", "model.down_modules.0.0.blocks.1.block.1.bias", "model.down_modules.0.0.cond_encoder.1.weight", "model.down_modules.0.0.cond_encoder.1.bias", "model.down_modules.0.0.residual_conv.weight", "model.down_modules.0.0.residual_conv.bias", "model.down_modules.0.1.blocks.0.block.0.weight", "model.down_modules.0.1.blocks.0.block.0.bias", "model.down_modules.0.1.blocks.0.block.1.weight", "model.down_modules.0.1.blocks.0.block.1.bias", "model.down_modules.0.1.blocks.1.block.0.weight", "model.down_modules.0.1.blocks.1.block.0.bias", "model.down_modules.0.1.blocks.1.block.1.weight", "model.down_modules.0.1.blocks.1.block.1.bias", "model.down_modules.0.1.cond_encoder.1.weight", "model.down_modules.0.1.cond_encoder.1.bias", "model.down_modules.0.2.conv.weight", "model.down_modules.0.2.conv.bias", "model.down_modules.1.0.blocks.0.block.0.weight", "model.down_modules.1.0.blocks.0.block.0.bias", "model.down_modules.1.0.blocks.0.block.1.weight", "model.down_modules.1.0.blocks.0.block.1.bias", "model.down_modules.1.0.blocks.1.block.0.weight", "model.down_modules.1.0.blocks.1.block.0.bias", "model.down_modules.1.0.blocks.1.block.1.weight", "model.down_modules.1.0.blocks.1.block.1.bias", "model.down_modules.1.0.cond_encoder.1.weight", "model.down_modules.1.0.cond_encoder.1.bias", "model.down_modules.1.0.residual_conv.weight", "model.down_modules.1.0.residual_conv.bias", "model.down_modules.1.1.blocks.0.block.0.weight", "model.down_modules.1.1.blocks.0.block.0.bias", "model.down_modules.1.1.blocks.0.block.1.weight", "model.down_modules.1.1.blocks.0.block.1.bias", "model.down_modules.1.1.blocks.1.block.0.weight", "model.down_modules.1.1.blocks.1.block.0.bias", "model.down_modules.1.1.blocks.1.block.1.weight", "model.down_modules.1.1.blocks.1.block.1.bias", "model.down_modules.1.1.cond_encoder.1.weight", "model.down_modules.1.1.cond_encoder.1.bias", "model.down_modules.1.2.conv.weight", "model.down_modules.1.2.conv.bias", "model.down_modules.2.0.blocks.0.block.0.weight", "model.down_modules.2.0.blocks.0.block.0.bias", "model.down_modules.2.0.blocks.0.block.1.weight", "model.down_modules.2.0.blocks.0.block.1.bias", "model.down_modules.2.0.blocks.1.block.0.weight", "model.down_modules.2.0.blocks.1.block.0.bias", "model.down_modules.2.0.blocks.1.block.1.weight", "model.down_modules.2.0.blocks.1.block.1.bias", "model.down_modules.2.0.cond_encoder.1.weight", "model.down_modules.2.0.cond_encoder.1.bias", "model.down_modules.2.0.residual_conv.weight", "model.down_modules.2.0.residual_conv.bias", "model.down_modules.2.1.blocks.0.block.0.weight", "model.down_modules.2.1.blocks.0.block.0.bias", "model.down_modules.2.1.blocks.0.block.1.weight", "model.down_modules.2.1.blocks.0.block.1.bias", "model.down_modules.2.1.blocks.1.block.0.weight", "model.down_modules.2.1.blocks.1.block.0.bias", "model.down_modules.2.1.blocks.1.block.1.weight", "model.down_modules.2.1.blocks.1.block.1.bias", "model.down_modules.2.1.cond_encoder.1.weight", "model.down_modules.2.1.cond_encoder.1.bias", "model.final_conv.0.block.0.weight", "model.final_conv.0.block.0.bias", "model.final_conv.0.block.1.weight", "model.final_conv.0.block.1.bias", "model.final_conv.1.weight", "model.final_conv.1.bias", "mask_generator._dummy_variable". 
# 
# 
# 	Unexpected key(s) in state_dict: "cfg", "state_dicts", "pickles". 
#%%
states['state_dicts']['ema_model'].keys()
#%%
ema_model.state_dict().keys()
#%%
# load_checkpoint(path=ckpt_path)
path=ckpt_path
path = pathlib.Path(path)
payload = torch.load(path.open('rb'), pickle_module=dill)
# load_payload(payload, 
#     exclude_keys=exclude_keys, 
#     include_keys=include_keys)
# #%%
# # if exclude_keys is None:
exclude_keys = tuple()
# # if include_keys is None:
include_keys = payload['pickles'].keys()
print("include_keys: ",include_keys)
print("payload.keys(): ",payload.keys())
for key, value in payload['state_dicts'].items():
    print("key: ",key)
#     # print("value: ",value)
#     if key not in exclude_keys:
#         self.__dict__[key].load_state_dict(value, **kwargs)
# for key in include_keys:
#     if key in payload['pickles']:
#         self.__dict__[key] = dill.loads(payload['pickles'][key])

# return payload
'''
include_keys:  dict_keys(['_output_dir', 'global_step', 'epoch'])
payload.keys():  dict_keys(['cfg', 'state_dicts', 'pickles'])
key:  ema_model
key:  model
key:  optimizer
'''
#%%
device
#%%
# ========= eval for this epoch ==========
model=model.to(device)
ema_model=ema_model.to(device)
policy = model
if cfg.training.use_ema:
    policy = ema_model

if ema_model is not None:
    ema_model.to(device)

# policy = model


policy.eval()
policy=policy.to(device)
#%%
policy.device
#%%



device = policy.device
dtype = policy.dtype
env = env_runner.env

# plan for rollout
n_envs = len(env_runner.env_fns)
n_inits = len(env_runner.env_init_fn_dills)
n_chunks = math.ceil(n_inits / n_envs)

# allocate data
all_video_paths = [None] * n_inits
all_rewards = [None] * n_inits
#%%
n_chunks
#%%
for chunk_idx in range(n_chunks):
    break
start = chunk_idx * n_envs
end = min(n_inits, start + n_envs)
this_global_slice = slice(start, end)
this_n_active_envs = end - start
this_local_slice = slice(0,this_n_active_envs)

this_init_fns = env_runner.env_init_fn_dills[this_global_slice]
n_diff = n_envs - len(this_init_fns)
if n_diff > 0:
    this_init_fns.extend([env_runner.env_init_fn_dills[0]]*n_diff)
assert len(this_init_fns) == n_envs

# init envs
env.call_each('run_dill_function', 
    args_list=[(x,) for x in this_init_fns])

# start rollout
obs = env.reset()
past_action = None
# policy.reset()

env_name = env_runner.env_meta['env_name']
pbar = tqdm.tqdm(total=env_runner.max_steps, desc=f"rollout robomimic lowdim runner {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
    leave=False, mininterval=env_runner.tqdm_interval_sec)

done = False
# %%
while not done:
# for _ in range(20):
    # create obs dict
    np_obs_dict = {
        # handle n_latency_steps by discarding the last n_latency_steps
        'obs': obs[:,:env_runner.n_obs_steps].astype(np.float32)
    }
    if env_runner.past_action and (past_action is not None):
        # TODO: not tested
        np_obs_dict['past_action'] = past_action[
            :,-(env_runner.n_obs_steps-1):].astype(np.float32)

    # device transfer
    obs_dict = dict_apply(np_obs_dict, 
        lambda x: torch.from_numpy(x).to(
            device=device))





    X_test_np=obs[:,1,:]


    # X_test_np  = obs

    # 5) (Optional) Verify loaded model predictions
    # label_hat = label_classifier.predict(X_test_np)
    # label_hat=int(label_hat)
    policy.TS=10

    # if label_hat==1 or label_hat==0:
    #     policy.TS=10  
    #     print(label_hat,"TS=10")  
    # else:
    #     policy.TS=100
    #     print(label_hat,"TS=100")


    # run policy
    with torch.no_grad():
        action_dict = policy.predict_action(obs_dict)

    # device_transfer
    np_action_dict = dict_apply(action_dict,
        lambda x: x.detach().to('cpu').numpy())

    # handle latency_steps, we discard the first n_latency_steps actions
    # to simulate latency
    action = np_action_dict['action'][:,env_runner.n_latency_steps:]
    if not np.all(np.isfinite(action)):
        print(action)
        raise RuntimeError("Nan or Inf action")

    # step env
    env_action = action
    if env_runner.abs_action:
        env_action = env_runner.undo_transform_action(action)

    obs, reward, done, info = env.step(env_action)
    done = np.all(done)
    past_action = action

    # update pbar
    pbar.update(action.shape[1])
    if done:
        print('done!!!')
pbar.close()

# collect data for this round
all_video_paths[this_global_slice] = env.render()[this_local_slice]
all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

#%%
#%%
import collections

# logimport collections

max_rewards = collections.defaultdict(list)
log_data = dict()
# results reported in the paper are generated using the commented out line below
# which will only report and average metrics from first n_envs initial condition and seeds
# fortunately this won't invalidate our conclusion since
# 1. This bug only affects the variance of metrics, not their mean
# 2. All baseline methods are evaluated using the same code
# to completely reproduce reported numbers, uncomment this line:
# for i in range(len(self.env_fns)):
# and comment out this line
# for i in range(n_inits):
#     seed = env_runner.env_seeds[i]
#     prefix = env_runner.env_prefixs[i]
#     max_reward = np.max(all_rewards[i])
#     max_rewards[prefix].append(max_reward)
#     log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

#     # visualize sim
#     video_path = all_video_paths[i]
#     if video_path is not None:
#         sim_video = wandb.Video(video_path)
#         log_data[prefix+f'sim_video_{seed}'] = sim_video

# # log aggregate metrics
# for prefix, value in max_rewards.items():
#     name = prefix+'mean_score'
#     value = np.mean(value)
#     log_data[name] = value
# #%%
# all_rewards
# #%%

# log_data


# save_video(log_data,local_epoch_idx,output_dir_media)

#%%
# env_meta=env_runner.env_meta
# robomimic_env = create_env(
#                     env_meta=env_meta, 
#                     obs_keys=obs_keys
#                 )
# img = robomimic_env.render(mode='rgb_array',height=256,width=256)
# img

#img.shape =(256, 256, 3)
# how to plot
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
# copy_env=copy.deepcopy(env)
#%%
frame = env.render(mode='rgb_array')#%%
frame
# ('data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250126_095406__TS2_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS2/media/d8hjgpnv.mp4',)
#%%
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# clear_output(wait=True)       # Clears old plot in Jupyter
# plt.imshow(frame)
# plt.title(f"Step {0} - reward={reward}")
# plt.axis('off')
# plt.show()

#%%
frame
#%%
# pbar.close()

# # collect data for this round
# all_video_paths[this_global_slice] = env.render()[this_local_slice]
# all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]


'''
data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250126_111657__TS2_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS2/media/w4zvnoxb.mp4
'''

#%%
# run rollout
if (epoch % cfg.training.rollout_every) == 0:
    TS=10
    classifier_bool=False
    runner_log = env_runner.run(policy,label_classifier,classifier_bool,TS)
    # log all
    # step_log.update(runner_log)
    save_video(runner_log,local_epoch_idx,output_dir_media)
    
    mean_score_test = runner_log['test/mean_score']
    mean_score_train = runner_log['train/mean_score']
    mean_score_test=round(mean_score_test,2)
    mean_score_train=round(mean_score_train,2)
    # i want to save all the values in runner_log into step_log. how?
    # step_log.update(runner_log)
    # print("mean_score_train: ",mean_score_train)
    # print("mean_score_test: ",mean_score_test)
    step_log['test_mean_score'] = mean_score_test
    step_log['train_mean_score'] = mean_score_train
    average_test_success.update(mean_score_test)
    average_train_success.update(mean_score_train)

#%%
len(env_runner.env_fns)
#%%
obs=env_runner.env.reset()
#%%
obs.shape
#%%
# policy.reset()
#%%
np_obs_dict = {
        # handle n_latency_steps by discarding the last n_latency_steps
        'obs': obs[:,:env_runner.n_obs_steps].astype(np.float32)
    }
np_obs_dict
obs_dict = dict_apply(np_obs_dict, 
    lambda x: torch.from_numpy(x).to(
        device=device))
obs_dict
#%%
# print("Loaded model fromname1)
# 
#%%

X_test_np=obs[:,1,:]


# X_test_np  = obs

# 5) (Optional) Verify loaded model predictions
# label_hat = label_classifier.predict(X_test_np)
# label_hat
#%%

#%%
# label_hat=int(label_hat)
policy.TS=10
# if label_hat==1:
#     policy.TS=10    
# else:
#     policy.TS=100
#%%
with torch.no_grad():
    action_dict = policy.predict_action(obs_dict)
action_dict
#%%
action_dict['action'].shape
#%%
action_dict['action_pred'].shape

#%%
runner_log['test/mean_score']
#%%

#%%
# run validation
if (epoch % cfg.training.val_every) == 0:
    with torch.no_grad():
        val_losses = list()
        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {epoch}", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss = model.compute_loss(batch)
                val_losses.append(loss)
                if (cfg.training.max_val_steps is not None) \
                    and batch_idx >= (cfg.training.max_val_steps-1):
                    break
        if len(val_losses) > 0:
            val_loss = torch.mean(torch.tensor(val_losses)).item()
            # log epoch average validation loss
            step_log['val_loss'] = val_loss

# run diffusion sampling on a training batch
if (epoch % cfg.training.sample_every) == 0:
    with torch.no_grad():
        # sample trajectory from training set, and evaluate difference
        # batch = train_sampling_batch
        obs_dict = {'obs': batch['obs']}
        gt_action = batch['action']
        
        result = policy.predict_action(obs_dict)
        if cfg.pred_action_steps_only:
            pred_action = result['action']
            start = cfg.n_obs_steps - 1
            end = start + cfg.n_action_steps
            gt_action = gt_action[:,start:end]
        else:
            pred_action = result['action_pred']
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        # log
        step_log['train_action_mse_error'] = mse.item()
        # release RAM
        del batch
        del obs_dict
        del gt_action
        del result
        del pred_action
        del mse

#%%
# end of epoch
# log of last step is combined with validation and rollout
step_log['test_mean_score_last10'] = average_test_success.value
step_log['train_mean_score_last10'] = average_train_success.value
wandb_run.log(step_log, step=local_epoch_idx)
json_logger.log(step_log)
global_step += 1
epoch += 1
end1=time.time()
time_took=end1-start1
time_took=round(time_took,2)
if local_epoch_idx%10==0:
    print('logging.name',cfg.logging.name,'scriptname:',scriptname,'logging.project:',cfg.logging.project)
    print('processes: ',print_process_counts())
    print('train_success:',mean_score_train,'test_success:',mean_score_test)
    print('avg10_train_success:',average_train_success.value,'avg10_test_success:',average_test_success.value)
    print("device: ",device)
    # print('to complete 1000 epochs:',int(time_took*1000/60/60),'hours')
    print(' ')
print(f"epoch {local_epoch_idx} - {int(time_took)} sec",'50 epoch for 5000 epochs: ',int(time_took*100/60/60),'hours', '50 epoch for 1000 epochs: ',int(time_took*20/60/60),'hours')
#%%
import pathlib#
import sys
import os
from omegaconf import OmegaConf

# Make sure your PYTHONPATH or sys.path includes the location
# of 'TrainDiffusionUnetLowdimWorkspace' if it's in another folder.
# from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

# def main():
# 1. Load an existing YAML config


# workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
# workspace.run()

# if __name__ == "__main__":
#     main()


# %%
'''
code i used to run:
v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M

path=data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/checkpoints/latest.ckpt



python diffusion_policy/run4/a2_unified_both_lowdim_abs.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=classifier_v2_infer_ode2_v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=0.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M \
    training.resume=False \
    +policy.path_type=GVP2 \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=. \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=10 \
    +policy.last_step=. \
    +policy.last_step_size=. \
    +policy.train_eps=0 \
    +policy.sample_eps=0 \
    +policy.prediction=velocity \
    +policy.Tsam=uniform \
    +policy.Ssam=uniform \
    +policy.Tdiscrete=Tdiscrete \
    +policy.model.t_minus_one=t_minus_one \
    training.seed=1000000 \
    training.num_epochs=5000 \
    +policy.beta_params=_ \
    +policy.beta_scaling=_ \
    +training.training=False \
    +training.infer_from=4900


'''
# %%

# %%
