



python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=1



python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=2



python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=3




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=4


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=5




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=6


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=7




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=8

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=9

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=tool_hang_lowdim_abs  \
    logging.name=tool_hang_lowdim_abs \
    logging.project=unet_tool_hang_linear_new_v3_ode123_seed200000 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100407_tool_hang_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=ODE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    task.env_runner.test_start_seed=200000 \
    +policy.TS=10

