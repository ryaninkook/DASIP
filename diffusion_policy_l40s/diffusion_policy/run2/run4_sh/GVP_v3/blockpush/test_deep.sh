

python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=ODE \
    +policy.DDPM=euler \
    +policy.diffusion_form=none \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100


python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100


python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.last_step=Mean \
    +policy.last_step_size=0.01 \
    +policy.TS=100


python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.last_step=Euler \
    +policy.last_step_size=0.01 \
    +policy.TS=100


python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.last_step=Euler \
    +policy.last_step_size=0.1 \
    +policy.TS=100


python diffusion_policy/run2/a1_3_3_GVP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=v6_GVP_blockpush_ode_sde_study \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_3_GVP_new_v3/20241001_070116_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=GVP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.last_step=Mean \
    +policy.last_step_size=0.1 \
    +policy.TS=100
