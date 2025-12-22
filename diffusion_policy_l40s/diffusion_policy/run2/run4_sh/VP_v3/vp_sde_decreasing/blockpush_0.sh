python diffusion_policy/run2/a1_4_1_VP_sample_only_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_lowdim_abs_VP_SDE_Euler_decreasing_v5 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241013_061613_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=decreasing \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100

