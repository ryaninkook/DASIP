

python diffusion_policy/run1/a21_1_3_linear_new_v3_both.py \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=transport_lowdim_abs  \
    logging.name=transport_lowdim_abs \
    logging.project=transport_infer_unet_VP_velocity_lowdim_abs_combined_SDE_Seps4_Teps3_v8 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +training.save_50=True \
    +training.save_from=3499.0 \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow \
    +training.save_path=data/output2/a21_2_2_VP_new_v3_without_eval/20241108_121457_transport_lowdim_abs_unet_Euler_VP_velocity_Teps0.001_Seps0.0_TS2/checkpoints \
    training.resume=False \
    +policy.path_type=VP \
    +policy.ODE_SDE=SDE \
    +policy.DDPM=Euler \
    +policy.diffusion_form=linear \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +policy.model.T=100 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.TS=100 \
    +policy.last_step=Euler \
    +policy.last_step_size=0.01 \
    +policy.train_eps=0.001 \
    +policy.prediction=velocity \
    +policy.sample_eps=0.0001 \
    +training.training=False \
    +training.threshold_epoch=3500