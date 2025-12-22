
python diffusion_policy/run1/a21_1_2_linear_new_v3_without_eval.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=pusht_lowdim_abs \
    logging.name=pusht_lowdim_abs \
    logging.project=pusht_train_unet_velocity_linear_T10000_TVP_Teps0_v8 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=10000 \
    +policy.T=10000 \
    +policy.model.T=10000 \
    +policy.TS=2 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +policy.ODE_SDE=ODE \
    +policy.train_eps=0.0 \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.prediction=velocity \
    +policy.sample_eps=0.0 \
    training.resume=False \
    +training.save_from=0 \
    +policy.Tsam=vp_like
