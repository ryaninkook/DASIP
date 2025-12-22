

python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=1




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=2



python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=4




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=10


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=50




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=sigma \
    +policy.TS=100






python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=1




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=2



python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=4




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=10


python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=50




python diffusion_policy/run2/a1_2_2_linear_sample_new_v3.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=square_lowdim_abs  \
    logging.name=square_lowdim_abs \
    logging.project=unet_square_linear_new_v3_v2 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_100415_square_lowdim_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=decreasing \
    +policy.TS=100


