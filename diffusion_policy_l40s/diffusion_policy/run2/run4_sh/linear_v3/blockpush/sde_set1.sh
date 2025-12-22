
python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=increasing-decreasing \
    +policy.TS=100


python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=linear \
    +policy.TS=100


python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=SBDM \
    +policy.TS=100

python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=constant \
    +policy.TS=100



python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=increasing-decreasing \
    +policy.TS=50


python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=linear \
    +policy.TS=50


python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=SBDM \
    +policy.TS=50

python diffusion_policy/run2/a1_2_3_linear_sample_new_v3_from_3500.py \
    --config-path=../config \
    --config-name=train_diffusion_unet_lowdim_workspace.yaml \
    task=blockpush_lowdim_abs  \
    logging.name=blockpush_lowdim_abs \
    logging.project=unet_blockpush_linear_sde_set1 \
    +training.architecture=unet \
    dataloader.num_workers=0 \
    +policy.DDPM=Euler \
    +policy.WSCALE=1000 \
    +policy.T=100 \
    +training.save_50=True \
    policy._target_=diffusion_policy.policy.diffusion_unet_lowdim_policy_flow_euler.DiffusionUnetLowdimPolicy_flow_euler \
    policy.model._target_=diffusion_policy.model.diffusion.conditional_unet1d_flow.ConditionalUnet1D_Flow\
    +policy.path_type=Linear \
    +training.save_path=data/output2/a21_1_linear_new_v3/20240930_173106_blockpush_lowdim_seed_abs_unet_euler_W1000_TS2/checkpoints \
    +policy.ODE_SDE=SDE \
    +policy.T_CHANGED=True \
    +policy.model.T_CHANGED=True \
    +policy.model.T=100 \
    +policy.diffusion_form=constant \
    +policy.TS=50
