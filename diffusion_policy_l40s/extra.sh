scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_2_VP_new_v3 /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_3_GVP_new_v3 /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_linear_new_v3 /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_2_3_VP_new_v3_without_eval_eps0001 /mnt/l1/code/robo/diffusion_policy_l40s/data/output2


scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a1_DDPM_robomimic_lowdim_abs_sample_only /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_2_linear_new_v3_without_eval /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_3_image_linear_new_v3_without_eval /mnt/l1/code/robo/diffusion_policy_l40s/data/output2

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_3_2_GVP_new_v3_without_eval /mnt/l1/code/robo/diffusion_policy_l40s/data/output2


scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_3_2_GVP_new_v3_without_eval /mnt/l1/code/robo/diffusion_policy_l40s/data/output2


scp -P 22 \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output4_train/can_lowdim.zip \
    /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train
    
scp -P 22 \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output4_train/lift_lowdim.zip \
    /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train
    
scp -P 22 \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output4_train/pusht_lowdim.zip \
    /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train


scp -P 22 \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output4_train/tool_hang_lowdim.zip \
    /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train


scp -P 22 \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output4_train/unified_both_lowdim_abs.zip \
    /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train


unzip pusht_lowdim.zip -d pusht_lowdim
unzip 



mv -r /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/tool_hang_lowdim \
 /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train

vs


scp -r /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/tool_hang_lowdim.zip \
 /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train

unzip tool_hang_lowdim.zip -d pusht_lowdim


copy and unzip everything as its name as folder for all below.

unified_both_lowdim_abs.zip
tool_hang_lowdim.zip
lift_lowdim.zip
can_lowdim.zip
pusht_lowdim.zip



# 1. Copy the zip files from source to destination
cd /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train

scp /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/unified_both_lowdim_abs.zip \
    /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train
unzip unified_both_lowdim_abs.zip -d unified_both_lowdim_abs

scp /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/tool_hang_lowdim.zip \
    /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train
unzip tool_hang_lowdim.zip         -d tool_hang_lowdim

scp /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/lift_lowdim.zip \
    /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train
unzip lift_lowdim.zip              -d lift_lowdim

scp /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/can_lowdim.zip \
    /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train
unzip can_lowdim.zip               -d can_lowdim

scp /mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train/pusht_lowdim.zip \
    /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train
unzip pusht_lowdim.zip             -d pusht_lowdim
# 2. Change directory to the destination folder
cd /mnt/A3/1_code/3_version/diffusion_policy_l40s/data/output4_train

# 3. Unzip each file into a directory with the same base name
