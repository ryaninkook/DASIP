#%%
import os
import shutil
import zipfile
import subprocess

def print_disk_usage():
    # This will print the disk usage using `df -h` command
    subprocess.run(["df", "-h"], check=True)

def create_zip(directory_path, archive_name=None):
    """
    Creates a zip archive of the directory specified by directory_path.
    archive_name (without .zip) can be specified; 
    if not provided, it defaults to the directory name.
    """
    directory_path = os.path.abspath(directory_path)
    if archive_name is None:
        archive_name = os.path.basename(directory_path)

    # The resulting zip file will be placed in the parent directory of directory_path
    parent_dir = os.path.dirname(directory_path)
    zip_path = os.path.join(parent_dir, archive_name)  # without extension
    shutil.make_archive(zip_path, 'zip', directory_path)
    return zip_path + '.zip'

def verify_zip(zip_path):
    """
    Verifies that the given file is a valid zip archive.
    """
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Attempt to list contents to ensure zip integrity
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False

def remove_directory(directory_path):
    """
    Removes the given directory recursively.
    """
    shutil.rmtree(directory_path)

# def main():
    # Directories to archive and remove
    # Replace this with the actual directories you want to process
# directories = [
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a1_2_2_linear_sample_new_v3",
#     # "/path/to/your/large/folder2",
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a1_2_2_linear_sample_new_v3_only_last",
#     '/data/ryan/code/robo/diffusion_policy/data/output2/a1_2_3_linear_sample_new_v3_from_3500',
#     '/data/ryan/code/robo/diffusion_policy/data/output2/a1_2_linear_robomimic_lowdim_abs_sample_only',
#     # Add as many as you need
#     # /data/ryan/code/robo/diffusion_policy/data/output2/a1_3_2_GVP_sample_only
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a1_3_2_GVP_sample_only",
#     # /data/ryan/code/robo/diffusion_policy/data/output2/a1_DDPM_robomimic_lowdim_abs_sample_only
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a1_DDPM_robomimic_lowdim_abs_sample_only",
#     #/data/ryan/code/robo/diffusion_policy/data/output2/a15_3_unet_linear_interpol_mimic_abs
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a15_3_unet_linear_interpol_mimic_abs",
#     # /data/ryan/code/robo/diffusion_policy/data/output2/a21_1_linear_new_v3
#     "/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_linear_new_v3",
# ]
directories = [
    # data/output2/a1_3_3_GVP_sample_only_from_3500
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a1_3_3_GVP_sample_only_from_3500',
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a1_3_GVP_robomimic_lowdim_abs_sample_only',
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a1_4_1_VP_sample_only_from_3500',
    # /data/ryan/code/robo/diffusion_policy/data/output2/a15_4_2_unet_VP_mimic_abs_new_v3
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a15_4_2_unet_VP_mimic_abs_new_v3',
    #/data/ryan/code/robo/diffusion_policy/data/output2/a15_4_unet_VP_mimic_abs
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a15_4_unet_VP_mimic_abs',
    #/data/ryan/code/robo/diffusion_policy/data/output2/a15_5_unet_GVP_mimic_abs
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a15_5_unet_GVP_mimic_abs',
    #//data/ryan/code/robo/diffusion_policy/data/ryan/code/robo/diffusion_policy//data/ryan/code/robo/diffusion_policy/data/output2/a15_6_2_train_unet_DDPM_without_eval
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a15_6_2_train_unet_DDPM_without_eval',
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_2_linear_new_v3_without_eval',
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a21_1_3_image_linear_new_v3_without_eval',
    # '/data/ryan/code/robo/diffusion_policy/data/output2/a21_2_2_VP_new_v3_without_eval',
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/blockpush_lowdim_seed_abs',
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/lift_lowdim',
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/can_lowdim'
    # '/data/ryan/code/robo/diffusion_policy/data/outputs'
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/pusht_lowdim',
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/tool_hang_lowdim',
    # '/data/ryan/code/robo/diffusion_policy/data/output4_train/unified_both_lowdim_abs'
    '/data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim'
]
print("Disk usage before:")
print_disk_usage()


for dir_path in directories:
    if not os.path.isdir(dir_path):
        print(f"Skipping {dir_path}, not a directory.")
        continue

    print(f"Zipping directory: {dir_path}")
    zip_file = create_zip(dir_path)
    print(f"Created zip file: {zip_file}")

    print(f"Verifying zip file: {zip_file}")
    if verify_zip(zip_file):
        print(f"Zip verification successful. Removing original directory: {dir_path}")
        remove_directory(dir_path)
        print("Directory removed.")
    else:
        print(f"Verification failed for {zip_file}. Not removing {dir_path}.")

print("Disk usage after:")
print_disk_usage()

# if __name__ == "__main__":
#     main()
#%%
# %%
'''

scp -r ryan@192.168.2.105:/data/ryan/code/robo/diffusion_policy/data/output2/a21_2_2_VP_new_v3_without_eval.zip   /mnt/H1/code/robo/diffusion_policy_l40s/data/output2

scp -r /data/ryan/code/robo/diffusion_policy/data/output2/a21_2_2_VP_new_v3_without_eval.zip \
      ryan@<LINUX3_IP>:/mnt/H1/code/robo/diffusion_policy_l40s/data/output2


scp -P 22 \
    /data/ryan/code/robo/diffusion_policy/data/output4_obs.zip \
    linux3@207.38.141.4:/mnt/H1/code/robo/diffusion_policy_l40s/data/output2


scp -P 5555 \
    /data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim.zip \
    linux1@207.38.141.4:/mnt/l1/code/robo/diffusion_policy_l40s/data/output4_train
    
    
    '''

# import subprocess
# for dir_path in directories:

#     # Define the local file to send, the port, and the destination
#     # zip_file='a21_2_2_VP_new_v3_without_eval.zip'
#     # '''            /data/ryan/code/robo/diffusion_policy/data/output4_train/blockpush_lowdim_seed_abs'''
#     # local_file = f"/data/ryan/code/robo/diffusion_policy/data/output2/{zip_file}"
#     local_file=f"{dir_path}.zip"
#     remote_user = "linux3"
#     remote_host = "207.38.141.4"
#     remote_path = "/mnt/H1/code/robo/diffusion_policy_l40s/data/output2"
#     port = "22"

#     # Build the SCP command
#     cmd = [
#         "scp",
#         "-P", port,  # -P is uppercase for scp
#         local_file,
#         f"{remote_user}@{remote_host}:{remote_path}"
#     ]

#     # Run the command
#     subprocess.run(cmd)
#     print('success v2 - ',dir_path)
# %%
