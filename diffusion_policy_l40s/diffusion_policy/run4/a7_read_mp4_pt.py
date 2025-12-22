#%%

video_path='/data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250129_105600__TS10_[10, 10, 10, 10, 10, 10, 10]_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS10/media/3500_d6bnkcra_1.0.mp4'
video_path='/data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250129_122044__TS10_[10, 10, 10, 10, 10, 10, 10]_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS10/media/3500_fpzrb9l7_1.0.mp4'

def save_obs(video_path):
    pt_path=video_path.replace('.mp4', '.pt')
    # pt_path   ='/data/ryan/code/robo/diffusion_policy/data/output4_train/square_lowdim/v11_square_train_gvp2_velocity_uniform_Tdiscrete_t_minus_one_seed1M/media/20250129_105600__TS10_[10, 10, 10, 10, 10, 10, 10]_square_lowdim_abs_unet_Euler_GVP2_velocity_ODE_Teps0_Seps0_Ssamuniform_____TS10/media/3500_d6bnkcra_1.0.pt'
    #%%
    unique_name=video_path.split('/')[-1].split('.mp4')[0]
    unique_name

    #%%
    origin=video_path.split('media')[0]
    origin
    #%%
    origin1=origin.replace('output4_train','output4_obs')
    #%%
    specific=video_path.split('media')[1]
    #%%
    media_path=origin+'media'+specific
    media_path1=origin1+'media'+specific
    # media_path=os.path.join(origin, 'media',specific)
    media_path

    #%%
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    # read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    frames = frames[..., ::-1]

    # read pt
    pt = torch.load(pt_path)

    # %%
    frames.shape
    #%%
    pt
    #%%
    pt.shape
    # # %%
    # frames_per_obs = frames.shape[0] // pt.shape[0]  # e.g., 400 // 50 = 8

    # for i in range(pt.shape[0]):
    #     obs_i = pt[i] 
    #     frame_start = i * frames_per_obs
    #     frame_end   = (i + 1) * frames_per_obs
    #     step_frames = frames[frame_start:frame_end]  # shape: (8, 128, 128, 3)
        
    import os
    import cv2
    import numpy as np
    import torch

    frames_per_obs = frames.shape[0] // pt.shape[0]  # e.g. 8
    # save_dir = "linked_frames_obs"
    save_dir=os.path.join(media_path, 'obs',unique_name)
    save_dir1=os.path.join(media_path1, 'obs',unique_name)
    #%%
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir1, exist_ok=True)
    for i in range(pt.shape[0]):

        # 1) The observation at step i
        obs_i = pt[i].numpy()  # or keep it as torch

        # 2) The slice of frames for step i
        start_idx = i * frames_per_obs
        end_idx   = (i + 1) * frames_per_obs
        step_frames = frames[start_idx:end_idx]  # shape e.g. (8, H, W, 3)

        # For example, pick the last frame in the chunk
        rep_frame = step_frames[-1]

        # 3) Save the representative frame
        # Convert from RGB to BGR if you want to use cv2.imwrite
        rep_frame_bgr = rep_frame[..., ::-1]  
        frame_path = os.path.join(save_dir, f"obs_{i:03d}.png")
        cv2.imwrite(frame_path, rep_frame_bgr)

        frame_path1 = os.path.join(save_dir1, f"obs_{i:03d}.png")
        cv2.imwrite(frame_path1, rep_frame_bgr)
        # Also save the low‐dim obs as a .npy or something
        np.save(os.path.join(save_dir, f"obs_{i:03d}.npy"), obs_i)
        np.save(os.path.join(save_dir1, f"obs_{i:03d}.npy"), obs_i)
    # %%








for i in range(pt.shape[0]):
    print(i)
# %%
step_frames
# %%

i=0
#%%
# 1) The observation at step i
obs_i = pt[i].numpy()  # or keep it as torch
obs_i
#%%
# 2) The slice of frames for step i
start_idx = i * frames_per_obs
end_idx   = (i + 1) * frames_per_obs
step_frames = frames[start_idx:end_idx]  # shape e.g. (8, H, W, 3)

# For example, pick the last frame in the chunk
rep_frame = step_frames[-1]

# 3) Save the representative frame
# Convert from RGB to BGR if you want to use cv2.imwrite
rep_frame_bgr = rep_frame[..., ::-1]  
frame_path = os.path.join(save_dir, f"obs_{i:03d}.png")
cv2.imwrite(frame_path, rep_frame_bgr)

# Also save the low‐dim obs as a .npy or something
np.save(os.path.join(save_dir, f"obs_{i:03d}.npy"), obs_i)
# %%
print("start_idx: ",start_idx)
print("end_idx: ",end_idx)

# %%
step_frames.shape
# %%
# step_frames[8]
# %%
