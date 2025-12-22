
from PIL import Image
import re
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil,ast,subprocess,requests
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt

import wandb

import os
import torch
import copy
from collections import deque
from collections import deque

class Average_past:
    def __init__(self, maxlen,average_all=False):
        # Use deque to store up to maxlen values
        self.values = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.average_all=average_all

    def update(self, new_value):
        # Add the new value to the deque (old values are removed automatically if the max length is exceeded)
        self.values.append(new_value)

    @property
    def value(self):
        # Return 0 if no values have been stored
        if len(self.values) == 0:
            return 0
        # Return the last value if less than maxlen values have been stored
        if self.average_all:
            return sum(self.values) / len(self.values)
        else:
            if len(self.values) < self.maxlen:
                return self.values[-1]  # Return the last value added
        # Otherwise, calculate the average of all values in the deque
        return sum(self.values) / len(self.values)

# class Average_past:
#     def __init__(self,maxlen):
#         # Use deque to store up to 10 values, with a maximum length of 10
#         self.values = deque(maxlen=maxlen)

#     def update(self, new_value):
#         # Add the new value to the deque (old values are removed automatically if the max length is exceeded)
#         self.values.append(new_value)

#     @property
#     def value(self):
#         # Calculate the average of all values in the deque
#         if len(self.values) == 0:
#             return 0  # Handle the case where no values have been recorded
#         return sum(self.values) / len(self.values)

def load_policy(policy,ema_policy, optimizer, device,pyname,optimizer_to=None):
    # Get the current Python file name
    
    
    # Define the base folder for the model
    base_folder = 'data/model/'
    folder2 = os.path.join(base_folder, pyname)

    # Check if the folder exists
    if not os.path.exists(folder2):
        print('Folder does not exist')
        return policy, optimizer,ema_policy  # Return unmodified policy and optimizer if folder doesn't exist
    else:
        print('Folder exists')
        data2 = os.listdir(folder2)
        
        # Check if there are any files in the folder
        if len(data2) == 0:
            print('No model files found in the folder')
            return policy, optimizer,ema_policy  # Return unmodified policy and optimizer if no files

        # Sort files and load the latest one
        data2 = sorted(data2)
        pth1 = os.path.join(folder2, data2[-1])  # Get the latest model file path

        # Load the model state
        policy = policy.to('cpu')  # Move policy to CPU before loading
        policy.load_state_dict(torch.load(pth1, map_location='cpu'))  # Load the model state
        policy = policy.to(device)  # Move policy back to the specified device
        
        # Create a copy of the policy for EMA (Exponential Moving Average)
        # ema_policy = copy.deepcopy(policy).to(device)
        
        # Move optimizer to the specified device (assuming you have defined optimizer_to)
        if optimizer_to != None:
            optimizer_to(optimizer, device)  # You may want to define this function if not already done
        
        print(f'Model loaded from {pth1}')
    
    return policy, optimizer,ema_policy

import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate

def save_single_image(title, img, a_true, a_hat, agent_pos, script_name, epoch, mse,show=False):
    
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(a_hat, np.ndarray):
        a_hat = torch.tensor(a_hat)
    
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
    
    if isinstance(agent_pos, np.ndarray):
        agent_pos = torch.tensor(agent_pos)
    
    if a_true is not None and isinstance(a_true, np.ndarray):
        a_true = torch.tensor(a_true)
    
    # Check if inputs have a batch dimension, if not, add one
    if len(a_hat.shape) == 2:
        a_hat = a_hat.unsqueeze(0)  # Make it 1 x 16 x 2
    
    if len(img.shape) == 4:
        img = img.unsqueeze(0)  # Make it 1 x 2 x 3 x 96 x 96
    
    if len(agent_pos.shape) == 2:
        agent_pos = agent_pos.unsqueeze(0)  # Make it 1 x 2 x 2

    # Create a folder to save the images if it doesn't exist
    base_save_folder = f'data/grid/{script_name}'
    os.makedirs(base_save_folder, exist_ok=True)

    # Create a figure for the single image
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figure size as needed

    # Process the image (taking the first observation)
    img1 = img[0][0].detach().cpu()  # Use the first image in the batch
    img1_transposed = img1.permute(1, 2, 0).numpy()  # Transpose to [96, 96, 3]
    img1_transposed = img1_transposed.astype(np.uint8)
    img_pil = Image.fromarray(img1_transposed)
    img_resized = img_pil.resize((512, 512), Image.BILINEAR)
    img_resized = np.array(img_resized)

    # Process the agent position
    pos1 = agent_pos[0][0].detach().cpu().numpy()

    # Plot the image and scatter the agent position
    ax.imshow(img_resized)
    ax.scatter(pos1[0], pos1[1], color='blue', label='Agent Position', s=100)

    # Conditionally draw a_true if it's not None
    if a_true is not None:
        a_true2 = a_true[0].detach().cpu().numpy()
        for action in a_true2:
            ax.scatter(action[0], action[1], color='green')
        ax.scatter(a_true2[0][0], a_true2[0][1], color='green', label='a_true')

    # Always draw a_hat
    a_hat2 = a_hat[0].detach().cpu().numpy()
    for action in a_hat2:
        ax.scatter(action[0], action[1], color='red')
    ax.scatter(a_hat2[0][0], a_hat2[0][1], color='red', label='a_hat')

    # Calculate MSE if a_true is not None
    if a_true is not None:
        a_true2 = torch.tensor(a_true2).unsqueeze(0)
        a_hat2 = torch.tensor(a_hat2).unsqueeze(0)
        mse_each = torch.nn.functional.mse_loss(a_true2, a_hat2)
        mse_each = round(mse_each.item(), 2)
        ax.set_title(f'{script_name}_{title}, MSE: {mse_each}')
    else:
        ax.set_title(f'{script_name}_{title}')

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()

    # Save the figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mse = round(mse, 2)
    filename = f'{timestamp}_{epoch}_{mse}.png'
    fullpath = os.path.join(base_save_folder, filename)
    plt.savefig(fullpath)
    plt.close(fig)  # Close the figure to free up memory
    if show:
        return fig

def print_nested_structure(element, prefix=""):
    if isinstance(element, dict):
        for k, v in element.items():
            print(f"{prefix}{k}:")
            print_nested_structure(v, prefix=prefix + "  ")
            print('----------------')
    else:
        try:
            print(f"{prefix}Shape: {element.shape}")

            # print(f"{prefix}Value: {element.numpy()}")
            print('----------------')
        except AttributeError:
            print(f"{prefix}Shape: {getattr(element, 'shape', 'N/A')}")
            # print(f"{prefix}Value: {element}")

            print('----------------')
import time
import subprocess
def find_empty_gpu_v2():
    try:
        # Run the command to get the memory usage of all GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE
        )
        
        # Parse the result
        memory_free = [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]
        
        # Find the GPU with the most free memory
        gpu_id = memory_free.index(max(memory_free))
        return f'cuda:{int(gpu_id)}'
    except Exception as e:
        print(f"Error while finding an empty GPU: {e}")
        return 0  # Default to GPU 0 if there's an issue


from einops import rearrange, reduce
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
def remove_prefix(script_name):
    # Use regular expression to remove the prefix if it starts with 'a' followed by digits and an underscore
    return re.sub(r'^a\d+_', '', script_name)


import torch.fft
def high_frequency_loss(pred_actions, weight=1.0):
    """
    Penalizes high-frequency oscillations in the predicted actions.

    Args:
        pred_actions (torch.Tensor): The predicted actions. Shape: (batch_size, time_steps, action_dim)
        weight (float): Weight for the high-frequency loss.

    Returns:
        torch.Tensor: The high-frequency loss.
    """
    # Move tensor to CPU for FFT calculation
    pred_actions_cpu = pred_actions.cpu()
    
    # Apply Fourier Transform along the time dimension
    fft_actions = torch.fft.fft(pred_actions_cpu, dim=1)
    
    # Calculate the power spectrum (magnitude of the Fourier coefficients)
    power_spectrum = torch.abs(fft_actions) ** 2
    
    # Define a filter that penalizes high frequencies more
    freqs = torch.fft.fftfreq(pred_actions_cpu.shape[1])
    high_freq_penalty = torch.abs(freqs) ** 2  # Squared frequency penalty

    # Apply the high-frequency penalty to the power spectrum
    high_freq_loss = (power_spectrum * high_freq_penalty.unsqueeze(-1)).mean()

    # Scale the loss by the weight
    return weight * high_freq_loss.to(pred_actions.device)  # Move the result back to the original device


def smoothness_loss( a_hat):    
    smoothness_pred = a_hat[:, 1:] - a_hat[:, :-1]
    smoothness_loss = (smoothness_pred ** 2).mean()

    return smoothness_loss

import torch.nn.functional as F
def velocity_loss(y_pred, y_true):
    velocity_pred = y_pred[:, 1:] - y_pred[:, :-1]
    velocity_true = y_true[:, 1:] - y_true[:, :-1]
    return F.mse_loss(velocity_pred, velocity_true)





def save_grid_v2(title,img,a_true,a_hat,agent_pos,script_name,epoch,mse):
    
    # Create a folder to save the images if it doesn't exist

    base_save_folder = f'data/grid/{script_name}'
    os.makedirs(base_save_folder, exist_ok=True)
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Assuming img, agent_pos, stats, a_true, and a_hat are defined elsewhere

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for count, i in enumerate([0, 1, 2, 3]):
        # Process the image
        if len(img.shape)==5:
            img1 = img[i][0].detach().cpu()
        elif len(img.shape)==4:
            img1 = img[i].detach().cpu()
        img1_transposed = img1.permute(1, 2, 0).numpy()  # Transpose to [96, 96, 3]
        img1_transposed = img1_transposed.astype(np.uint8)
        img_pil = Image.fromarray(img1_transposed)
        img_resized = img_pil.resize((512, 512), Image.BILINEAR)
        img_resized = np.array(img_resized)

        # Process the positions and actions
        
        # agent_pos2 = unnormalize_data_v2(agent_pos, stats['agent_pos'])
        pos1 = agent_pos[i][0].detach().cpu().numpy()
        a_true2 = a_true[i].detach().cpu().numpy()
        a_hat2 = a_hat[i].detach().cpu().numpy()

        # Access the appropriate subplot in the grid
        ax = axs[count // 2, count % 2]  
        
        # Plot the image and scatter points
        ax.imshow(img_resized)
        ax.scatter(pos1[0], pos1[1], color='blue', label='Agent Position', s=100)

        for action in a_true2:
            ax.scatter(action[0], action[1], color='green')
        ax.scatter(action[0], action[1], color='green', label='a_true')

        for action in a_hat2:
            ax.scatter(action[0], action[1], color='red')
        ax.scatter(action[0], action[1], color='red', label='a_hat')

        a_true2 = torch.tensor(a_true2).unsqueeze(0)
        a_hat2 = torch.tensor(a_hat2).unsqueeze(0)
        mse_each = torch.nn.functional.mse_loss(a_true2, a_hat2)
        mse_each = round(mse_each.item(), 2)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{script_name}_{title}, MSE: {mse_each}')
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    # plt.savefig("grid_of_images.png")

    # Display the plot
    # plt.show()

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mse=round(mse,2)
    filename = f'{timestamp}_{epoch}_{mse}.png'
    fullpath = os.path.join(base_save_folder, filename)
    plt.savefig(fullpath)
    plt.close(fig)  # Close t



def save_grid_v3(title,img,a_true,a_hat,agent_pos,script_name,epoch,mse):
    
    # Create a folder to save the images if it doesn't exist

    base_save_folder = f'data/grid/{script_name}'
    os.makedirs(base_save_folder, exist_ok=True)
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Assuming img, agent_pos, stats, a_true, and a_hat are defined elsewhere

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for count, i in enumerate([0, 1, 2, 3]):
        # Process the image
        if len(img.shape)==5:
            img1 = img[i][0].detach().cpu()
        elif len(img.shape)==4:
            img1 = img[i].detach().cpu()
        img1_transposed = img1.permute(1, 2, 0).numpy()  # Transpose to [96, 96, 3]
        img1_transposed = img1_transposed.astype(np.uint8)
        img_pil = Image.fromarray(img1_transposed)
        img_resized = img_pil.resize((512, 512), Image.BILINEAR)
        img_resized = np.array(img_resized)

        # Process the positions and actions
        
        # agent_pos2 = unnormalize_data_v2(agent_pos, stats['agent_pos'])
        pos1 = agent_pos[i][0].detach().cpu().numpy()
        a_true2 = a_true[i].detach().cpu().numpy()
        a_hat2 = a_hat[i].detach().cpu().numpy()

        # Access the appropriate subplot in the grid
        ax = axs[count // 2, count % 2]  
        
        # Plot the image and scatter points
        ax.imshow(img_resized)
        ax.scatter(pos1[0], pos1[1], color='blue', label='Agent Position', s=100)

        for action in a_true2:
            ax.scatter(action[0], action[1], color='green')
        ax.scatter(action[0], action[1], color='green', label='a_true')

        for action in a_hat2:
            ax.scatter(action[0], action[1], color='red')
        ax.scatter(action[0], action[1], color='red', label='a_hat')

        a_true2 = torch.tensor(a_true2).unsqueeze(0)
        a_hat2 = torch.tensor(a_hat2).unsqueeze(0)
        mse_each = torch.nn.functional.mse_loss(a_true2, a_hat2)
        mse_each = round(mse_each.item(), 2)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{script_name}_{title}, MSE: {mse_each}')
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    # plt.savefig("grid_of_images.png")

    # Display the plot
    # plt.show()

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mse=round(mse,2)
    filename = f'{timestamp}_{epoch}_{mse}_{title}.png'
    fullpath = os.path.join(base_save_folder, filename)
    plt.savefig(fullpath)
    plt.close(fig)  # Close t


def save_grid_v4(title,img,a_true,a_hat,agent_pos,epoch,mse,base_save_folder):
    
    # Create a folder to save the images if it doesn't exist
    base_save_folder=os.path.join(base_save_folder,'grid')
    # base_save_folder = f'data/grid/{script_name}'
    os.makedirs(base_save_folder, exist_ok=True)
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    # Assuming img, agent_pos, stats, a_true, and a_hat are defined elsewhere

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    for count, i in enumerate([0, 1, 2, 3]):
        # Process the image
        if len(img.shape)==5:
            img1 = img[i][0].detach().cpu()
        elif len(img.shape)==4:
            img1 = img[i].detach().cpu()
        img1_transposed = img1.permute(1, 2, 0).numpy()  # Transpose to [96, 96, 3]
        img1_transposed = img1_transposed.astype(np.uint8)
        img_pil = Image.fromarray(img1_transposed)
        img_resized = img_pil.resize((512, 512), Image.BILINEAR)
        img_resized = np.array(img_resized)

        # Process the positions and actions
        
        # agent_pos2 = unnormalize_data_v2(agent_pos, stats['agent_pos'])
        pos1 = agent_pos[i][0].detach().cpu().numpy()
        a_true2 = a_true[i].detach().cpu().numpy()
        a_hat2 = a_hat[i].detach().cpu().numpy()

        # Access the appropriate subplot in the grid
        ax = axs[count // 2, count % 2]  
        
        # Plot the image and scatter points
        ax.imshow(img_resized)
        ax.scatter(pos1[0], pos1[1], color='blue', label='Agent Position', s=100)

        for action in a_true2:
            ax.scatter(action[0], action[1], color='green')
        ax.scatter(action[0], action[1], color='green', label='a_true')

        for action in a_hat2:
            ax.scatter(action[0], action[1], color='red')
        ax.scatter(action[0], action[1], color='red', label='a_hat')

        a_true2 = torch.tensor(a_true2).unsqueeze(0)
        a_hat2 = torch.tensor(a_hat2).unsqueeze(0)
        mse_each = torch.nn.functional.mse_loss(a_true2, a_hat2)
        mse_each = round(mse_each.item(), 2)

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{title}, MSE: {mse_each}')
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    # plt.savefig("grid_of_images.png")

    # Display the plot
    # plt.show()

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mse=round(mse,2)
    filename = f'{timestamp}_{epoch}_{mse}_{title}.png'
    fullpath = os.path.join(base_save_folder, filename)
    plt.savefig(fullpath)
    plt.close(fig)  # Close t
import os
import shutil
import json
from datetime import datetime

import os
import shutil
import json
from datetime import datetime

# Helper function to get current timestamp
def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
def path_join(*list1):
    return os.path.join(*list1)

class Minimum_check():
    def __init__(self):
        self.value = float('inf')  # Initialize with infinity, so any loss will be smaller

    def update(self, v):
        # Check if v is a tensor, and if so, get its item (float) value
        if isinstance(v, torch.Tensor):
            v = v.item()
        
        # Check if the value is numeric (float, int)
        if isinstance(v, (float, int)):
            # Update the stored value if the new value is smaller
            if v < self.value:
                self.value = v
        return self.value

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0  # value for the last update
        self.avg = 0  # average of accumulated values
        self.sum = 0  # sum of accumulated values
        self.count = 0  # number of accumulated values

    def update(self, val, n=1):
        if isinstance(val, float):
            pass
        elif isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print2(*args):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):  # Check if the argument is a float
            formatted_arg = f'{round(arg, 3):.3f}'  # Round and format the float
        else:
            formatted_arg = str(arg)
        formatted_args.append(formatted_arg)
    print('\t'.join(formatted_args))

def reset_index(df):
    df=df.reset_index()    
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])
    return df

def read_excel(path):
    if path.endswith(".csv"):

        df=pd.read_csv(path,index_col=0)
    else:
        df=pd.read_excel(path,index_col=0)
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])    
    return df


def print1(content_dict,filename):
    if '.txt' not in filename:
        filename=filename+'.txt'
    path='data/log/'
    if not os.path.exists(path):
        os.makedirs(path)
    fullpath=path+filename
    timedelta1=0
    timedelta2=100
    script_name = os.path.basename(__file__)
    today1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1)).strftime("%Y-%m-%d %H:%M:%S"))
    today2=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta1))).timestamp()
    past1=((datetime.now(timezone('US/Eastern'))-timedelta(timedelta2)).strftime("%Y-%m-%d"))
    past2=datetime.strptime('20210218', "%Y%m%d").timestamp()

    with open(fullpath,'a+') as f:
        f.write(str(content_dict))
        f.write(f' <-- {today1} {script_name} -->')   
        f.write('\n') 


#%%
import psutil

# Function to count all Python processes
def count_all_python_processes():
    processes = psutil.process_iter(['pid', 'name', 'cmdline'])
    python_process_count = 0

    for process in processes:
        try:
            if 'python' in process.info['name'] or any('python' in cmd for cmd in process.info['cmdline']):
                python_process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return python_process_count

# Function to count main Python processes
def count_main_python_processes():
    processes = psutil.process_iter(['pid', 'name', 'cmdline', 'ppid'])
    python_process_count = 0

    for process in processes:
        try:
            if 'python' in process.info['name'] or any('python' in cmd for cmd in process.info['cmdline']):
                parent = psutil.Process(process.info['ppid']) if process.info['ppid'] else None
                if not parent or 'python' not in parent.name():
                    python_process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return python_process_count
def count_active_running_python_processes():
    processes = psutil.process_iter(['pid', 'name', 'cmdline', 'status'])
    python_process_count = 0

    for process in processes:
        try:
            # Check if the process name or command line contains 'python'
            if 'python' in process.info['name'] or any('python' in cmd for cmd in process.info['cmdline']):
                # Filter for processes that are in the 'R' (running) state
                if process.info['status'] == psutil.STATUS_RUNNING:
                    python_process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass  # Ignore processes that no longer exist or we don't have access to

    return python_process_count
# Function to count actively running Python processes
def count_active_python_processes():
    processes = psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'cpu_percent'])
    python_process_count = 0

    for process in processes:
        try:
            if 'python' in process.info['name'] or any('python' in cmd for cmd in process.info['cmdline']):
                if process.info['status'] == psutil.STATUS_RUNNING or process.info['cpu_percent'] > 0:
                    python_process_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return python_process_count

# Function to print all counts
def print_process_counts():
    all_python = count_all_python_processes()
    main_python = count_main_python_processes()
    active_python = count_active_python_processes()
    run_python=count_active_running_python_processes()
    # print(f"Number of all Python processes: {all_python}")
    # print(f"Number of main Python processes: {main_python}")
    # print(f"Number of active Python processes: {active_python}")
    # print("count_active_running_python_processes(): ",count_active_running_python_processes())
    return all_python, main_python, active_python,run_python

# %%
