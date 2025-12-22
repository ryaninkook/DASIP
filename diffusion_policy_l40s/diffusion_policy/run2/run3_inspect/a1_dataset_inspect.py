#%%
import h5py

dataset_path = 'path_to_your_dataset.hdf5'
#data/robomimic/datasets/${task.task_name}/${task.dataset_type}/low_dim_abs.hdf5
dataset_path='/data/ryan/code/robo/diffusion_policy/data/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5'
with h5py.File(dataset_path, 'r') as f:
    data_group = f['data']
    demo_keys = list(data_group.keys())
    print("Available demos:", demo_keys)
    
    # Inspect the first demo
    demo = data_group[demo_keys[0]]
    obs_group = demo['obs']
    obs_keys = list(obs_group.keys())
    print("\nAvailable observation keys:")
    for key in obs_keys:
        data = obs_group[key]
        print(f"Key: {key}, Shape: {data.shape}, Dtype: {data.dtype}")

# %%
self.T=None


if self.T:
    print('hi')
#%%
def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {obj.name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {obj.name}")
with h5py.File(dataset_path, 'r') as f:
    # Traverse the entire file
    f.visititems(print_hdf5_structure)

# %%

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {obj.name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {obj.name}")
with h5py.File(dataset_path, 'r') as f:
    # Traverse the entire file
    f.visititems(print_hdf5_structure)

# %%
