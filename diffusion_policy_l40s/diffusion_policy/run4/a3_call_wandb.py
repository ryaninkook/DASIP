#%%
import wandb
wandb.login()
#%%
# Initialize the API
api = wandb.Api()
# infer_ode2_v11_pusht_train_vp_velocity_uniform_seed1_v3
# Define the project and entity
project_name = "infer_ode2_v11_pusht_train_vp_velocity_uniform_seed1_v3"
entity = "bluerand3"  # This could be your username or organization name

# Fetch all runs from the project
runs = api.runs(f"{entity}/{project_name}")

# Store final values for the metric
final_values = {}

# Iterate through runs to find the specific graph data
for run in runs:
    print(f"Processing run: {run.id}")
    try:
        # Fetch the history for the specific metric
        history = run.history(keys=["test_mean_score_last10"], pandas=True)
        
        # Get the final value for the metric
        if not history.empty:
            final_value = history["test_mean_score_last10"].iloc[-1]  # Last recorded value
            final_values[run.id] = final_value
            print(f"Run {run.id}: Final value = {final_value}")
        else:
            print(f"Run {run.id}: No data for the specified metric.")
    except Exception as e:
        print(f"Error processing run {run.id}: {e}")

# Print the final values for all runs
print("\nFinal values for 'test_mean_score_last10':")
#%%
for run_id, value in final_values.items():
    # print(f"Run {run_id}: {value}")
    value=round(value,3)
    print("value: ",value)

# %%
