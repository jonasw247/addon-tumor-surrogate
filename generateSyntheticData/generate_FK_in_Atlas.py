#%%
from TumorGrowthToolkit.FK import Solver as FKSolver
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import time
import scipy.ndimage
import nibabel as nib

#wm_data = nib.load('dataset/WM.nii.gz').get_fdata()
#gm_data = nib.load('dataset/GM.nii.gz').get_fdata()

data = nib.load("../DTIAtlas/sub-mni152_tissues_space-sri.nii.gz").get_fdata()

wm_data = data == 3
gm_data = data == 2

#%%
z = 90
plt.imshow(wm_data[:,:,z] + 0.5*gm_data[:,:,z], cmap='gray')

plt.colorbar()
#%%
origin = np.random.rand(3)

origin[0] = 0

#check if origin is in the white or gray matter
intOrigin = (origin * gm_data.shape).astype(int)
if gm_data[intOrigin[0], intOrigin[1], intOrigin[2]] == 0 and wm_data[intOrigin[0], intOrigin[1], intOrigin[2]]== 0:
    print("Origin is in the background")



#%%
# Set up parameters
parameters = {
    'Dw': 1.0,          # Diffusion coefficient for white matter
    'rho': 0.10,         # Proliferation rate
    'RatioDw_Dg': 10,  # Ratio of diffusion coefficients in white and grey matter
    'gm': gm_data,      # Grey matter data
    'wm': wm_data,      # White matter data
    'NxT1_pct': 0.3,    # tumor position [%]
    'NyT1_pct': 0.7,
    'NzT1_pct': 1,
    'init_scale': 1., #scale of the initial gaussian
    'resolution_factor': 1, #resultion scaling for calculations
    'th_matter': 0.1, #when to stop diffusing: at th_matter > gm+wm
    'verbose': True, #printing timesteps 
    'time_series_solution_Nt': None,#64, #64, # number of timesteps in the output
    'stopping_volume' : 1000, #stop when the volume of the tumor is less than this value
    'stopping_time' : 10000, #stop when the time is greater than this value
}

# %%
# Create custom color maps
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['black', 'white'], 256)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap2', ['black', 'green', 'yellow', 'red'], 256)

# Calculate the slice index
NzT = int(parameters['NzT1_pct'] * gm_data.shape[2])

# Plotting function
def plot_tumor_states(wm_data, initial_state, final_state, slice_index):
    plt.figure(figsize=(12, 6))

    # Plot initial state
    plt.subplot(1, 2, 1)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(initial_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Initial Tumor State")

    # Plot final state
    plt.subplot(1, 2, 2)
    plt.imshow(wm_data[:, :, slice_index], cmap=cmap1, vmin=0, vmax=1, alpha=1)
    plt.imshow(final_state[:, :, slice_index], cmap=cmap2, vmin=0, vmax=1, alpha=0.65)
    plt.title("Final Tumor State")
    plt.show()
    

# %%
# Run the FK_solver and plot the results
start_time = time.time()
fk_solver = FKSolver(parameters)
result = fk_solver.solve()
end_time = time.time()  # Store the end time
execution_time = int(end_time - start_time)  # Calculate the difference

print(f"Execution Time: {execution_time} seconds")
if result['success']:
    print("Simulation successful!")
    plot_tumor_states(wm_data, result['initial_state'], result['final_state'], NzT)
else:
    print("Error occurred:", result['error'])
# %%
