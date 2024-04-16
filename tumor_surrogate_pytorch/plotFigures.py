#%%
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# %%
pat = "rec020_pre"
z = 30
pat = "rec006_pre"
z = 32
#pat = "rec034_pre"
#z = 32

path = "/mnt/8tb_slot8/jonas/workingDirDatasets/addon-tumor-surrogate-output/fixOrigin/realData/optimizeOutputPatients/"+pat+"/"

prediction_masked = np.flip(nib.load(path + "prediction_masked.nii.gz").get_fdata(), axis=0)
prediction_smooth = np.flip(nib.load(path + "tumorPlot.nii.gz").get_fdata(), axis=0)
output_ground_truth = np.flip(nib.load(path + "output_ground_truth.nii.gz").get_fdata(), axis=0)

outoutPath = "./figurePlots/"+pat+"/"
os.makedirs(outoutPath, exist_ok=True)

# %%
plt.imshow(output_ground_truth[:,:,z]**0.5, cmap="Greens", alpha=0.7 * output_ground_truth[:,:,z]**0.00001 )
plt.savefig(outoutPath + "output_ground_truth.svg")
# %%
# %%
start_color = '#C1F7CB'  
#start_color = "#E1FBE6"
#start_color = "#BEE0C4"
end_color = "#094413"#'#086218'    
custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', [start_color, end_color])

plt.imshow(prediction_smooth[:,:,z], cmap=custom_cmap, alpha=0.8*(prediction_smooth[:,:,z] >0.01))
plt.colorbar()

plt.savefig(outoutPath + "prediction_smooth.svg")

# %% difference
toplt = np.abs(prediction_masked[:,:,z] - output_ground_truth[:,:,z])
plt.imshow(toplt, cmap="Reds", alpha=toplt)
plt.colorbar()


# %%
