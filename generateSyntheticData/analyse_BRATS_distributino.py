#%%
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
"/mnt/8tb_slot8/jonas/datasets/brats/BraTS2021_00545/preop/sub-BraTS2021_00545_ses-preop_space-sri_seg.nii.gz"
bratsDatasetPath = "/mnt/8tb_slot8/jonas/datasets/brats/"

folders = os.listdir(bratsDatasetPath)
# %%
volumeT1, volumeFlair, volumeNec, totalVolume = [], [], [], []
for folder in folders:
    if "." in folder:
        continue
    patientID = folder.split("_")[-1]
    segPath = os.path.join(bratsDatasetPath, folder, "preop", "sub-BraTS2021_"+patientID+"_ses-preop_space-sri_seg.nii.gz")
    seg = nib.load(segPath).get_fdata()

    volumeNec.append(np.sum(seg == 1))
    volumeFlair.append(np.sum(seg == 2))
    volumeT1.append(np.sum(seg == 4))
    totalVolume.append(np.sum(seg != 0))

# %% plot histogram outlines
bins = 30
plt.hist(volumeT1, bins=bins, alpha=0.5, label='T1')
plt.hist(volumeFlair, bins=bins, alpha=0.5, label='Flair')
plt.hist(volumeNec, bins=bins, alpha=0.5, label='Nec')
plt.hist(totalVolume, bins=bins, alpha=0.5, label='Total')
plt.legend(loc='upper right')
plt.show()

#%%
print("T1: ", np.mean(volumeT1), "+-", np.std(volumeT1) / np.sqrt(len(volumeT1)))
# %%
print("Flair: ", np.mean(volumeFlair), "+-", np.std(volumeFlair) / np.sqrt(len(volumeFlair)))
# %%
print("Nec: ", np.mean(volumeNec), "+-", np.std(volumeNec) / np.sqrt(len(volumeNec)))

plt.hist(volumeT1)  
# %%
