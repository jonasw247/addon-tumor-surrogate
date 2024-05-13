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


#%% t1 and nec over flair 
volumeT1 = np.array(volumeT1)
volumeFlair = np.array(volumeFlair)
volumeNec = np.array(volumeNec)
totalVolume = np.array(totalVolume)

volumeRatio = (volumeT1 +volumeNec) /totalVolume

plt.hist(volumeRatio, bins=30)

#%% search
pathToSynthetic = "/mnt/8tb_slot8/jonas/workingDirDatasets/synthetic_FK_Michals_solver/"
folders = os.listdir(pathToSynthetic)
#%%
volumeT1, volumeFlair, volumeNec, totalVolume, dicts = [], [], [], [], []
for folder in folders[:500]:
    if "." in folder:
        continue
    patientID = folder.split("_")[-1]

    tumorCPath = os.path.join(pathToSynthetic, folder, "tumor_concentration.nii.gz")
    dictPAth = os.path.join(pathToSynthetic, folder, "saveDict.npy")
    tumorC = nib.load(tumorCPath).get_fdata()
    dict = np.load(dictPAth, allow_pickle=True).item()

    dicts.append(dict)

    volumeNec.append(0)
    thflair = 0.1
    thT1 = 0.95
    #volumeFlair.append(np.sum(tumorC >= thflair))
    volumeT1.append(np.sum(tumorC >=thT1))
    totalVolume.append(np.sum(tumorC >= thflair))

volumeT1 = np.array(volumeT1)
volumeFlair = np.array(volumeFlair)
volumeNec = np.array(volumeNec)
totalVolume = np.array(totalVolume)
#%%
ratioT1overTotal = volumeT1 / totalVolume
plt.hist(ratioT1overTotal, bins=30, alpha=0.5, label='T1')
#%%
bins = 30
plt.hist(volumeT1, bins=bins, alpha=0.5, label='T1')
plt.hist(volumeFlair, bins=bins, alpha=0.5, label='Flair')
plt.hist(volumeNec, bins=bins, alpha=0.5, label='Nec')
plt.hist(totalVolume, bins=bins, alpha=0.5, label='Total')
plt.legend(loc='upper right')
plt.show()
#%%
def logisticGrowth(x):
    return 1 / (1 + 100* np.exp(- x))

x = np.linspace(-10, 10, 100)
y = logisticGrowth(x)
plt.plot(x, y)

def diffusion_dominated_growth(x):
    return 1 - np.exp(- x)
#%%
def grwothFunction(x):
    return (1-x)

val = 0.1
y = []
for i in range(1000):
    y.append(val)
    val = val + grwothFunction(val) *0.01

plt.plot(y)
#%%
print("T1: ", np.mean(volumeT1), "+-", np.std(volumeT1) / np.sqrt(len(volumeT1)))
# %%
print("Flair: ", np.mean(volumeFlair), "+-", np.std(volumeFlair) / np.sqrt(len(volumeFlair)))
# %%
print("Nec: ", np.mean(volumeNec), "+-", np.std(volumeNec) / np.sqrt(len(volumeNec)))

plt.hist(volumeT1)  
# %%
