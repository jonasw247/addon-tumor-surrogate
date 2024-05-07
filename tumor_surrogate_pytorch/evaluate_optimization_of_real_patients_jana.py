#%%
import numpy as np
import torch
import os
freeOrigin = False#False
synthetic  =True

if freeOrigin:
    res_folder = "/mnt/8tb_slot8/jonas/workingDirDatasets/addon-tumor-surrogate-output/freeOrigin/realData/optimizeOutputPatients/"
else:
    res_folder = "/mnt/8tb_slot8/jonas/workingDirDatasets/addon-tumor-surrogate-output/fixOrigin/realData/optimizeOutputPatients/"

if synthetic:
    res_folder = res_folder.replace("realData", "synthetic")

# %%
folders = np.sort(os.listdir(res_folder))
# %%
alldicesEnhancing, alldicesEdema, usedPatients, runtimes = [], [], [], []
for folder in folders:
    #if not "rec" in folder:
    #    continue
    if "30" in folder:
        continue
    print(folder)
    savedDic = torch.load(os.path.join(res_folder, folder, "logDict.pth"))
    logDicts = savedDic["logDicts"]
    dicesEnhancing, dicesEdema = [], []
    for logDict in logDicts:
        dicesEnhancing.append(logDict["dice_0.67"])
        dicesEdema.append(logDict["dice_0.24"])
    usedPatients.append(folder)
    alldicesEnhancing.append(dicesEnhancing)
    alldicesEdema.append(dicesEdema)
    runtimes.append(savedDic["runtime"])

alldicesEnhancing = np.array(alldicesEnhancing)
alldicesEdema = np.array(alldicesEdema)
# %%
import matplotlib.pyplot as plt
for i in range(len(alldicesEnhancing)):
    plt.plot(alldicesEnhancing[i], label=usedPatients[i])
#plt.plot(np.array(alldicesEdema).T, label="Enhancing")
plt.legend()
#%%
for i in range(len(alldicesEnhancing)):
    plt.plot(alldicesEdema[i], label=usedPatients[i])
#plt.plot(np.array(alldicesEdema).T, label="Enhancing")
plt.legend()
# %%
finalEnhancing = alldicesEnhancing.T[-1]
finalEdema = alldicesEdema.T[-1]
   
# %%
print("Enhancing: ", round(np.mean(finalEnhancing), 2), "+-", round(np.std(finalEnhancing) / np.sqrt(len(finalEnhancing)), 2))
print("Edema: ", round(np.mean(finalEdema), 2), "+-", round(np.std(finalEdema)/np.sqrt(len(finalEdema)), 2))
#rounded runtime in minutes
print("Runtime: ", round(np.mean(runtimes) / 60, 3), "+-", round(np.std(runtimes) / 60 / np.sqrt(len(runtimes)), 3))

# %%
