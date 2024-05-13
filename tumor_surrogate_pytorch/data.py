#%%
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
import nibabel as nib


class MyDataset(Dataset):        
    def __init__(self, start = 0, stop=3000):
        self.datasetPath = "/mnt/8tb_slot8/jonas/datasets/tumorSimulationsIvanAtlas/DS_copy/"#"/mnt/8tb_slot8/jonas/datasets/tumorSimulationsIvansDatasetSmallTest/" 
        self.atlasTissue = np.load("/home/jonas/workspace/programs/addon-tumor-surrogate/tumor_surrogate_pytorch/Atlasfiles/anatomy/npzstuffData_0000.npz")['data'][:, :, :, 1]
        self.patients = np.sort(os.listdir(self.datasetPath))[start:stop]
        self.inputDim = 128

    def __len__(self):
        # Return the size of your dataset
        return len(self.patients)

    def __getitem__(self, idx):
        self.allParams, self.allLabels = [], []

        counter = 0
        patient = self.patients[idx]

        params = np.load(self.datasetPath+ str(patient)+"/parameter_tag2.pkl", allow_pickle=True)

        img = np.load(self.datasetPath+ str(patient)+"/Data_0001_thr2.npz")
        
        tumorImg = np.array([img['data']])[:,:self.inputDim , :self.inputDim , :self.inputDim ]

        centerOfMass = scipy.ndimage.center_of_mass(tumorImg[0])
        COMInt = [int(centerOfMass[0]), int(centerOfMass[1]), int(centerOfMass[2])]
        D = float(params['Dw'])
        rho = float(params['rho'])
        T = float(params['Tend'])
        #encode x,y,z as shift from center of mass
        x = float(params['icx']) - COMInt[0] / self.inputDim 
        y = float(params['icy']) - COMInt[1] / self.inputDim 
        z = float(params['icz']) - COMInt[2] / self.inputDim 
        muD = np.sqrt(D*T).astype(np.float32)
        muRho = np.sqrt(rho*T).astype(np.float32)
    	
        # Return a tuple of your data and label at the given index
        atlImg = torch.tensor(self.atlasTissue[:self.inputDim ,:self.inputDim ,:self.inputDim ].astype(np.float32))
        
        rollX =  self.inputDim // 2 - COMInt[0]
        rollY =  self.inputDim // 2 - COMInt[1]
        rollZ =  self.inputDim // 2 - COMInt[2]

        atlImg = atlImg.roll(shifts=(rollX, rollY, rollZ ), dims=(0, 1, 2))
        allAtlasImgs = torch.stack((atlImg, atlImg, atlImg), dim = 0)
        tumorImg = torch.tensor(tumorImg.astype(np.float32)).roll(shifts=(rollX, rollY, rollZ  ), dims=(1, 2, 3))

        #crop to 64
        lowerBound = self.inputDim // 2 -  self.inputDim // 4
        upperBound = self.inputDim // 2 +  self.inputDim // 4

        allAtlasImgs = allAtlasImgs[:, lowerBound:upperBound, lowerBound:upperBound, lowerBound:upperBound]

        tumorImg = tumorImg[:, lowerBound:upperBound, lowerBound:upperBound, lowerBound:upperBound]
        
        # check params should be: torch.tensor([x, y, z, muD, muRho]) not torch.tensor([D, rho, T])
        return  allAtlasImgs,torch.tensor([x, y, z, muD, muRho]), tumorImg

class realPatientsDataset(Dataset):        
    def __init__(self, start = 0, stop=10):
        self.datasetPath = "/mnt/8tb_slot8/jonas/datasets/mich_rec/"
        self.atlasTissue = np.load("/home/jonas/workspace/programs/addon-tumor-surrogate/tumor_surrogate_pytorch/Atlasfiles/anatomy/npzstuffData_0000.npz")['data'][:, :, :, 1]

        self.patients = np.sort(os.listdir(self.datasetPath))[start:stop]
        self.patients = np.sort(os.listdir(self.datasetPath +"rescaled_128_PatientData_SRI"))[start:stop]
        self.inputDim = 128

        #tumorSegmentationPath = directoryPath + "rescaled_128_PatientData_SRI/rec" + patStr + "_pre/"
        #tissuePath = directoryPath + "mich_rec_SRI_S3_maskedAndCut_rescaled_128/rec" + patStr + "_pre/"

    def __len__(self):
        # Return the size of your dataset
        return len(self.patients)

    def __getitem__(self, idx):
        self.allParams, self.allLabels = [], []

        patient = self.patients[idx]

        tumorCore = nib.load(self.datasetPath + "rescaled_128_PatientData_SRI/" + patient + "/tumorCore_flippedCorrectly.nii").get_fdata() >0
        tumorEdema = nib.load(self.datasetPath + "rescaled_128_PatientData_SRI/" + patient + "/tumorFlair_flippedCorrectly.nii").get_fdata() >0

        img = tumorCore * (0.675 - 0.25) + tumorEdema * 0.25
        
        tumorImg = np.array([img])[:,:self.inputDim , :self.inputDim , :self.inputDim ]


        centerOfMass = scipy.ndimage.center_of_mass(tumorImg[0])
        COMInt = [int(centerOfMass[0]), int(centerOfMass[1]), int(centerOfMass[2])]
        D = 0 #float(params['Dw'])
        rho = 0 #float(params['rho'])
        T = 0 # float(params['Tend'])
        #encode x,y,z as shift from center of mass
        x = 0 #float(params['icx']) - COMInt[0] / self.inputDim 
        y = 0 #float(params['icy']) - COMInt[1] / self.inputDim 
        z = 0 #float(params['icz']) - COMInt[2] / self.inputDim 
        muD = np.sqrt(D*T).astype(np.float32)
        muRho = np.sqrt(rho*T).astype(np.float32)
    	
        # Return a tuple of your data and label at the given index
        atlasImg = torch.tensor(self.atlasTissue[:self.inputDim ,:self.inputDim ,:self.inputDim ].astype(np.float32))

        wm = nib.load(self.datasetPath + "mich_rec_SRI_S3_maskedAndCut_rescaled_128/" + patient + "/WM_flippedCorrectly.nii").get_fdata()
        gm = nib.load(self.datasetPath + "mich_rec_SRI_S3_maskedAndCut_rescaled_128/" + patient + "/GM_flippedCorrectly.nii").get_fdata()
        allTissue = np.abs(0.2 * wm + 0.1 * gm)
        allTissue[allTissue < 0.03] = 0
        tisImg = torch.tensor(allTissue[:self.inputDim ,:self.inputDim ,:self.inputDim ].astype(np.float32))
        
        rollX =  self.inputDim // 2 - COMInt[0]
        rollY =  self.inputDim // 2 - COMInt[1]
        rollZ =  self.inputDim // 2 - COMInt[2]

        tisImg = tisImg.roll(shifts=(rollX, rollY, rollZ ), dims=(0, 1, 2))
        tissueImg = torch.stack((tisImg, tisImg, tisImg), dim = 0)
        tumorImg = torch.tensor(tumorImg.astype(np.float32)).roll(shifts=(rollX, rollY, rollZ  ), dims=(1, 2, 3))

        # dont look at segmentation in CSF
        tumorImg[:,tisImg < 0.0001] = 0

        #crop to 64
        lowerBound = self.inputDim // 2 -  self.inputDim // 4
        upperBound = self.inputDim // 2 +  self.inputDim // 4

        tissueImg = tissueImg[:, lowerBound:upperBound, lowerBound:upperBound, lowerBound:upperBound]

        tumorImg = tumorImg[:, lowerBound:upperBound, lowerBound:upperBound, lowerBound:upperBound]
        
        return  tissueImg, torch.tensor([x, y, z, muD, muRho]), tumorImg

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    '''# path = '/mnt/Drive2/ivan/data/tumor_mparam/v/'
    data_dir = '/mnt/Drive2/ivan/data'
    dataset = 'tumor_mparam/v/' #or valid
    dataset = TumorDataset(data_dir, dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (x, y, z) in enumerate(loader):
        if i == 100:
            break
    '''
    '''
    dataset = MyDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (x, y, z) in enumerate(loader):
        if i == 100:
            break
        print(x.shape, y.shape, z.shape)

        plt.imshow(x[0, 0, :, :, 32].cpu())

        plt.show()
        plt.imshow(z[0, 0, :, :, 32].cpu())
        break
    '''
    dataset = realPatientsDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (x, y, z) in enumerate(loader):
        if i == 10:
            break
        print(x.shape, y.shape, z.shape)

        plt.imshow(x[0, 0, :, :, 32].cpu(), cmap="gray")
        
        plt.imshow(z[0, 0, :, :, 32].cpu(), cmap="Reds", alpha= z[0, 0, :, :, 32].cpu())
        plt.show()
        
        plt.hist(x.numpy().flatten(), 100)
        plt.show()
# %%
