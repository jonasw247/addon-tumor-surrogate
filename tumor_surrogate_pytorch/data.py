#%%
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage


class TumorDataset(Dataset):
    def __init__(self, data_path, dataset):
        data_dir = os.path.join(data_path, dataset)
        self.data_list = sorted(glob.glob(data_dir + '*'))
        self.args = {}
        self.y_range = []
        self.y_num = []
        self.atlasTissue = np.load("Atlasfiles/anatomy/npzstuffData_0000.npz")['data'][:, :, :, 1]
        with open(os.path.join(data_path, 'tumor_mparam/args.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                arg, arg_value = line[:-1].split(': ')
                self.args[arg] = arg_value

        self.c_num = int(self.args['num_param'])
        for i in range(self.c_num):
            p_name = self.args['p%d' % i]
            p_min = float(self.args['min_{}'.format(p_name)])
            p_max = float(self.args['max_{}'.format(p_name)])
            p_num = int(self.args['num_{}'.format(p_name)])
            self.y_range.append([p_min, p_max])
            self.y_num.append(p_num)

    def __len__(self):
        return len(self.data_list)

    def crop(self, x, center_x, center_y, center_z):
        center_x = int(round(center_x * 128))
        center_y = int(round(center_y * 128))
        center_z = int(round(center_z * 128))
        return x[center_x - 32:center_x + 32,
               center_y - 32:center_y + 32,
               center_z - 32:center_z + 32]

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        x = data['x'][:, :, :, 1:]
        parameters = data['y']
        output = data['x'][:, :, :, 0:1]

        x = self.crop(x, parameters[3], parameters[4], parameters[5])
        output = self.crop(output, parameters[3], parameters[4], parameters[5])
        for i, ri in enumerate(self.y_range):
            parameters[i] = (parameters[i] - ri[0]) / (ri[1] - ri[0]) * 2 - 1

        x = torch.tensor(x).permute((3, 0, 1, 2)).float()
        parameters = torch.tensor(parameters).float()
        output = torch.tensor(output).permute((3, 0, 1, 2)).float()

        return x, torch.round(parameters[:3] * 10 ** 2) / 10 ** 2, output
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
        
        #TODO WRONG check params should be: torch.tensor([x, y, z, muD, muRho]) not torch.tensor([D, rho, T])
        return  allAtlasImgs,torch.tensor([x, y, z, muD, muRho]), tumorImg

if __name__ == '__main__':
    '''# path = '/mnt/Drive2/ivan/data/tumor_mparam/v/'
    data_dir = '/mnt/Drive2/ivan/data'
    dataset = 'tumor_mparam/v/' #or valid
    dataset = TumorDataset(data_dir, dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, (x, y, z) in enumerate(loader):
        if i == 100:
            break
    '''
    import matplotlib.pyplot as plt
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
# %%
