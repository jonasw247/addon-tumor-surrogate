#%%
import math

import torch
import wandb


from config import get_config
from data import TumorDataset, MyDataset
from model import TumorSurrogate
import os
import matplotlib.pyplot as plt
torch.manual_seed(42)
import torch.autograd as autograd
import torch.nn.functional as F

import utils


# %%
model = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
#os.environ['CUDA_VISIBLE_DEVICES'] = "6"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
# chose from epoch 17 
model.load_state_dict(torch.load('/mnt/8tb_slot8/jonas/workingDirDatasets/tumor-surrogate-model-states/daily-cherry-69/modelsWeights/epoch17.pth'))


# %% load test data:
start = 20000 
number_of_samples = 3
stop = start + number_of_samples
dataset = MyDataset(start=start, stop=stop)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=number_of_samples,
                                            num_workers=16, pin_memory=True, shuffle=False)

nBatch = len(data_loader)

#%% iter over data look at data...
model.to(device)
model.zero_grad()
losses = []
for i, (input, parameters, ground_truth) in enumerate(data_loader):
    
    parameters[:,-1] = 3
    input = input.to(device)
    parameters = parameters.to(device)
    parameters.requires_grad = True
    input.requires_grad = True
    model.eval() 
    prediction = model(input, parameters)
    
    prediction_masked = prediction.to(device) * mask.to(device)
    loss = F.mse_loss(prediction_masked, ground_truth.to(device))
    grad_input, = autograd.grad(loss, input, retain_graph=True)
    grad_parameters, = autograd.grad(loss, parameters, retain_graph=True)
    mask = input[:, 0].unsqueeze(1)  > 0.001
    dices = []
    print(grad_parameters)

    for i in range(number_of_samples):
        plt.title(f'iteration {i} of {nBatch}')
        plt.imshow(prediction_masked.cpu()[i].detach().numpy()[0, :, :, 32], vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        plt.title(f'iteration {i} of {nBatch} ground truth')
        plt.imshow(ground_truth.cpu()[i].detach().numpy()[0, :, :, 32], vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        dices.append(utils.compute_dice_score(prediction_masked[i].cpu(), ground_truth[i].cpu(), threshold=0.5))
    

# %%
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")


# %
# %%
