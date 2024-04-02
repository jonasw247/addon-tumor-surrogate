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
import numpy as np
import torch.optim as optim

import utils



# %%
model = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
#os.environ['CUDA_VISIBLE_DEVICES'] = "6"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
# chose from epoch 17 
model.load_state_dict(torch.load('/mnt/8tb_slot8/jonas/workingDirDatasets/tumor-surrogate-model-states/daily-cherry-69/modelsWeights/epoch17.pth'))

# %% load test data:
start = 20005 
number_of_samples = 1
stop = start + number_of_samples
dataset = MyDataset(start=start, stop=stop)
plotforStep = 10

data_loader = torch.utils.data.DataLoader(dataset, batch_size=number_of_samples,
                                            num_workers=16, pin_memory=True, shuffle=False)

nBatch = len(data_loader)
myOptim = torch.optim.Adam
optimizationSteps = 100


configForWandb = { 'nBatch': nBatch, 'number_of_samples': number_of_samples, 'start': start, 'stop': stop, 'optimizationSteps': optimizationSteps, "myOptim": myOptim.__name__}
wandb.init(project="optimizeInput", config=configForWandb)
wandb.watch(model)


# iter over data look at data...
model.to(device)
model.eval() 

#model.zero_grad()
losses = []
for i0, (input_tissue, parameters_ground_truth, output_ground_truth) in enumerate(data_loader):
    
    parameters = parameters_ground_truth.clone()
    parameters[:,0] = 0
    parameters[:,1] = 0
    parameters[:,2] = 0
    parameters[:,3] = 1
    parameters[:,4] = 5
    input_tissue = input_tissue.to(device)
    parameters = parameters.to(device)
    parameters.requires_grad = True

    optimizer = myOptim([parameters], lr=0.01)

    for optimStep in range(optimizationSteps):
        optimizer.zero_grad()
        prediction = model(input_tissue, parameters)
        mask = input_tissue[:, 0].unsqueeze(1)  > 0.001
        prediction_masked = prediction.to(device) * mask.to(device)
        #loss = F.mse_loss(prediction_masked, output_ground_truth.to(device))
        loss = F.l1_loss(prediction_masked, output_ground_truth.to(device))
        grad_parameters, = autograd.grad(loss, parameters, retain_graph=True)
        
        loss.backward()
        optimizer.step()
               
        logDict = {}
        logDict["_loss"] = loss.item()
        logDict["_grad_parameters_mean"] = grad_parameters.mean().item()
        #log each grad_parameter
        parameterDiff = torch.abs(parameters - parameters_ground_truth.to(device))
        labels = ["x", "y", "z", "muD", "muRho"]
        for j in range(5):
            logDict["grad_parameters_" + labels[j]] = grad_parameters[0,j].item()
            logDict["parameters_" + labels[j]] = parameters[0,j].item()
            logDict["parameters_difference_" + labels[j]] = parameterDiff[0,j].item()
            
    
        for threshold in np.linspace(0.1, 0.9, 10):
            dice = utils.compute_dice_score(prediction_masked, output_ground_truth.to(device), threshold=threshold)
            stringTh = "dice_" + str(round(threshold, 2))
            logDict[stringTh] = dice.item()
            #print( "dice", threshold, dice.item())

        wandb.log(logDict)

        if optimStep % plotforStep == 0 or optimStep == 0:
            print("step: ", optimStep, loss.item())
            print(loss.item())
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 2 columns

            # Plot the predicted mask
            ax1 = axes[0]

            img1 = ax1.imshow(input_tissue.cpu().detach().numpy()[0,0, :, :, 32], cmap='gray')
            toPlot = prediction_masked.cpu().detach().numpy()[0,0, :, :, 32]
            img1_1 = ax1.imshow(toPlot, vmin=0, vmax=1, cmap='Blues', alpha=toPlot)
            ax1.set_title(f'Iteration {optimStep} of {nBatch}: Prediction')
            fig.colorbar(img1_1, ax=ax1, fraction=0.046, pad=0.04)

            # Plot the ground truth
            ax2 = axes[1]
            img2 = ax2.imshow(input_tissue.cpu().detach().numpy()[0,0, :, :, 32], cmap='gray')
            toPlot = output_ground_truth.cpu().detach().numpy()[0,0, :, :, 32]
            img2_1 = ax2.imshow(toPlot, vmin=0, vmax=1, cmap='Greens', alpha=toPlot)
            ax2.set_title(f'Iteration {optimStep} of {nBatch}: Ground Truth')
            fig.colorbar(img2_1, ax=ax2, fraction=0.046, pad=0.04)

            #plot the difference
            ax3 = axes[2]
            img3 = ax3.imshow(input_tissue.cpu().detach().numpy()[0,0, :, :, 32], cmap='gray')
            toPlot = np.abs(prediction_masked.cpu().detach().numpy()[0,0, :, :, 32] - output_ground_truth.cpu().detach().numpy()[0,0, :, :, 32])
            img3_1 = ax3.imshow(toPlot, vmin=0, vmax=1, cmap='Reds', alpha=toPlot)
            ax3.set_title(f'Iteration {optimStep} of {nBatch}: Difference')
            fig.colorbar(img3_1, ax=ax3, fraction=0.046, pad=0.04)
            
            # Display the figure
            plt.tight_layout()
            path = "optimizeOutput/" + wandb.run.name + "/" 
            os.makedirs(path, exist_ok=True)
            plt.savefig(path + "step" + str(optimStep) + ".pdf")
        
wandb.finish()

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
