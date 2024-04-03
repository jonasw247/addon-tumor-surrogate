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
# %% load test data:
# chose from epoch 17, 25000 was also good with dice. from daily-cherry-69
model.load_state_dict(torch.load('/mnt/8tb_slot8/jonas/workingDirDatasets/tumor-surrogate-model-states/daily-cherry-69/modelsWeights/epoch17.pth'))

start = 20010
number_of_samples = 1
stop = start + number_of_samples
dataset = MyDataset(start=start, stop=stop)
plotforStep = 10
fixOrigin = False

data_loader = torch.utils.data.DataLoader(dataset, batch_size=number_of_samples,
                                            num_workers=16, pin_memory=True, shuffle=False)

nBatch = len(data_loader)
myOptim = torch.optim.Adam
optimizationSteps = 100
learningRate = 0.01#0.01


configForWandb = { 'nBatch': nBatch, 'number_of_samples': number_of_samples, 'start': start, 'stop': stop, 'optimizationSteps': optimizationSteps, "myOptim": myOptim.__name__, "learningRate": learningRate, "model": model.__class__.__name__, "fixOrigin": fixOrigin, "plotforStep": plotforStep}
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

    optimizer = myOptim([parameters], lr=learningRate)

    for optimStep in range(optimizationSteps):
        optimizer.zero_grad()
        prediction = model(input_tissue, parameters)
        mask = input_tissue[:, 0].unsqueeze(1)  > 0.001
        prediction_masked = prediction.to(device) * mask.to(device)
        loss = F.mse_loss(prediction_masked, output_ground_truth.to(device))
        #loss = F.l2_loss(prediction_masked, output_ground_truth.to(device))
        grad_parameters, = autograd.grad(loss, parameters, retain_graph=True)
        
        loss.backward()

        # Manually zero out gradients for fixed entries
        # dont allow to change the origin, but fix it at COM
        if fixOrigin:
            with torch.no_grad():
                parameters.grad[:,0] = 0
                parameters.grad[:,1] = 0
                parameters.grad[:,2] = 0
        optimizer.step()

               
        logDict = {}
        logDict["_loss"] = loss.item()
        logDict["_grad_parameters_mean"] = grad_parameters.mean().item()
        #log each grad_parameter
        parameterDiff = parameters - parameters_ground_truth.to(device)
        labels = ["x", "y", "z", "muD", "muRho"]
        for j in range(5):
            logDict["grad_parameters_" + labels[j]] = grad_parameters[0,j].item()
            logDict["parameters_" + labels[j]] = parameters[0,j].item()
            logDict["parameters_difference_" + labels[j]] = parameterDiff[0,j].item()
        #log mean parameter difference
        logDict["_parameters_difference_mean"] = parameterDiff.abs().mean().item()
            
    
        for threshold in np.linspace(0.1, 0.9, 9):
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

# Printing only the weights
for name, param in model.named_parameters():
    if "weight" in name:  # Check if 'weight' is in parameter name
        print(f"Layer: {name} | Size: {param.size()} | Values : \n{param.data} \n")

# %%
def check_weights(model1, model2):
    models_identical = True
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2 or not torch.equal(param1, param2):
            print(f"Weights differ at layer: {name1}")
            models_identical = False
            break
    if models_identical:
        print("All model weights are identical.")
    else:
        print("Models have different weights.")

model2 = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
model2 = model2.to(device=device)
model2.load_state_dict(torch.load('/mnt/8tb_slot8/jonas/workingDirDatasets/tumor-surrogate-model-states/daily-cherry-69/modelsWeights/epoch17.pth'))
check_weights(model, model2)
# %%
