#%%
import math

import torch
import wandb


from config import get_config
from data import TumorDataset, MyDataset
from model import TumorSurrogate
from utils import AverageMeter, loss_function, compute_dice_score, mean_absolute_error, compute_dice_score_in_mask
import os
torch.manual_seed(42)

def visualize(output, input, ground_truth, step, path):

    import matplotlib.pyplot as plt
    print('output', output.shape)   
    print('input', input.shape)
    print('ground_truth', ground_truth.shape)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    img0 = axs[0].imshow(output[0, 0,32].detach().cpu().numpy())
    axs[0].set_title("Output")
    fig.colorbar(img0, ax=axs[0], orientation='vertical')

    img1 = axs[1].imshow(input[0, 0,32].detach().cpu().numpy())
    axs[1].set_title("Input")
    axs[1].set_title("Input")
    fig.colorbar(img1, ax=axs[1], orientation='vertical')
    img2 = axs[2].imshow(ground_truth[0, 0,32].detach().cpu().numpy())
    axs[2].set_title("Ground Truth")
    fig.colorbar(img2, ax=axs[2], orientation='vertical')
    # draw color bar for each img
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    path= os.path.join(path, f"output_{step}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)    
    plt.savefig(path)
    plt.show()
    plt.close()

class Trainer():

    def __init__(self, config, device):
        self.config = config
        self.device = device# "cpu"# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#device
        self.save_path = config.save_path
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.global_step = 0

        self.startTrain = 0
        self.stopTrain = 1
        self.startVal = self.stopTrain+1
        self.stopVal = 10
        self.name  = "init_noNameYet"

    """ learning rate """

    def _calc_learning_rate(self, epoch, batch=0, nBatch=None):
        T_total = self.config.max_epoch * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * (self.config.lr_max - self.config.lr_min) * (1 + math.cos(math.pi * T_cur / T_total)) + self.config.lr_min
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self):
        train_dataset = MyDataset(start=0, stop=2)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.train_batch_size,
                                                  num_workers=16, pin_memory=True, shuffle=False)

        nBatch = len(data_loader)

        net = TumorSurrogate(widths=[128, 128, 128, 128], n_cells=[5, 5, 5, 4], strides=[2, 2, 2, 1])
        net = net.to(device=self.device)
        optimizer = torch.optim.Adam(
            net.parameters(), self.config.lr_min, weight_decay=self.config.weight_decay, betas=(self.config.beta1, self.config.beta2)
        )

        #visualize Init 
        for i, (input, parameters, ground_truth) in enumerate(data_loader):
            input, parameters, ground_truth = input.to(self.device), parameters.to(self.device), ground_truth.to(self.device)
            output = net(input, parameters)
            visualize(output, input, ground_truth, -1, self.save_path + "/" + self.name + "/outputImgs/")
            break
        
        wandb.init(project="tumorSimOrigIVanilla")
        # log important settings
        wandb.config.update(self.config)
        wandb.config.update({"nBatch": nBatch})
        wandb.config.update({"device": self.device})
        wandb.config.update({"model": net.__class__.__name__})

        self.name = wandb.run.name
        for epoch in range(self.config.max_epoch):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            save_frequency = 1
            validation_frequency = 1
            losses = AverageMeter()
            mae = AverageMeter()
            dice_score = AverageMeter()
            dice_score_masked = AverageMeter()

            # switch to train mode
            net.train()

            for i, (input, parameters, ground_truth) in enumerate(data_loader):
                # lr
                self.adjust_learning_rate(optimizer, epoch, batch=i, nBatch=nBatch)
                # train weight parameters if not fix_net_weights
                input, parameters, ground_truth = input.to(self.device), parameters.to(self.device), ground_truth.to(self.device)

                output = net(input, parameters)  # forward (DataParallel)
                # loss
                loss = loss_function(u_sim=ground_truth, u_pred=output, mask=input[:, 0].unsqueeze(1)  > 0.001)
                # measure accuracy and record loss
                mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                if mae_wm_value is not None and mae_gm_value is not None and mae_csf_value is not None:
                    mae_mean_value = (mae_wm_value.item() + mae_gm_value.item() + mae_csf_value.item()) / 3
                    mae.update(mae_mean_value, input.size(0))
                    wandb.log({"Mae/train": mae_mean_value}, step=self.global_step)

                dice = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                diceInsideBrain = compute_dice_score_in_mask(u_pred=output, u_sim=ground_truth, threshold=0.4, mask=input[:, 0] > 0.001)
                losses.update(loss, input.size(0))
                if dice is not None:
                    dice_score.update(dice, input.size(0))
                    wandb.log({"Loss/train": loss.item()}, step=self.global_step)
                    if dice is not None:
                        wandb.log({"Dice/train": dice.item()}, step=self.global_step)

                if diceInsideBrain is not None:
                    dice_score_masked.update(diceInsideBrain, input.size(0))
                    wandb.log({"DiceMasked/train": diceInsideBrain.item()}, step=self.global_step)
                self.global_step += 1

                # compute gradient and do SGD step
                net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                optimizer.step()  # update weight parameters

                if i % save_frequency == 0:
                    # save model
                    savepath = self.save_path + self.name + "/modelsWeights"
                    os.makedirs(savepath, exist_ok=True)
                    torch.save(net.state_dict(), savepath+"/epoch"+ str(epoch) + '.pth')
            visualize(output, input, ground_truth, epoch, self.save_path + "/" + self.name + "/outputImgs/")
            # validate
            if (epoch + 1) % validation_frequency == 0:
                val_loss, val_mae, val_dice = self.validate(net=net)

                # tensorboard logging
                wandb.log({"Loss/val": val_loss}, step=self.global_step)
                wandb.log({"Mae/val": val_mae}, step=self.global_step)
                wandb.log({"Dice/val": val_dice}, step=self.global_step)

    def validate(self, net,  step=0):
        valid_dataset = MyDataset(start=self.startVal, stop= self.stopVal)

        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.config.val_batch_size, num_workers=16, pin_memory=True,
                                                        shuffle=False)# shuffle TODO

        net.eval()

        losses = AverageMeter()
        mae = AverageMeter()
        dice_score_avg = AverageMeter()
        dice_score_02 = []
        dice_score_04 = []
        dice_score_08 = []

        mae_wm = []
        mae_gm = []
        mae_csf = []

        with torch.no_grad():
            ground_truths = []
            outputs = []
            for i, (input_batch, parameters, ground_truth_batch) in enumerate(valid_data_loader):
                input_batch, parameters, ground_truth_batch = input_batch.to(self.device), parameters.to(self.device), ground_truth_batch.to(
                    self.device)

                # compute output
                output_batch = net(input_batch, parameters)

                loss = loss_function(u_sim=ground_truth_batch, u_pred=output_batch, mask=input_batch[:, 0].unsqueeze(1)  > 0.001)
                losses.update(loss, input_batch.size(0))

                for output, ground_truth, input in zip(output_batch, ground_truth_batch, input_batch):
                    output, ground_truth, input = output[None, :], ground_truth[None, :], input[None, :]  # get batch dim back
                    if i == 0:
                        outputs.append(output)
                        ground_truths.append(ground_truth)
                    # measure mae, dice score and record loss
                    mae_wm_value, mae_gm_value, mae_csf_value = mean_absolute_error(ground_truth=ground_truth, output=output, input=input)
                    if mae_wm_value is None:
                        print("MAE was None, skipping this sample in validation")
                        continue

                    mae_wm.append(mae_wm_value)
                    mae_gm.append(mae_gm_value)
                    mae_csf.append(mae_csf_value)

                    mae_mean_value = (mae_wm_value.item() + mae_gm_value.item() + mae_csf_value.item()) / 3
                    dice_02 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.2)
                    dice_04 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.4)
                    dice_08 = compute_dice_score(u_pred=output, u_sim=ground_truth, threshold=0.8)
                    dice_02_insideBrain = compute_dice_score_in_mask(u_pred=output, u_sim=ground_truth, threshold=0.2, mask=input[:, 0] > 0.001)
                    dice_04_insideBrain = compute_dice_score_in_mask(u_pred=output, u_sim=ground_truth, threshold=0.4, mask=input[:, 0] > 0.001)
                    dice_08_insideBrain = compute_dice_score_in_mask(u_pred=output, u_sim=ground_truth, threshold=0.8, mask=input[:, 0] > 0.001)


                    mae.update(mae_mean_value, input.size(0))
                    if dice_04 is not None:
                        dice_score_avg.update(dice_04, input.size(0))
                        dice_score_04.append(dice_04.cpu().item())
                    if dice_02 is not None:
                        dice_score_02.append(dice_02.cpu().item())
                    if dice_08 is not None:
                        dice_score_08.append(dice_08.cpu().item())
                    if dice_02_insideBrain is not None:
                        dice_score_02.append(dice_02_insideBrain.cpu().item())
                    if dice_04_insideBrain is not None:
                        dice_score_04.append(dice_04_insideBrain.cpu().item())
                    if dice_08_insideBrain is not None:
                        dice_score_08.append(dice_08_insideBrain.cpu().item())

            return losses.avg, mae.avg, dice_score_avg.avg


if __name__ == '__main__':
    config, unparsed = get_config()
    trainer = Trainer(config, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    trainer.train()

# %%
