import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model.retinasim_healthy.glm import GLM

# loads training data from batch 0 to batch num_of_batches-1
def get_train(folder, cell_type, cell_index, num_of_batches):
    
    train_data = []
    stim = np.load("{}/stimulus.npy".format(folder))
    for batch_num in range(num_of_batches):
#         batch = stim[:, :, batch_num*250:(batch_num+1)*250]
        batch = stim[:, :, batch_num*500:(batch_num+1)*500]
        # center and scale the data
#         batch = (np.transpose(np.array([batch[:, :, i:i+5] for i in range(0, 245, 1)]), (0, 3, 1, 2))-0.5)*10
        batch = (np.transpose(np.array([batch[:, :, i:i+5] for i in range(0, 495, 1)]), (0, 3, 1, 2))-0.5)*10
        batch = torch.tensor(batch, dtype=torch.float32)
        train_data.append(batch)
        
    train_target = []
    for batch_num in range(num_of_batches):
        batch_target = torch.tensor(np.load("{}/{}_binned_{}_{}.npy".format(folder, batch_num, cell_type, cell_index))[5:], dtype=torch.float32)
        batch_target = batch_target.view(-1, 1)
        train_target.append(batch_target)
        
    return train_data, train_target

# loads validation data from batch start to batch start+num_of_batches-1
def get_val(folder, cell_type, cell_index, start, num_of_batches):
    
    val_data = []
    stim = np.load("{}/stimulus.npy".format(folder))
    for batch_num in range(start, start+num_of_batches):
#         batch = stim[:, :, batch_num*250:(batch_num+1)*250]
        batch = stim[:, :, batch_num*500:(batch_num+1)*500]
        # center and scale the data
#         batch = (np.transpose(np.array([batch[:, :, i:i+5] for i in range(0, 245, 1)]), (0, 3, 1, 2))-0.5)*10
        batch = (np.transpose(np.array([batch[:, :, i:i+5] for i in range(0, 495, 1)]), (0, 3, 1, 2))-0.5)*10
        batch = torch.tensor(batch, dtype=torch.float32)
        val_data.append(batch)
        
    val_target = []
    for batch_num in range(start, start+num_of_batches):
        batch_target = torch.tensor(np.load("{}/{}_binned_{}_{}.npy".format(folder, batch_num, cell_type, cell_index))[5:], dtype=torch.float32)
        batch_target = batch_target.view(-1, 1)
        val_target.append(batch_target)
        
    return val_data, val_target

def train_glm(folder, cell_type, cell_index, save_folder):
    
    train_batches = 90
    val_batches = 5
    
    train_data, train_target = get_train(folder, cell_type, cell_index, train_batches)
    val_data, val_target = get_val(folder, cell_type, cell_index, train_batches, val_batches)

#     for linear_filter_mode, activation_mode, loss_type in [("combined", "relu", "mse"), ("separate", "shi", "poisson")]:
    for linear_filter_mode, activation_mode, loss_type in [("combined", "relu", "mse")]:

        reg_type = "laplacian"

        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        # need to change the training data if changing num_time_step
        net = GLM(linear_filter_mode=linear_filter_mode, activation_mode=activation_mode, spatial_res=48, num_time_step=5)
        net.cuda(device=1)

        if loss_type == "poisson":
            criterion = nn.PoissonNLLLoss()
        elif loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            raise ValueError("loss type invalid")

        if reg_type == "none":
            r = 0
        elif reg_type == "laplacian":
            r = 1
        else:
            raise ValueError("regularization invalid")

        optimizer = optim.Adam(net.parameters(), lr=0.000001)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_losses_over_time = []
        val_losses_over_time = []

#         for epoch in range(30000):
        for epoch in range(3000):

            # train
            train_loss = 0
            for batch_num in range(train_batches):
                optimizer.zero_grad()
                batch_input = train_data[batch_num].cuda(device=1)
                batch_target = train_target[batch_num].cuda(device=1)
                output = net(batch_input)
                loss = criterion(output, batch_target) + r*net.laplacian_reg()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= train_batches
            train_losses_over_time.append(train_loss)

            # validation
            with torch.no_grad():
                val_loss = 0
                for batch_num in range(val_batches):
                    batch_input = val_data[batch_num].cuda(device=1)
                    batch_target = val_target[batch_num].cuda(device=1)
                    output = net(batch_input)
                    loss = criterion(output, batch_target) + r*net.laplacian_reg()
                    val_loss += loss.item()
                val_loss /= val_batches
                val_losses_over_time.append(val_loss)

            if epoch % 100 == 0:
                print("epoch {}: train loss {}, val loss {}".format(epoch, train_loss, val_loss))
                np.save("{}/train_loss.npy".format(save_folder), np.array(train_losses_over_time))
                np.save("{}/val_loss.npy".format(save_folder), np.array(val_losses_over_time))
                torch.save(net.state_dict(), "{}/model_weights.pth".format(save_folder))
                
            if epoch % 2000 == 0:
                scheduler.step()
