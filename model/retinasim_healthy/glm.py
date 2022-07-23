import torch
import torch.nn as nn
import numpy as np

# expects input of shape [batch_size, temporal_steps, spatial_resolution, spatial_resolution]
class GLM(nn.Module):
    
    def __init__(self, linear_filter_mode, activation_mode, spatial_res, num_time_step, gpu_device=1):
        
        super(GLM, self).__init__()
        self.activation_mode = activation_mode
        self.linear_filter_mode = linear_filter_mode
        self.spatial_res = spatial_res
        self.num_time_step = num_time_step
        self.gpu_device = gpu_device
        
        if self.linear_filter_mode == "separate":
            self.spatial_filter = nn.Linear(self.spatial_res*self.spatial_res, 1)
            self.temporal_filter = nn.Linear(self.num_time_step, 1)
        elif self.linear_filter_mode == "combined":
            self.linear = nn.Linear(self.num_time_step*self.spatial_res*self.spatial_res, 1)
        else:
            raise ValueError("linear filter mode not valid")
        
        if self.activation_mode == "relu":
            self.activation = nn.ReLU()
            if self.linear_filter_mode == "separate":
                nn.init.kaiming_uniform_(self.spatial_filter.weight, nonlinearity='relu')
                nn.init.kaiming_uniform_(self.temporal_filter.weight, nonlinearity='relu')
            elif self.linear_filter_mode == "combined":
                nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif self.activation_mode == "leaky_relu":
            self.activation = nn.LeakyReLU()
            if self.linear_filter_mode == "separate":
                nn.init.kaiming_uniform_(self.spatial_filter.weight, nonlinearity='leaky_relu')
                nn.init.kaiming_uniform_(self.temporal_filter.weight, nonlinearity='leaky_relu')
            elif self.linear_filter_mode == "combined":
                nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='leaky_relu')
        elif self.activation_mode == "shi":
            if self.linear_filter_mode == "separate":
                nn.init.xavier_uniform_(self.spatial_filter.weight)
                nn.init.xavier_uniform_(self.temporal_filter.weight)
            elif self.linear_filter_mode == "combined":
                nn.init.xavier_uniform_(self.linear.weight)
        else:
            raise ValueError("activation mode not valid")
         
    def forward(self, x):
        
        if self.linear_filter_mode == "separate":
            x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
            x = self.spatial_filter(x)
            x = torch.squeeze(x)
            x = self.temporal_filter(x)
        elif self.linear_filter_mode == "combined":
            x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
            x = self.linear(x)
            
            
        if self.activation_mode == "shi":
            x = torch.log(1+torch.exp(x))
        else:
            x = self.activation(x)

        return x
    
    def laplacian_reg(self, space_only=True):
        if self.linear_filter_mode == "separate":
            laplacian_filter_2d = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), dtype=torch.float32), dim=0), dim=0)
            laplacian_filter_2d = laplacian_filter_2d.cuda(device=self.gpu_device)
            reg_space = torch.norm(nn.functional.conv2d(torch.unsqueeze(torch.reshape(self.spatial_filter.weight, (1, self.spatial_res, self.spatial_res)), dim=0), laplacian_filter_2d), p=2)
            if space_only:
                return reg_space
            laplacian_filter_1d = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array([1,-2,1]), dtype=torch.float32), dim=0), dim=0)
            laplacian_filter_1d = laplacian_filter_1d.cuda(device=self.gpu_device)
            reg_time = torch.norm(nn.functional.conv1d(torch.unsqueeze(self.temporal_filter.weight, dim=0), laplacian_filter_1d), p=2)
            return reg_space + reg_time
        elif self.linear_filter_mode == "combined":
            if space_only:
                reg = 0
                weight = torch.reshape(self.linear.weight, (1, self.num_time_step, self.spatial_res, self.spatial_res))
                laplacian_filter_2d = torch.unsqueeze(torch.unsqueeze(torch.tensor(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]), dtype=torch.float32), dim=0), dim=0)
                laplacian_filter_2d = laplacian_filter_2d.cuda(device=self.gpu_device)
                for i in range(self.num_time_step):
                    reg += torch.norm(nn.functional.conv2d(weight[:, i:i+1, :, :], laplacian_filter_2d), p=2)
                return reg
            else:
                weight = torch.reshape(self.linear.weight, (1, 1, self.num_time_step, self.spatial_res, self.spatial_res))
                laplacian_filter_3d = np.array([[[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[1,-6,1],[0,1,0]], [[0,0,0],[0,1,0],[0,0,0]]])
                laplacian_filter_3d = torch.unsqueeze(torch.unsqueeze(torch.tensor(laplacian_filter_3d, dtype=torch.float32), dim=0), dim=0)
                laplacian_filter_3d = laplacian_filter_3d.cuda(device=self.gpu_device)
                reg = torch.norm(nn.functional.conv3d(weight, laplacian_filter_3d), p=2)
                return reg