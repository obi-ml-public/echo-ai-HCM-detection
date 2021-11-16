import torch
import lz4.frame
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader

# loads data and generates pytorch dataloaders for 'client.py' file
def load_data(path, batch_size):
    EPS=0.00000000001
    
    trainData=np.load(path + 'trainData.npy')
    trainScale=np.load(path + 'trainScale.npy')
    trainLabel=np.load(path + 'trainLabel.npy')

    validData=np.load(path + 'validData.npy')
    validScale=np.load(path + 'validScale.npy')
    validLabel=np.load(path + 'validLabel.npy')

    trainData = torch.Tensor(trainData)
    validData = torch.Tensor(validData)
    
    trainScale = torch.Tensor(trainScale).unsqueeze(-1)
    validScale = torch.Tensor(validScale).unsqueeze(-1)
    
    trainLabel = torch.Tensor(trainLabel).unsqueeze(-1)
    validLabel = torch.Tensor(validLabel).unsqueeze(-1)
    
    trainData = trainData.squeeze().unsqueeze(1)
    validData = validData.squeeze().unsqueeze(1)
    
    print("Train:", trainData.shape, trainScale.shape, trainLabel.shape)
    print("Validation:", validData.shape, validScale.shape, validLabel.shape)
    
    train_dataset = TensorDataset(trainData, trainScale, 1/(trainScale+EPS), trainLabel)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    valid_dataset = TensorDataset(validData, validScale, 1/(validScale+EPS), validLabel)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_dataloader, valid_dataloader
