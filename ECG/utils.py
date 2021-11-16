import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Loads train, valid and test data from given path and generates pytorch dataloaders to send back to client.py
def load_data(path, batch_size):
    SHAPE=(2500,12,1)
    trainData=(np.load(path + 'trainData.npy')/1000).astype('float32')
    trainLabel=np.load(path + 'trainLabel.npy').astype('float32')
    dlen=trainData.shape[0]
    trainData=trainData.reshape((dlen,)+SHAPE)

    validData=(np.load(path + 'validData.npy')/1000).astype('float32')
    validLabel=np.load(path + 'validLabel.npy').astype('float32')
    dlen=validData.shape[0]
    validData=validData.reshape((dlen,)+SHAPE)
    
    testData=(np.load(path + 'testData.npy')/1000).astype('float32')
    testLabel=np.load(path + 'testLabel.npy').astype('float32')
    dlen=testData.shape[0]
    testData=testData.reshape((dlen,)+SHAPE)
    
    trainData = torch.Tensor(trainData)
    validData = torch.Tensor(validData)
    testData = torch.Tensor(testData)
    
    trainLabel = torch.Tensor(trainLabel).unsqueeze(-1)
    validLabel = torch.Tensor(validLabel).unsqueeze(-1)
    testLabel = torch.Tensor(testLabel).unsqueeze(-1)
    
    trainData = trainData.squeeze().unsqueeze(1)
    validData = validData.squeeze().unsqueeze(1)
    testData = testData.squeeze().unsqueeze(1)
    
    print("Train:", trainData.shape, trainLabel.shape)
    print("Validation:", validData.shape, validLabel.shape)
    print("Test:", testData.shape, testLabel.shape)
    
    train_dataset = TensorDataset(trainData, trainLabel)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    valid_dataset = TensorDataset(validData, validLabel)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    
    test_dataset = TensorDataset(testData, testLabel)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, valid_dataloader, test_dataloader
