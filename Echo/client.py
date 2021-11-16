from collections import OrderedDict

import os

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from sklearn.metrics import roc_auc_score

from adabelief_pytorch import AdaBelief

import random

from models import *
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    """Create model, load data, define Flower client, start Flower client."""
    net = Model(filters=32)
    net = torch.nn.DataParallel(net)
    net.to(DEVICE)
    
    # Path where data in .npy format resides.
    path = '/'
    trainloader, validloader = load_data(path=path, batch_size=20)

    # Flower client
    class FedClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            state_dict = OrderedDict(
                    {
                        k: torch.Tensor(np.atleast_1d(v))
                        for k, v in zip(net.state_dict().keys(), parameters)
                    })
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            lr = float(config['learning_rate'])
            self.set_parameters(parameters)
            train(net, trainloader, epochs=2, lr=lr, rnd=int(config['round']))
            return self.get_parameters(), len(trainloader.dataset), {}

        def evaluate(self, parameters, config):   
            rnd=int(config['round'])
            self.set_parameters(parameters)
            loss, accuracy, auc = test(net, validloader, rnd)
            
            return float(loss), len(validloader.dataset), {"accuracy": float(accuracy), "auc score:":float(auc)}

    # Start client and connect it to the IP where server was initiated.
    fl.client.start_numpy_client("10.9.9.5:7100", client=FedClient())


# Loss criterion when mixup is used
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def mixup_data(x, x_sc, x_inv, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    lam = np.random.uniform()
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_x_sc = lam * x_sc + (1 - lam) * x_sc[index, :]
    mixed_x_inv = lam * x_inv + (1 - lam) * x_inv[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    y_a, y_b = y, y[index]
    return mixed_x, mixed_x_sc, mixed_x_inv, mixed_y, y_a, y_b, lam

# Add gaussian noise to given tensor
def add_noise(tensor, device, mean=0, std=0.2):
    return tensor + torch.randn(tensor.size()).to(device) * std + mean

# Replace values of random pixels with 1
def salt(tensor, device, ratio=10):
    tensor = tensor.cpu().numpy()
    noise = np.random.randint(ratio, size=tensor.shape)
    tensor = np.where(noise == 0, 1, tensor)
    return torch.from_numpy(tensor).to(device)

# Replace values of random pixels with 0
def pepper(tensor, device, ratio=10):
    tensor = tensor.cpu().numpy()
    noise = np.random.randint(ratio, size=tensor.shape)
    tensor = np.where(noise == 0, 0, tensor)
    return torch.from_numpy(tensor).to(device)

# Augments tensor by changing brightness
def brightness(tensor, device):
    factor = np.random.uniform(1.1, 2.1)
    return tensor * factor

# Rotates the video frames by one of the defined angles
def rotate(tensor, device):
    angles = [-30, -15, -10, 0, 10, 15, 30]
    angle = random.choice(angles)
    for i in range(tensor.shape[0]):
        tensor[i] = TF.rotate(tensor[i], angle)
        
    return tensor.to(device)

# Apply one of the selected augmentations to given tensor   
def augment(tensor, device):
    do = np.random.uniform()
    if do < 0.2:
        return tensor
    else:
        temp = np.random.uniform()
        if temp < 0.3:
            tensor = add_noise(tensor, device)
        elif temp < 0.5:
            tensor = salt(tensor, device)
        elif temp < 0.7:
            tensor = pepper(tensor, device)
        elif temp < 0.85:
            tensor = brightness(tensor, device)
        elif temp < 1.:
            tensor = rotate(tensor, device)

        return tensor


    
def train(net, trainloader, epochs, lr, rnd):
    """Train the network on the training set."""
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    if rnd > 20:
        alpha = 0 
    elif rnd > 15:
        alpha = 0.2
    elif rnd > 10:
        alpha = 0.1
    elif rnd > 5:
        alpha = 0.2
    else:
        alpha = 0.1
    
    for e in range(epochs):
        running_loss = 0.0
        for images, scale, invscale, labels in trainloader:
            images, scale, invscale, labels = images.to(DEVICE), scale.to(DEVICE), invscale.to(DEVICE), labels.to(DEVICE)
            images = augment(images, DEVICE)
            
            outputs = net(images, scale, invscale)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Epoch:", str(e+1), " Loss:", running_loss / len(trainloader))

def test(net, validloader, rnd):
    """Validate the network on the entire test set."""
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()

    preds, ground_truths = [], []
    with torch.no_grad():
        for images, scale, invscale, labels in validloader:
            images, scale, invscale, labels = images.to(DEVICE), scale.to(DEVICE), invscale.to(DEVICE), labels.to(DEVICE)
            outputs = net(images, scale, invscale)
            
            loss += criterion(outputs, labels).item()

            predicted = outputs > 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            outputs = outputs.cpu().squeeze().numpy()
            labels = labels.cpu().squeeze().numpy()
            
            preds.append(outputs)
            ground_truths.append(labels)
        preds = np.concatenate(preds)
        ground_truths = np.concatenate(ground_truths)
            
    auc_score = roc_auc_score(ground_truths, preds)
    accuracy = correct / total
    print("Round:", rnd, "Accuracy:", accuracy, " AUC:", auc_score)
    return loss, accuracy, auc_score


if __name__ == "__main__":
    main()
