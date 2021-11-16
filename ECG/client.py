from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from models import *
from utils import *

import copy
import os

import warnings 
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    """Create model, load data, define Flower client, start Flower client."""
    net = Convolutional2d_new(initial_filters=16)
    net = torch.nn.DataParallel(net)
    net.to(DEVICE)
    
    # Path where data in .npy format resides.
    path = '/mnt/obi0/djsolanki/projects/ECG/data_new/'
    trainloader, validloader, testloader = load_data(path=path, batch_size=64)

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
            
            if rnd == 0:
                loss, accuracy, auc = test(net, testloader, rnd)
            else:
                loss, accuracy, auc = test(net, validloader, rnd)
            
            return float(loss), len(validloader.dataset), {"accuracy": float(accuracy), "auc score:":float(auc)}

    # Start client and connect it to the IP where server was initiated.
    fl.client.start_numpy_client("10.9.9.6:7100", client=FedClient())
    
# Loss criterion when mixup is used
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    lam = np.random.uniform()
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    y_a, y_b = y, y[index]
    return mixed_x, mixed_y, y_a, y_b, lam

# Add gaussian noise to given tensor
def add_noise(tensor, device, mean=0, std=0.1):
    return tensor + torch.randn(tensor.size()).to(device) * std + mean

# Replace values of random pixels with 1
def salt(tensor, device, ratio=20):
    put = np.random.uniform(low=tensor.min(), high=tensor.max(), size=(1,))
    tensor = tensor.cpu().numpy()
    noise = np.random.randint(ratio, size=tensor.shape)
    tensor = np.where(noise == 0, put, tensor)
    return torch.from_numpy(tensor).to(device)

# Replace values of random pixels with 0
def pepper(tensor, device, ratio=20):
    tensor = tensor.cpu().numpy()
    noise = np.random.randint(ratio, size=tensor.shape)
    tensor = np.where(noise == 0, 0, tensor)
    return torch.from_numpy(tensor).to(device)

# Add/Subtract given value from tensor(Shifting)
def add(tensor, value):
    temp = np.random.uniform()
    if temp < 0.5:
        return tensor + value
    else:
        return tensor - value
# Add sin curve to given tensor
def add_sin(tensor):
    SHAPE=(2500,12,1)
    sinnois=(np.sin(np.array(range(SHAPE[0]))*0.01)*100).astype('float32')
    sinnois12=np.zeros([SHAPE[0],SHAPE[1]],dtype='float32')
    for i in range(SHAPE[1]):
        sinnois12[:,i]=sinnois
    sinnois12 = torch.from_numpy(sinnois12) / 1000
    sinnois12 = sinnois12.squeeze().unsqueeze(0).to(DEVICE)
    
    temp = np.random.uniform()
    if temp < 0.5:
        return tensor + sinnois12
    else:
        return tensor - sinnois12

# Apply one of the selected augmentations to given tensor    
def augment(tensor, device):
    do = np.random.uniform()
    if do < 0.1:
        return tensor
    else:
        temp = np.random.uniform()
        if temp < 0.4:
            tensor = add_noise(tensor, device)
        elif temp < 0.6:
            tensor = pepper(tensor, device)
        elif temp < 0.8:
            tensor = add(tensor, 1.)
        else:
            tensor = add_sin(tensor)

        return tensor

    
def fedprox_loss(output, target, model, glob_model, criterion, rnd, mu=1):
    if rnd <= 1:
        mu=0.0
    elif rnd <=2:
        mu=0.1
    elif rnd <=4:
        mu=0.5
    else:
        mu=1
    
    local_loss = criterion(output, target)
    
    for (x,y) in zip(model.parameters(), glob_model.parameters()):
        prox = torch.sum(torch.pow(x - y, 2))
        
    return local_loss + (mu / 2) * prox

def train(net, trainloader, epochs, lr, rnd):
    """Train the network on the training set."""
    glob_net = copy.deepcopy(net)
    
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()

    for e in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
#             images, labels, y_a, y_b, lam = mixup_data(images, labels)
#             images, labels, y_a, y_b, lam = images.to(DEVICE), labels.to(DEVICE), y_a.to(DEVICE), y_b.to(DEVICE), lam
            
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images = augment(images, DEVICE)
            
            outputs = net(images)
            loss = criterion(outputs, labels)
#             loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
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
        for images, labels in validloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
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
