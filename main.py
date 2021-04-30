from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import random
import torch
import torchvision
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import os

np.random.seed(2021)

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3,3), stride=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output
    
    def forward_feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)

        x = F.relu(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        y = self.fc3(x)

        output = F.log_softmax(y, dim=1)
        return x, output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

def closest_feature_vectors(model, device, test_loader):
    model.eval()
    test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
    test_predictions = []
    test_targets = []
    test_embeddings = torch.zeros((0, 64), dtype=torch.float32)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embeddings, output = model.forward_feature(data)
            preds = torch.argmax(output, dim=1)
            test_predictions.extend(preds.tolist())
            test_targets.extend(target.tolist())
            test_embeddings = torch.cat((test_embeddings, embeddings), 0)
            test_imgs = torch.cat((test_imgs, data), 0)
        test_imgs = np.array(test_imgs)
        test_embeddings = np.array(test_embeddings)
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)
    
    chosen_imgs = test_imgs[:4]

    # Create plt plot:
    fig, axes = plt.subplots(4, 9)

    for idx, img in enumerate(chosen_imgs):
        fv_orig = test_embeddings[idx]
        dists = []
        axes[idx, 0].imshow(img[0], cmap='gray')
        for idx2, img2 in enumerate(test_imgs):
            dists.append((idx2, np.linalg.norm(fv_orig - test_embeddings[idx2])))
        
        dists_sorted = sorted(dists, key=lambda x: x[1])
        for i in range(8):
            get_idx = dists_sorted[i][0]
            axes[idx, i + 1].imshow(test_imgs[get_idx][0], cmap='gray')
    
    plt.show()

def feature_vectors(model, device, test_loader):
    model.eval()
    test_imgs = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
    test_predictions = []
    test_targets = []
    test_embeddings = torch.zeros((0, 64), dtype=torch.float32)
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embeddings, output = model.forward_feature(data)
            preds = torch.argmax(output, dim=1)
            test_predictions.extend(preds.tolist())
            test_targets.extend(target.tolist())
            test_embeddings = torch.cat((test_embeddings, embeddings), 0)
            test_imgs = torch.cat((test_imgs, data), 0)
        test_imgs = np.array(test_imgs)
        test_embeddings = np.array(test_embeddings)
        test_targets = np.array(test_targets)
        test_predictions = np.array(test_predictions)

    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(test_embeddings)
    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 10
    for lab in range(num_categories):
        indices = test_predictions==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

def conf_matrix(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    y_true = []
    y_pred = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for tens in pred: y_pred.append(tens.item())
            for tens in target: y_true.append(tens.item())
    
    print(confusion_matrix(y_true, y_pred))

def visualize_kernels(model):
    # Visualize conv filter
    model.eval()
    kernels = model.conv1.weight.detach()
    fig, axarr = plt.subplots(3, 3)
    for idx in range(kernels.size(0)):
        axarr[idx % 3, idx // 3].imshow(kernels[idx].squeeze(), cmap = 'gray')
    fig.suptitle("First Layer Learned Kernels")
    plt.show()

def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))
    
    return test_loss

def test_examples(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    wrong = 0
    images = []
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for idx, tens in enumerate(pred):
                if tens.item() != target[idx].item() and wrong < 9:
                    images.append(data[idx])
                    wrong += 1
            
            if wrong >= 9: break
    
    fig, axs = plt.subplots(3, 3)
    for i in range(9):
        axs[i % 3, i // 3].imshow(images[i][0], cmap='gray')
    plt.show()

def load_indices(train_ds, subset=None):
    class_counts = defaultdict(list)
    train_inds = []
    val_inds = []

    for idx, (data, label) in enumerate(train_ds):
        class_counts[label].append(idx)
    
    for class_, inds in class_counts.items():
        total = len(inds)
        vals = int(0.15 * total)
        random.shuffle(inds)
        train_inds.extend(inds[vals:])
        val_inds.extend(inds[:vals])

    return train_inds, val_inds

def plot_subsets():
    num_trains = [51005, 25505, 12755, 6379, 3193]
    train_loss = [0.0546, 0.0630, 0.0657, 0.0944, 0.1205]
    test_loss = [0.0643, 0.0862, 0.0746, 0.1588, 0.1540]

    plt.loglog(num_trains, train_loss, 'o-', label = "train loss")
    plt.loglog(num_trains, test_loss, 'o-', label = "test loss")
    plt.legend()
    plt.xlabel('Num Training Examples (log)')
    plt.ylabel('Loss (log)')
    plt.title('Loss With Fewer Training Examples')
    plt.show()

def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        # model = fcNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)
        # test_examples(model, device, test_loader)
        # visualize_kernels(model)
        # conf_matrix(model, device, test_loader)
        # feature_vectors(model, device, test_loader)
        # closest_feature_vectors(model, device, test_loader)

        return
        

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    # transforms.RandomCrop(24),
                    # transforms.Resize(28),
                    # transforms.GaussianBlur(11, sigma=(0.1, 2.0)),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # if running on a subset:
    
    # subset_half = random.sample(list(range(len(train_dataset))), int(0.5 * len(train_dataset)))
    # subset_quarter = random.sample(list(range(len(train_dataset))), int(0.25 * len(train_dataset)))
    # subset_eighth = random.sample(list(range(len(train_dataset))), int((1/8) * len(train_dataset)))
    # subset_sixteenth = random.sample(list(range(len(train_dataset))), int((1/16) * len(train_dataset)))
    
    # new_train = []
    # for idx, data in enumerate(train_dataset):
    #     if idx in subset_sixteenth:
    #         new_train.append(data)
    
    # train_dataset = new_train

    subset_indices_train, subset_indices_valid = load_indices(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, 
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    # model = ConvNet().to(device)
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    epochs = []
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        train_loss = test(model, device, train_loader)
        val_loss = test(model, device, val_loader)
        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here
    
    # plot_subsets()

    print("Accuracy On Train Set")
    test(model, device, train_loader)
    print("Accuracy On Val Set")
    test(model, device, val_loader)

    plt.plot(epochs, train_losses, 'o-', label = "train loss")
    plt.plot(epochs, val_losses, 'o-', label = "val loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.show()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")

if __name__ == '__main__':
    main()
