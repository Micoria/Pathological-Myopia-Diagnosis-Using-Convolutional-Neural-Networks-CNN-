import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import SqueezeNet
import torchsummary
from dataloader import LoadData
import copy

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

model = SqueezeNet(num_classes=2).to(device)
print(model)
print(torchsummary.summary(model, (3, 224, 224), 1))


# loading training set and validating set
train_data = LoadData("/Users/micoria/Desktop/SqueezeNet/train.txt", True)
train_dl = torch.utils.data.DataLoader(train_data, batch_size=16, pin_memory=True,
                                           shuffle=True, num_workers=0)
test_data = LoadData("/Users/micoria/Desktop/SqueezeNet/valid.txt", True)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=16, pin_memory=True,
                                           shuffle=True, num_workers=0)


# write training matrix
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  
    num_batches = len(dataloader)  
    print('num_batches:', num_batches)
    train_loss, train_acc = 0, 0  # initializing the training loss and accuracy

    for X, y in dataloader:  # load images and labels
        X, y = X.to(device), y.to(device)
        # compute standard eerror
        pred = model(X)  
        loss = loss_fn(pred, y)  #compute loss

        # backpropagation
        optimizer.zero_grad()  
        loss.backward()  # backpropagation
        optimizer.step()  # refresh the model

        # record accuracy and loss
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss

# functions for validation set
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  
    num_batches = len(dataloader)  
    test_loss, test_acc = 0, 0

    # when training stops, stop computing to save memory
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)

            # compute loss
            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_loss += loss.item()
            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()

    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss


# start training

epochs = 20

train_loss = []
train_acc = []
test_loss = []
test_acc = []

best_acc = 0  # initialize accuracy


loss_function = nn.CrossEntropyLoss()  # define the loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # use Adam optimizer

for epoch in range(epochs):

    model.train()
    epoch_train_acc, epoch_train_loss = train(train_dl, model, loss_function, optimizer)

    model.eval()
    epoch_test_acc, epoch_test_loss = test(test_dl, model, loss_function)

    # save best_model
    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        best_model = copy.deepcopy(model)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    # get study rate
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}, Lr:{:.2E}')
    print(template.format(epoch + 1, epoch_train_acc * 100, epoch_train_loss,
                          epoch_test_acc * 100, epoch_test_loss, lr))

# save best model in file
PATH = './best_model.pth'  
torch.save(best_model.state_dict(), PATH)

print('Done')

import matplotlib.pyplot as plt
#hide warnings
import warnings
warnings.filterwarnings("ignore")               
plt.rcParams['font.sans-serif']    = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False      
plt.rcParams['figure.dpi']         = 100        

epochs_range = range(epochs)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()