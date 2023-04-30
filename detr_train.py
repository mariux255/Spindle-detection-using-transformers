class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



from cProfile import label
import torch
from m1 import Net
from m1_dataloader import sentdex_data
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter



from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import sleep


# Loading data, setting up GPU use, setting up variables for model training
batch_s = 64

training_dataset = sentdex_data()
trainloader = DataLoader(training_dataset, batch_size=batch_s, shuffle=True, num_workers=0)

validation_dataset = sentdex_data()
valloader = DataLoader(validation_dataset, batch_size=batch_s, shuffle=True, num_workers=0)

print("Device used:")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


food, label = next(iter(trainloader))
#print(food.size())
#print(label)



EPOCHS = 10

net = Net()
net = net.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

correct = 0
total = 0

training_accuracy = []
training_loss = []
validation_accuracy = []
validation_loss = []


#=======================================================================#
# Running model training and validation loops. Model is fed with data, which
# split into 4 different frequencies. Accuracy and loss are displayed in the terminal output

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    net.train()
    
    with tqdm(trainloader, unit="batch") as tepoch:
        running_acc = []
        running_loss = []
        labels_temp = []
        for i, data in enumerate(tepoch):
            tepoch.set_description(f"{bcolors.WARNING} T Epoch {bcolors.ENDC} {epoch}")
            
            # Loading the 4 different frequency bands and their respective label
            # Loading everything to device for GPU training
            food, labels = data
            #print(labels)
            food = food.to(device)
            labels = labels.to(device)


            # Model training procedures
            optimizer.zero_grad()
            outputs = net(food)
            #predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            #print(labels.type())
            #print(outputs.type())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculating metrics: loss and accuracy
            running_loss.append(loss.item())

            #correct = (predictions == labels).sum().item()
            #accuracy = correct / batch_s
            #running_acc.append(accuracy)
            
            # tqdm progress bar update
            if (i % 10 == 9) or (i == 0):
                tepoch.set_postfix(loss=sum(running_loss)/len(running_loss), accuracy=0)
            sleep(0.1)

    # Saving accuracy and loss values    
    #training_accuracy.append(100. * (sum(running_acc)/len(running_acc)))
    training_loss.append(loss.item())
        

    net.eval()
    with tqdm(valloader, unit="batch") as tepoch:
        running_acc = []
        running_loss = []
        for i, data in enumerate(tepoch):
        # get the inputs; data is a list of [inputs, labels]
            tepoch.set_description(f"{bcolors.HEADER} V Epoch {bcolors.ENDC} {epoch}")
            # Loading the 4 different frequency bands and their respective label
            # Loading everything to device for GPU training
            food, labels = data
            food = food.to(device)
            labels = labels.to(device)

            # Model training procedures
            optimizer.zero_grad()
            outputs = net(food)
           # predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculating metrics: loss and accuracy
            running_loss.append(loss.item())
            #correct = (predictions == labels).sum().item()
            #accuracy = correct / batch_s
           # running_acc.append(accuracy)
            
            # tqdm progress bar update
            if (i % 10 == 9) or (i == 0):
                tepoch.set_postfix(loss=sum(running_loss)/len(running_loss), accuracy=0)
            sleep(0.1)
    
    # Saving accuracy and loss values 
    #validation_accuracy.append(100. * (sum(running_acc)/len(running_acc)))
    validation_loss.append(loss.item())
print(outputs)
print(labels)


torch.save(net, '/home/marius/Documents/OneDrive/MSc/StartUP/Code/m1_stats_features.pt')
print('Finished Training')