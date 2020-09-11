# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:18:35 2019

@author: Vik Jakkula
"""


import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
from torchvision.transforms import transforms

train_dataset = datasets.MNIST(root='./data', 
                               train=True, 
                               download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, 
                              download=True, 
                              transform=transforms.ToTensor())


print(train_dataset)
print(test_dataset)

#make dataset iteratable
batch_size = 100
n_iters = 3000

num_epochs = 5

training = torch.utils.data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
testing = torch.utils.data.DataLoader(dataset=test_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)


#create model class
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
    
        #convolution 1
        self.cnn1 = nn.Conv2d(in_channels = 1,
                          out_channels = 16,
                          kernel_size = 5,
                          stride = 1,
                          padding = 2)
        self.relu1 = nn.ReLU()
    
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
    
        #convolution 2
        self.cnn2 = nn.Conv2d(in_channels = 16,
                          out_channels = 32,
                          kernel_size = 5,
                          stride = 1,
                          padding = 2)
        self.relu2 = nn.ReLU()
    
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
    
        self.fc1 = nn.Linear(32*7*7,10)
    
    def forward(self,x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out


model = CNNModel()
criterion = nn.CrossEntropyLoss()
learning_Rate = 0.01

optimizer = optim.SGD(model.parameters(),lr=learning_Rate)

# Train the model

iter = 0
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(training):
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        output = model(images)
        
        loss = criterion(output,images)
        loss.backward()
        optimizer.step()
        
        iter=iter+1
        #if((iter+1)%100 == 0):
            #print("epoch[%d][%d]")
        
model.eval()
correct = 0
total = 0

for images,labels in testing:
    images = Variable(images)
    outputs = model(images)
    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print("Accuracy",correct)

    


