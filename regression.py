# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:37:34 2019

@author: Vik Jakkula
"""

import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

x = Variable(torch.Tensor([[1],[2],[3],[4]]))
y = Variable(torch.Tensor([[2],[4],[6],[8]]))

print(x)

class LinearRegressionModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_size,output_size)
        
    def forward(self,x):
        y_predict = self.linear(x)
        return y_predict
    
model = LinearRegressionModel(1,1)
criteria = nn.MSELoss()
# 0.01 is learning rate
optimizer = optim.SGD(model.parameters(),0.01)

for epoch in range(500):
    y_predict = model(x)
    loss = criteria(y_predict,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch, float(loss.data[0]))


test = Variable(torch.Tensor([20]))
z = model.forward(test)
print(float(z[0]))


