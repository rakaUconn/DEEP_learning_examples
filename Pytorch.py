"""
Created on Mon Nov 20 00:00:25 2023

@author: raj18011
"""

import torch
import torch.nn as nn
from torch.optim import SGD
current=torch.tensor([1.0,2.0,3.0,6.0,8.0])
voltage=torch.tensor([2.0,3.0,5.9,12.5,16.1])
#
# Initialize resistance (the parameter to be learned)
resistance = torch.randn(1,requires_grad=True)
# define ohm model
class Ohmmodel(nn.Module):
    def __init__(self):
        super(Ohmmodel, self).__init__()
        self.resistance=nn.Parameter((torch.tensor(1.0)))
        
                                     
    def forward(self,I):
        return I*self.resistance


# Create an instance of the Ohm's law model
model = Ohmmodel()
        
# Define loss function (Mean Squared Error)
criterion = nn.MSELoss()


# Optimizer (Stochastic Gradient Descent)
optimizer = SGD(model.parameters(), lr=0.01)


# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    predicted_voltage = model(current)
    
    # Calculate loss
    loss = criterion(predicted_voltage, voltage)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the learned resistance value
learned_resistance = model.resistance.item()
print(f'Learned Resistance: {learned_resistance:.4f}')
