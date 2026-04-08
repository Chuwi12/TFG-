import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class NeuralNetwork(nn.Module):

    def __init__(self):
        '''
        Método constructor donde definimos la aquitectura de la nuerona,
        Consta de una capa de entrada con 512 neuronas.
        Una primera capa oculta con 256 neuronas.
        Una segunda capa oculta con 128 neuronas.
        Una capa de salida con 77 neuronas.
        '''
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 77)
        )
    
    def forward(self, x):
        # x es el vector de 512 dimensiones que viene de tu Dataset
        logits = self.linear_relu_stack(x)
        return logits
    
