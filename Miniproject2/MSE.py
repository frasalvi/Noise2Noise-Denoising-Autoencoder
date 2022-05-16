from module import Module
from torch.nn import MSELoss as OrigMSE
import torch

class MSE(Module):
    def __init__(self):
        super()
        self.last_input_lenght = None
        self.last_input_diff = None
        self.last_input = None


    def forward(self, *input):
        assert len(input) == 2
        assert len(input[0]) == len(input[1])
        self.last_input_lenght = len(input[0])
        self.last_input_diff = (input[0] - input[1])
        self.last_input = input[0]
        return sum(self.last_input_diff**2)/self.last_input_lenght

    def backward(self, *gradwrtoutput):
        preliminary_loss =  ( - 2/self.last_input_lenght)*self.last_input_diff*self.last_input
        if len(gradwrtoutput) != 1:
            return preliminary_loss
        return gradwrtoutput[0]*preliminary_loss

    def param(self):
        return []
