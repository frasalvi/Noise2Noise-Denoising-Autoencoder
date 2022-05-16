from module import Module
from torch.nn import MSELoss as OrigMSE


x = OrigMSE()
x()

class MSE(Module):
    def __init__(self):
        super()
        self.last_input_lenght = None
        self.last_input_diff = None


    def forward(self, *input):
        assert len(input) == 2
        assert len(input[0]) == len(input[1])
        self.last_input_lenght = len(input[0])
        self.last_input_diff = (input[0] - input[1])
        return sum(self.last_input_diff**2)/self.last_input_lenght

    def backward(self, *gradwrtoutput):
        return ( - 2/self.last_input_lenght)*self.last_input_diff

    def param(self):
        return []
