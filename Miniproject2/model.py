# From torch: All these modules are either specified in the project file, or confirmed with TA's
from torch import ones, empty, cat, arange, load, float, set_grad_enabled
from torch.nn.functional import fold, unfold
from functools import reduce

set_grad_enabled(False)


class Module(object):
    '''
    Abstract class for implementing a generic module.
    '''
    def forward (self, *input):
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self):
        return []
    def __call__(self, *input):
        return self.forward(*input)


def uniform_initialization(tensor, kind='pytorch', gain=1):
    '''
    Performs weight initialization by drawing from a Uniform distribution,
    according to the chosen method.

    Args:
        tensor (torch.tensor): weight tensor.
        gain (float): reccomended gain according to chosen nonlinearity.
        kind (str): initialization method. One of:
            xavier: Var = 2*gain / (in_features + out_features)
            he: Var = gain / in_features
            pytorch: Var = 1 / (3 * in_features)
    Returns:
        tensor (torch.tensor): initialized weight tensor.
    '''
    out_size = tensor.shape[0]
    in_size = tensor.shape[1]

    # Conv layer
    if tensor.dim() > 2:
        kernel_size = reduce(lambda x, y: x*y, tensor.shape[2:])
        out_size *= kernel_size
        in_size *= kernel_size

    if(kind == 'xavier'):
        a = gain * (6 / (in_size + out_size))**0.5
    elif(kind == 'he'):
        a = gain * (3 / in_size)**0.5
    elif(kind == 'pytorch'):
        a = 1 / in_size**0.5
    else:
        raise ValueError('invalid initialization option')

    tensor.uniform_(-a, a)


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = empty((out_features, in_features))
        self.bias = empty(out_features)

        # Awful, maybe let's think of a cleaner solution later
        uniform_initialization(self.weight)

    def forward (self, *input):
        self.input = input[0]
        output = self.input @ self.weight.T + self.bias
        return output

    def backward(self, *gradwrtoutput):
        grad_output = gradwrtoutput[0]
        self.weight.grad = grad_output.T @ self.input
        self.bias.grad = grad_output.sum(axis=0)
        return grad_output @ self.weight

    def param(self):
        return [(self.weight, self.weight.grad),
                (self.bias, self.bias.grad)]


class Conv2d(Module):

    def  __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                device=None, dtype=None):
    # Implements 2D convolution.
    # TO DO: change stride, padding, dilation and padding_mode.
    # Check if kernel_size is correct
        if type(kernel_size)==int:
            kernel_size = (kernel_size,kernel_size)
        elif (type(kernel_size)==tuple and len(kernel_size)==2):
            pass
        else:
            raise ValueError('Invalid dimensions of kernel_size. It should be either an integer or a tuple of length 2.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Weight initialization.
        self.weight = empty(out_channels,in_channels,kernel_size[0],kernel_size[1])
        self.bias = empty(out_channels)
        uniform_initialization(self.weight, kind='pytorch')

    def forward(self, *input):
        self.input = input[0]
        assert self.input.dim() == 3 or self.input.dim() == 4
        if(self.input.dim() != 4):
            self.input = self.input[None, :]
        batch_size = self.input.shape[0]

        unfolded = unfold(self.input, kernel_size=self.kernel_size)
        convolved = self.weight.view(self.out_channels, -1).unsqueeze(0) @ unfolded + self.bias.view(1, -1, 1).unsqueeze(0)
        return convolved.view(batch_size, self.out_channels, self.input.shape[2] - self.kernel_size[0] + 1, self.input.shape[3] - self.kernel_size[1] + 1)

    def backward(self, *gradwrtoutput):
        batch_size = self.input.shape[0]
        output_size = (batch_size, self.out_channels, (self.input.shape[2] - self.kernel_size[0] + 1)*(self.input.shape[3] - self.kernel_size[1] + 1))
        input_unfolded = unfold(self.input, kernel_size=self.kernel_size)
        gradwrtoutput_unfolded = gradwrtoutput[0].view(output_size)
        kernel = self.weight.view(self.out_channels, -1)
        # print('unfolded gradient w.r.t. output shape: ',gradwrtoutput_unfolded.shape)
        # print('unfolded input: ',input_unfolded.shape)
        # print('unfolded kernel shape: ',kernel.shape)

        # print('grad wrt input shape: ',(kernel.transpose(0,1) @ gradwrtoutput_unfolded).shape)

        self.weight.grad = (gradwrtoutput_unfolded @ input_unfolded.transpose(1,2)).sum(axis=0).view(self.weight.shape)
        self.bias.grad = gradwrtoutput_unfolded.sum(axis=(0,2)).view(self.bias.shape)
        gradwrtinput_unfolded = (kernel.transpose(0,1) @ gradwrtoutput_unfolded)
        return fold(gradwrtinput_unfolded, output_size=self.input.shape[2:4], kernel_size=self.kernel_size)