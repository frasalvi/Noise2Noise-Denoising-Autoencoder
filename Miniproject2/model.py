# From torch: All these modules are either specified in the project file, or confirmed with TA's
from torch import ones, empty, cat, arange, load, float, set_grad_enabled
from torch.nn.functional import fold, unfold
from functools import reduce

set_grad_enabled(False)

def floor(x):
    return int(x//1)

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
    def zero_grad():
        for param in self.param():
            param[0].grad = 0


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
    def  __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1):
        # Implements 2D convolution.
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
        self.padding = padding
        self.dilation = dilation
        self.stride = stride

        # Weight initialization. Replace this with sth. more sophisticated later
        self.weight = empty(out_channels,in_channels,kernel_size[0],kernel_size[1])
        self.bias = empty(out_channels)
        uniform_initialization(self.weight, kind='pytorch')

    def forward(self, *input):
        # Get shapes
        self.input = input[0]
        assert self.input.dim() == 3 or self.input.dim() == 4
        if(self.input.dim() != 4):
            self.input = self.input[None, :]
        self.batch_size = self.input.shape[0]
        
        # Output shape (in 1D) = floor((H + 2P - D*(K-1) - 1)/S + 1)
        outH = floor((self.input.shape[2] + 2*self.padding - self.dilation * (self.kernel_size[0] - 1) -1 ) / self.stride + 1)
        outW = floor((self.input.shape[3] + 2*self.padding - self.dilation * (self.kernel_size[1] - 1) -1 ) / self.stride + 1)
        self.output_shape = (self.batch_size, self.out_channels, outH, outW)

        # Do convolution as matrix-matrix product
        unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        convolved = self.weight.view(self.out_channels, -1).unsqueeze(0) @ unfolded + self.bias.view(1, -1, 1).unsqueeze(0)
        return convolved.view(self.output_shape)

    def backward(self, *gradwrtoutput):
        # Get the unfolded versions of the input, gradient w.r.t. output and kernel
        input_unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        gradwrtoutput_unfolded = gradwrtoutput[0].view(self.batch_size, self.out_channels, self.output_shape[2]*self.output_shape[3])
        kernel = self.weight.view(self.out_channels, -1)

        # Calculate weight and bias updates
        self.weight.grad = (gradwrtoutput_unfolded @ input_unfolded.transpose(1,2)).sum(axis=0).view(self.weight.shape)
        self.bias.grad = gradwrtoutput_unfolded.sum(axis=(0,2)).view(self.bias.shape)
        gradwrtinput_unfolded = (kernel.transpose(0,1) @ gradwrtoutput_unfolded)
        return fold(gradwrtinput_unfolded, output_size=self.input.shape[2:4], kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

    
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.is_input_bigger_than_zero = None

    def forward(self, *input):
        real_input = input[0]
        self.is_input_bigger_than_zero = (real_input > 0).type(float)
        return real_input*self.is_input_bigger_than_zero

    def backward(self, *gradwrtoutput):
        grad_output = gradwrtoutput[0]
        return self.is_input_bigger_than_zero * grad_output


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, *input):
        real_input = input[0]
        t = real_input.clip(-100, 100)
        e_to_input = (-t).exp()
        activations = 1 / (1 + e_to_input)
        self.activations = activations
        return activations

    def backward(self, *gradwrtoutput):
        grad_output = gradwrtoutput[0]
        layer_grad = self.activations * (1 - self.activations)
        return layer_grad * grad_output


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, *input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, *gradwrtoutput):
        x = gradwrtoutput
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x


class NNupsampling(Module):
    def  __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, *input):
        # The function repeat_interleave(num_repeats, dim) repeats elements in a matrix
        # num_repeats times along the specified dimension.
        self.input = input[0]
        out = input[0].repeat_interleave(repeats=self.scale_factor, dim=3).repeat_interleave(repeats=self.scale_factor, dim=2)
        return out

    def backward(self, *gradwrtoutput):
        # The gradient of a NNupsampling layer is the sum of the upsampled elements in gradwrtoutput.
        # Example: if scale_factor=2, the gradient is the sum of 2x2 areas in gradwrtoutput.
        # First unfold the matrix to easily take the sum
        unfolded = unfold(gradwrtoutput[0], kernel_size=self.scale_factor, stride=self.scale_factor)
        summed = unfolded.view(self.input.shape[0], self.input.shape[1], self.scale_factor**2, unfolded.shape[-1]).sum(dim=2)
        return summed.view(self.input.shape)

class MSE(Module):
    def __init__(self):
        super()
        self.batch_size = None
        self.last_input_diff = None


    def forward(self, *input):
        assert len(input) == 2
        assert len(input[0]) == len(input[1])
        self.batch_size = input[0].shape[0]
        self.last_input_diff = (input[1] - input[0])
        return (self.last_input_diff**2).mean()

    def backward(self, *gradwrtoutput):
        preliminary_loss =  ( - 2/self.batch_size)*self.last_input_diff/self.last_input_diff.shape[1]
        if len(gradwrtoutput) != 1:
            return preliminary_loss
        return gradwrtoutput[0]*preliminary_loss

    def param(self):
        return []
