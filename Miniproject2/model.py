# From torch: All these modules are either specified in the project file, or confirmed with TA's
from turtle import forward
from torch import ones, zeros, empty, cat, arange, load, float, set_grad_enabled
from torch.nn.functional import fold, unfold
from functools import reduce
from math import floor 

set_grad_enabled(False)

flatten = lambda deep_list: [item for sublist in deep_list for item in sublist]

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
    def zero_grad(self):
        for param in self.param():
            param[0].grad = zeros(param[0].shape)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
        self.print_stuff = False

    def forward(self, *input):
        x = input[0]
        count = 0
        for layer in self.layers:
            self.print_stuff and print("layer nr " + str(count))
            self.print_stuff and print('Before layer: ',x.shape)
            x = layer.forward(x)
            self.print_stuff and print('After layer: ',x.shape)
            count += 1
        return x

    def backward(self, *gradwrtoutput):
        x = gradwrtoutput[0]
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x

    def param(self):
        parameter_list = [layer.param() for layer in self.layers]
        return flatten(parameter_list)


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
        self.bias = ones(out_features) * 0.1
        self.weight.grad = zeros(self.weight.shape)
        self.bias.grad = zeros(self.bias.shape)

        uniform_initialization(self.weight, kind='pytorch')

    def forward (self, *input):
        self.input = input[0]
        # If single element instead of batch
        if(self.input.dim() != 2):
            self.input = self.input[None, :]
        output = self.input @ self.weight.T + self.bias
        return output

    def backward(self, *gradwrtoutput):
        grad_output = gradwrtoutput[0]
        self.weight.grad += grad_output.T @ self.input
        self.bias.grad += grad_output.sum(axis=0)
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
            kernel_size = (kernel_size, kernel_size)
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
        self.bias = ones(out_channels) * 0.1
        self.weight.grad = zeros(self.weight.shape)
        self.bias.grad = zeros(self.bias.shape)
        uniform_initialization(self.weight, kind='pytorch')
        # a = 1/((in_channels*kernel_size[0]**2)**0.5)
        # self.bias.uniform_(-a, a)

    def forward(self, *input):
        # Get shapes

        self.input = input[0]
        assert self.input.dim() == 3 or self.input.dim() == 4
        # If single element instead of batch
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
        self.weight.grad += (gradwrtoutput_unfolded @ input_unfolded.transpose(1,2)).sum(axis=0).view(self.weight.shape)
        self.bias.grad += gradwrtoutput_unfolded.sum(axis=(0,2)).view(self.bias.shape)
        gradwrtinput_unfolded = (kernel.transpose(0,1) @ gradwrtoutput_unfolded)
        return fold(gradwrtinput_unfolded, output_size=self.input.shape[2:4], kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

    def param(self):
        return [(self.weight, self.weight.grad),
                (self.bias, self.bias.grad)]


class NearestUpsampling(Module):
    def  __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, *input):
        # The function repeat_interleave(num_repeats, dim) repeats elements in a matrix
        # num_repeats times along the specified dimension.
        self.input = input[0]
        # If single element instead of batch
        if(self.input.dim() != 4):
            self.input = self.input[None, :]
        out = input[0].repeat_interleave(repeats=self.scale_factor, dim=3).repeat_interleave(repeats=self.scale_factor, dim=2)
        return out

    def backward(self, *gradwrtoutput):
        # The gradient of a NNUpsampling layer is the sum of the upsampled elements in gradwrtoutput.
        # Example: if scale_factor=2, the gradient is the sum of 2x2 areas in gradwrtoutput.
        # First unfold the matrix to easily take the sum
        unfolded = unfold(gradwrtoutput[0], kernel_size=self.scale_factor, stride=self.scale_factor)
        summed = unfolded.view(self.input.shape[0], self.input.shape[1], self.scale_factor**2, unfolded.shape[-1]).sum(dim=2)
        return summed.view(self.input.shape)

    def param(self):
        return []


class Upsampling(Sequential):
    def  __init__(self, in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, scale_factor=2):
        super()

        upsampling = NearestUpsampling(scale_factor=scale_factor)
        conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)
        self.layers = [upsampling, conv]
        self.print_stuff = False


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

    def param(self):
        return []


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

    def param(self):
        return []


class MSE(Module):
    def __init__(self):
        super()
        self.last_input_diff = None

    # input, target
    def forward(self, *input):
        assert len(input) == 2
        assert len(input[0]) == len(input[1])
        self.last_input_diff = (input[1] - input[0])
        return (self.last_input_diff**2).mean()

    def backward(self, *gradwrtoutput):
        if not gradwrtoutput:
            gradwrtoutput = [ones(1)]
        preliminary_loss = -2 * self.last_input_diff / reduce((lambda x, y: x * y), 
                                                              self.last_input_diff.shape)
        return gradwrtoutput[0] * preliminary_loss

    def param(self):
        return []


class Optimizer():
    def __init__(self, params):
        self.params
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        for param in self.params():
            param[0].grad = zeros(param[0].shape) 


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0):
        self.params = params
        self.lr = lr

    def step(self):
        for (param, _) in self.params:
            param -= self.lr * param.grad


class Model():
    def __init__(self):
        in_channels = 3
        out_channels = 3

        self.model = Sequential(
                        Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1),
                        ReLU(),
                        Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                        ReLU(),
                        Upsampling(scale_factor=2, in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                        ReLU(),
                        Upsampling(scale_factor=2, in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                        Sigmoid()
                        )
        self.criterion = MSE()
        self.optimizer = SGD(self.model.param(), lr=1e-2)

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel.pth into the model
        self.model = load('bestmodel.pth')

    def train(self, train_input, train_target, num_epochs, **kwargs):
        # train ̇input: tensor of size (N, C, H, W) containing a noisy version of the images.
        # train target: tensor of size (N, C, H, W) containing another noisy version of the
        # same images, which only differs from the input by their noise.
        batch_size = 32
        self.losses = []
        avg_loss = 0
        
        for e in range(num_epochs):
            print('Doing epoch %d'%e)
            for b, (input, target) in enumerate(zip(train_input.split(batch_size),
                                                    train_target.split(batch_size))):
                output = self.model(input)
                loss = self.criterion(output, target)
                avg_loss += loss.item()

                # make step
                self.model.zero_grad()
                gradient = self.criterion.backward()
                gradient = self.model.backward(gradient)
                self.optimizer.step()
                
                # debug
                if(kwargs['debug']):
                    b_freq = 5
                    if b % b_freq == 0 and (b+e) > 0:
                        self.losses.append(avg_loss / b_freq)
                        avg_loss = 0
                        b % 50 == 0 and print(self.losses[-1])

    def predict(self, test_input):
        #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained
        # or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)
        return self.model(test_input)