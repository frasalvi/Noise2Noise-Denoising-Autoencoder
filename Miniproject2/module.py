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
