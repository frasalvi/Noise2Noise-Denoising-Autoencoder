# -*- coding: utf-8 -*-

import torch
from torch import nn

if torch.cuda.is_available():
  device = torch.device('cuda')
  print('Using GPU. ✅')
else:
  device = torch.device('cpu')
  print('Using CPU ❌ 😭')

class TNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, batchnorm=False):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.norm1 = torch.nn.BatchNorm2d(16) if batchnorm else lambda x: x 
    self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
    self.norm2 = torch.nn.BatchNorm2d(32) if batchnorm else lambda x: x 
    self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)
    self.norm3 = torch.nn.BatchNorm2d(32) if batchnorm else lambda x: x 

    self.up4 = nn.Upsample(scale_factor=4)
    self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
    self.norm4 = torch.nn.BatchNorm2d(64) if batchnorm else lambda x: x 
    self.up5 = nn.Upsample(scale_factor=4)
    self.conv5 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
    self.norm5 = torch.nn.BatchNorm2d(64) if batchnorm else lambda x: x 
    self.up6 = nn.Upsample(scale_factor=2)
    self.conv6 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
    self.norm6 = torch.nn.BatchNorm2d(3) if batchnorm else lambda x: x 
    self.conv7 = nn.Conv2d(6, out_channels, kernel_size=3, stride=1, padding=1)

    self.ReLU = lambda tens: nn.functional.leaky_relu(tens, negative_slope=0.1)

  def forward(self, x):
    # Encoder
    x1 = self.norm1(self.ReLU(self.conv1(self.ReLU(x))))
    x2 = self.norm2(self.ReLU(self.conv2(self.pool1(self.ReLU(x1)))))
    x3 = self.norm3(self.ReLU(self.conv3(self.pool2(self.ReLU(x2)))))
    latent_space_embedding = self.ReLU(self.pool3(self.ReLU(x3)))

    # Decoder
    x4 = self.norm4(torch.cat((self.up4(latent_space_embedding), x3), dim=1))
    x5 = self.norm5(torch.cat((self.up5(self.ReLU(self.conv4(self.ReLU(x4)))), x2), dim=1))
    x6 = torch.cat((self.up6(self.ReLU(self.conv5(self.ReLU(x5)))), x1), dim=1)
    x7 = self.norm6(self.ReLU(self.conv6(x6)))
    x8 = self.ReLU(self.conv7(torch.cat((x7,x),dim=1)))
    return x8

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, act_type='relu'):
        super(UNet, self).__init__()
        assert act_type in ('relu', 'lrelu')
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self._block2_1 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block2_4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True) if act_type == 'relu' else nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        pool1 = self._block1(x)
        pool2 = self._block2_1(pool1)
        pool3 = self._block2_2(pool2)
        pool4 = self._block2_3(pool3)
        pool5 = self._block2_4(pool4)

        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        out = self._block6(concat1)
        return out

        
class BaseNet(nn.Module):

  def __init__(self, in_channels=3, out_channels=3):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
    self.conv4 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
    self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
    self.conv6 = nn.Conv2d(32, 8, kernel_size=4, stride=1)

    self.upconv1 = nn.ConvTranspose2d(8, 32, kernel_size=4, stride=1)
    self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2)
    self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)
    self.upconv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
    self.upconv5 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
    self.upconv6 = nn.ConvTranspose2d(32, out_channels, kernel_size=5, stride=1)
    
    self.ReLU = nn.ReLU()

  def forward(self, x):
    # Encoder
    x1 = self.ReLU(self.conv1(x))
    x2 = self.ReLU(self.conv2(x1))
    x3 = self.ReLU(self.conv3(x2))
    x4 = self.ReLU(self.conv4(x3))
    x5 = self.ReLU(self.conv5(x4))
    x6 = self.ReLU(self.conv6(x5))
    # Decoder
    x7 = self.ReLU(self.upconv1(x6))
    x8 = self.ReLU(self.upconv2(x7))
    x9 = self.ReLU(self.upconv3(x8))
    x10 = self.ReLU(self.upconv4(x9))
    x11 = self.ReLU(self.upconv5(x10))
    out = self.ReLU(self.upconv6(x11))
    return out

class Model():
  def __init__(self):
    ## instantiate model + optimizer + loss function + any other stuff you need
    self.model = TNet()
    self.lr = 0.0005 # optimal lr
    self.batch_size = 32
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = torch.nn.MSELoss()
    self.losses = []
    
  def load_pretrained_model(self):
    ## This loads the parameters saved in bestmodel.pth into the model
    checkpoint = torch.load('./bestmodel.pth', map_location=device)
    self.model.load_state_dict(checkpoint['model'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.lr = checkpoint['lr']
    self.batch_size = checkpoint['batch_size']
    self.losses = checkpoint['losses']

  def train(self, train_input, train_target, num_epochs):
    #:train_input: tensor of size (N, C, H, W) containing a noisy version of the images
    #:train_target: tensor of size (N, C, H, W) containing another noisy version of the
    self.model.train()
    self.losses = [0 for x in range(num_epochs)]
    for e in range(num_epochs):
        for b in range(0, train_input.size(0), self.batch_size):
            # forward pass
            output = self.model(train_input[b:b+self.batch_size] / 255)
            loss = self.criterion(output, train_target[b:b+self.batch_size] / 255) 
            self.losses[e] += loss.item() / self.batch_size
            # make step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print('Epoch %d, loss = %.5f'%(e, self.losses[e]))

  def predict(self, test_input):
    #:test input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
    #:returns a tensor of the size (N1, C, H, W)
    # Predict model output in batches
    self.model.eval()
    # output = torch.zeros(test_input.shape)
    # for b in range(0, test_input.size(0), self.batch_size):
    #   output[b:b+self.batch_size] = self.model(test_input[b:b+self.batch_size] / 255)
    # return 255 * output.to(device)
    output = (self.model(test_input / 255.0) * 255.0).clip(0, 255)
    return output.to(test_input.dtype)

    