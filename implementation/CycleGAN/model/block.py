import sys
sys.path.append('..')
import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    A building block for a Residual Network with skip connections.
    """

    def __init__(self, dimension, dropout=False, bias=True, kernel_size=3):
        super().__init__()
        self.conv_block = self._build_conv_block(dimension, dropout, bias, kernel_size)

    def _build_conv_block(self, dimension, dropout, bias, kernel_size):
        ones_padding, padding_layer_class = False,nn.ReflectionPad2d
        pad_value = (1 if ones_padding else 0)
        norm_layer_class = nn.InstanceNorm2d

        layers = []
        if padding_layer_class is not None:
            layers.append(padding_layer_class(kernel_size // 2))

        layers += [
            nn.Conv2d(dimension, dimension, kernel_size, padding=pad_value, bias=bias),
            norm_layer_class(dimension),
            nn.ReLU(True)
        ]

        if dropout:
            layers.append(nn.Dropout(0.5))

        if padding_layer_class is not None:
            layers.append(padding_layer_class(kernel_size // 2))

        layers += [
            nn.Conv2d(dimension, dimension, kernel_size, padding=pad_value, bias=bias),
            norm_layer_class(dimension)
        ]

        return nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    A Residual Network-based Generator with multiple ResNetBlocks.
    """

    def __init__(self, in_channels, out_channels, num_filters=64, num_blocks=9, num_sampling=2, use_dropout=False):
        super().__init__()
        self.model = self._build_model(in_channels, out_channels, num_filters, num_blocks, num_sampling,use_dropout)

    def _build_model(self, in_channels, out_channels, num_filters, num_blocks, num_sampling,use_dropout):
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, num_filters, kernel_size=7, bias=True),
                  nn.InstanceNorm2d(num_filters), nn.ReLU(True)]

        in_channels = num_filters

        for _ in range(num_sampling):
            num_filters *= 2
            layers += [nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=2, padding=1,
                                 bias=True), nn.InstanceNorm2d(num_filters),
                       nn.ReLU(True)]
            in_channels = num_filters

        for _ in range(num_blocks):
            layers.append(ResNetBlock(num_filters, use_dropout, True))

        for _ in range(num_sampling):
            layers += [
                nn.ConvTranspose2d(num_filters, num_filters // 2, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=True), nn.InstanceNorm2d(num_filters // 2),
                nn.ReLU(True)]
            num_filters //= 2

        layers += [nn.ReflectionPad2d(3), nn.Conv2d(num_filters, out_channels, kernel_size=7), nn.Tanh()]

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    A PatchGAN discriminator that outputs a classification for each patch of the image.
    """

    def __init__(self, num_channels, num_filters=64, num_conv_layers=3, ker_size=4, padding=1):
        super().__init__()
        self.model = self._build_model(num_channels, num_filters, num_conv_layers, ker_size, padding)

    def _build_model(self, num_channels, num_filters, num_conv_layers, ker_size, padding):
        layers = [nn.Conv2d(num_channels, num_filters, kernel_size=ker_size, stride=2, padding=padding),
                  nn.LeakyReLU(0.2, True)]

        for i in range(1, num_conv_layers+1):
            num_filters *= 2
            layers += [nn.Conv2d(num_filters // 2, num_filters, kernel_size=ker_size, stride= 2 if i < num_conv_layers else 1, padding=padding,
                                 bias=True), nn.InstanceNorm2d(num_filters),
                       nn.LeakyReLU(0.2, True)]

        layers.append(nn.Conv2d(num_filters, 1, kernel_size=ker_size, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

'''
testNet = ResNetBlock(64, dropout=False, bias=True)
print(testNet)

testD = Discriminator(3)
print(testD)

testG = Generator(3, 3)
print(testG)
'''