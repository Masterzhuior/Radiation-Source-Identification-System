import torch
import torch.nn as nn
import torch.nn.parallel

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu=1, n_extra_layers=0, add_final_conv=True):
        super().__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 128, 1, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))

        self.main = main

        self.layers = nn.Sequential(
            self._block(64, 128, 4, 2, 1, 4),   # 
            self._block(128, 256, 4, 2, 1, 4),   # 
            self._block(256, 512, 4, 2, 1, 4),   # 
            self._block(512, 512, 4, 2, 1, 4))   # 1 512 1


    def _block(self, in_channel, out_channel, kernel_size, stride, padding, m_size=32):
        return nn.Sequential(
            nn.Conv1d(
                in_channel, out_channel, kernel_size, stride, padding, bias=False,# since BN is used, no need to bias
            ),
            nn.InstanceNorm1d(out_channel, affine=True),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=m_size, stride=m_size),
        )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = self.main(input)
            output = self.layers(input)

        return output


class Decoder(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        # input is Z, going into a convolution

        self.main = nn.Sequential(
            # input: 64, 200, 1

            self._block(1, 32, 64, 4, 30),  # 64, 1, 32
            self._block(32, 64, 64, 4, 30),  # 64, 32, 128
            #self._block(64, 128, 64, 2, 30),  # 64, 128, 512
            #self._block(, 128, 32, 2, 15),  # 64, 128, 2048
            nn.ConvTranspose1d(
                64, 1, kernel_size=15, stride=1, padding=7
            ),  # 64, 1, 8192
            nn.Tanh(),  # [-1, 1]
        )

    def _block(self, in_channel, out_channel, kernel_size, stride, padding):
        return nn.Sequential(
                nn.ConvTranspose1d(in_channel, out_channel, kernel_size, stride, padding,
                    bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(),
                )

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input = input.permute(0, 2, 1) # 1, 1, 256 mark
            output = self.main(input)
        return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        self.features = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.classifier = nn.Sequential(
                nn.Linear(512, 1),
                nn.Sigmoid(),
                )

    def forward(self, x):
        features = self.features(x)
        classifier = self.classifier(features.reshape(-1, 512))
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, isize, nz, nc, ngf, ngpu, extralayers):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(isize, nz, nc, ngf, ngpu, extralayers)
        self.decoder = Decoder(isize, nz, nc, ngf, ngpu, extralayers)
        self.encoder2 = Encoder(isize, nz, nc, ngf, ngpu, extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x) # 1 256 1
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o

class CNN(nn.Module):
    def __init__(self, in_dim, num_classes=3):
        super(CNN, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = 15744
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=256),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=64),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(64),
            nn.Flatten()
            )
        self.fc = nn.Sequential(
                nn.Linear(self.embed_dim, 2048),
                nn.BatchNorm1d(2048),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(2048, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, num_classes),
                )

    def forward(self, x):
        x = x.view(-1, 1, self.in_dim)
        out = self.conv1(x)
        out = self.fc(out)
        return out
