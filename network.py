import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(3):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=64))
                self.size = 64
            else:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=self.size * 2))
                self.size *= 2

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=8 * 8 * self.size, out_features=1024, bias=False),
                                nn.BatchNorm1d(num_features=1024),
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=1024, out_features=z_size)
        self.l_var = nn.Linear(in_features=1024, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=8 * 8 * size, bias=False),
                                nn.BatchNorm1d(num_features=8 * 8 * size),
                                nn.ReLU(True))
        self.size = size
        layers_list = []

        for i in range(3):
            layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))
            self.size //= 2
        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten, ten_from_normal=None):
        # this depends on what we're doing
        if self.training:
            # in training we forward a batch of sampled fron N(0,1) with a batch of sampled from the original images encoded
            # beacuse we need both of them for the losses
            ten = self.fc(ten)
            ten = ten.view(len(ten), -1, 8, 8)
            ten = self.conv(ten)

            ten_from_normal = self.fc(ten_from_normal)
            ten_from_normal = ten_from_normal.view(len(ten_from_normal), -1, 8, 8)
            ten_from_normal = self.conv(ten_from_normal)

            return ten, ten_from_normal
        else:
            # in test we just need to decode one input
            ten = self.fc(ten)
            ten = ten.view(len(ten), -1, 8, 8)
            ten = self.conv(ten)
            return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
    def __init__(self, channel_in=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)))
        self.size = 32
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=128))
        self.size = 128
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        self.size = 256
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=256))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=8 * 8 * self.size, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),

        )

    def forward(self, ten_original, ten, ten_from_normal):
        # discriminator is used only during training
        # we cat:
        # original->original images
        # ten->reconstructed images
        # normal->images generated from a sampled normal N(0,1)
        ten = torch.cat((ten_original, ten, ten_from_normal), 0)
        for i, lay in enumerate(self.conv):
            # we take the 9th layer as one of the outputs
            if i == 3:
                ten, layer_ten = lay(ten, True)
                # we need the layer representations just for the original and reconstructed,
                # so we strip the part related to the normal
                layer_ten = layer_ten[:-len(ten_from_normal)]
                # flatten, because it's a convolutional shape
                layer_ten = layer_ten.view(len(layer_ten), -1)
            else:
                ten = lay(ten)

        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        return F.sigmoid(ten), layer_ten

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)


class VaeGan(nn.Module):
    def __init__(self):
        super(VaeGan, self).__init__()
        # latent space size
        self.z_size = 128
        self.encoder = Encoder(z_size=self.z_size)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size)
        self.discriminator = Discriminator(channel_in=3)
        # self-defined function to init the parameters
        self.init_parameters()

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    nn.init.xavier_normal(m.weight)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, ten, gen_size=10):
        if self.training:
            # save the original images
            ten_original = ten
            # encode
            mus, log_variances = self.encoder(ten)
            # we need the true variances, not the log one
            variances = torch.exp(log_variances * 0.5)
            # sample from a gaussian
            ten_from_normal = Variable(torch.randn(len(ten), self.z_size).cuda(), requires_grad=True)
            # shift and scale using the means and variances
            ten = ten_from_normal * (variances) + mus
            # decode both the gaussian shifted and the gaussian in (0,1)
            ten, ten_from_normal = self.decoder(ten, ten_from_normal)
            # discriminator
            ten_class, ten_layer = self.discriminator(ten_original, ten, ten_from_normal)

            return ten, ten_class, ten_layer, mus, log_variances

        else:
            # just sample and decode
            ten = Variable(torch.randn(gen_size, self.z_size).cuda(), requires_grad=False)
            ten = self.decoder(ten)
            return ten, None, None

    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(ten_original, ten_predict, layer_original, layer_predicted, labels_original, labels_predicted,
             labels_sampled, mus, variances):
        """

        :param ten_original: original images
        :param ten_predict:  predicted images (output of the decoder)
        :param layer_original:  intermediate layer for original (intermediate output of the discriminator)
        :param layer_predicted: intermediate layer for reconstructed (intermediate output of the discriminator)
        :param labels_original: labels for original (output of the discriminator)
        :param labels_predicted: labels for reconstructed (output of the discriminator)
        :param labels_sampled: labels for sampled from gaussian (0,1) (output of the discriminator)
        :param mus: tensor of means
        :param variances: tensor of diagonals of log_variances
        :return:
        """

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = torch.mean((ten_original.view(len(ten_original), -1) - ten_predict.view(len(ten_predict), -1)) ** 2)
        # kl-divergence
        kl = torch.mean(0.5 * torch.sum(variances.exp() + mus ** 2 - variances - 1, 1))
        # mse between intermediate layers
        mse = torch.mean(torch.sum((layer_original - layer_predicted) ** 2, 1))
        # bce for decoder and discriminator for original,sampled and reconstructed
        # the only excluded is the bce_gen_original
        bce_gen_predicted = nn.BCELoss()(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCELoss()(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCELoss()(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCELoss()(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCELoss()(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        return nle, kl, mse, bce_gen_predicted, bce_gen_sampled, bce_dis_original, bce_dis_predicted, bce_dis_sampled
