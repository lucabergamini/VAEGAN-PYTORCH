import torch
import numpy

numpy.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)
from network import VaeGan
from torch.autograd import Variable
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torch.optim import RMSprop,Adam
from torch.optim.lr_scheduler import ExponentialLR
import progressbar
from torchvision.utils import make_grid
from generator import CELEBA
from utils import RollingMeasure

if __name__ == "__main__":

    writer = SummaryWriter(comment="_CELEBA_loss_1_test_23")
    net = VaeGan().cuda()
    # DATASET
    dataloader = torch.utils.data.DataLoader(CELEBA("/home/lapis/Desktop/img_align_celeba/train/"), batch_size=64,
                                             shuffle=True, num_workers=7)
    # DATASET for test
    # if you want to split train from test just move some files in another dir
    dataloader_test = torch.utils.data.DataLoader(CELEBA("/home/lapis/Desktop/img_align_celeba/test"), batch_size=64,
                                                  shuffle=False, num_workers=2)
    num_epochs = 30
    #margin and equilibirum
    margin = 0.35
    equilibrium = 0.68
    # OPTIM-LOSS
    # an optimizer for each of the sub-networks, so we can selectively backprop
    optimizer_encoder = RMSprop(params=net.encoder.parameters(), lr=0.0003)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=0.98)
    optimizer_decoder = RMSprop(params=net.decoder.parameters(), lr=0.0003)
    lr_decoder = ExponentialLR(optimizer_decoder, gamma=0.98)
    optimizer_discriminator = RMSprop(params=net.discriminator.parameters(), lr=0.0003)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=0.98)

    batch_number = len(dataloader)
    step_index = 0
    widgets = [

        'Batch: ', progressbar.Counter(),
        '/', progressbar.FormatCustomText('%(total)s', {"total": batch_number}),
        ' ', progressbar.Bar(marker="-", left='[', right=']'),
        ' ', progressbar.ETA(),
        ' ',
        progressbar.DynamicMessage('loss_nle'),
        ' ',
        progressbar.DynamicMessage('loss_encoder'),
        ' ',
        progressbar.DynamicMessage('loss_decoder'),
        ' ',
        progressbar.DynamicMessage('loss_discriminator'),
        ' ',
        progressbar.DynamicMessage("epoch")
    ]
    # for each epoch
    for i in range(num_epochs):
        progress = progressbar.ProgressBar(min_value=0, max_value=batch_number, initial_value=0,
                                           widgets=widgets).start()
        # reset rolling average
        loss_nle_mean = RollingMeasure()
        loss_encoder_mean = RollingMeasure()
        loss_decoder_mean = RollingMeasure()
        loss_discriminator_mean = RollingMeasure()
        print("LR:{}".format(lr_encoder.get_lr()))
        # for each batch
        for j, data_batch in enumerate(dataloader):
            # set to train mode
            net.train()
            # target and input are the same images
            data_target = Variable(data_batch, requires_grad=False).float().cuda()
            data_in = Variable(data_batch, requires_grad=True).float().cuda()
            # get output
            out, out_labels, out_layer, mus, variances = net(data_in)
            # split so we can get the different parts
            out_layer_original = out_layer[:len(out_layer) // 2]
            out_layer_predicted = out_layer[len(out_layer) // 2:]
            # TODO set a batch_len variable to get a clean code here
            out_labels_original = out_labels[:len(out_labels) // 3]
            out_labels_predicted = out_labels[len(out_labels) // 3:(len(out_labels) // 3) * 2]
            out_labels_sampled = out_labels[-len(out_labels) // 3:]
            # loss, nothing special here
            nle_value, kl_value, mse_value, bce_gen_predicted_value, bce_gen_sampled_value, bce_dis_original_value, \
            bce_dis_predicted_value, bce_dis_sampled_value = VaeGan.loss(data_target, out, out_layer_original,
                                                                         out_layer_predicted, out_labels_original,
                                                                         out_labels_predicted, out_labels_sampled, mus,
                                                                         variances)
            # THIS IS THE MOST IMPORTANT PART OF THE CODE
            loss_encoder = (kl_value + mse_value)
            loss_decoder = (1e-06 * mse_value - (bce_dis_original_value + bce_dis_sampled_value))
            loss_discriminator = (bce_dis_original_value + bce_dis_sampled_value)
            # selectively disable the decoder of the discriminator if they are unbalanced
            train_dis = True
            train_dec = True
            if bce_dis_original_value.data[0] < equilibrium-margin or bce_dis_sampled_value.data[0] < equilibrium-margin:
                train_dis = False
            if bce_dis_original_value.data[0] > equilibrium+margin or bce_dis_sampled_value.data[0] > equilibrium+margin:
                train_dec = False
            if train_dec is False and train_dis is False:
                train_dis = True
                train_dec = True

            # BACKPROP
            # clean grads
            net.zero_grad()
            # encoder
            loss_encoder.backward(retain_graph=True)
            # someone likes to clamp the grad here
            # [p.grad.data.clamp_(-1,1) for p in net.encoder.parameters()]
            # update parameters
            optimizer_encoder.step()
            # clean others, so they are not afflicted by encoder loss
            net.zero_grad()
            # decoder
            if train_dec:
                loss_decoder.backward(retain_graph=True)
                # [p.grad.data.clamp_(-1,1) for p in net.decoder.parameters()]
                optimizer_decoder.step()
                # clean the discriminator
                net.discriminator.zero_grad()
            # discriminator
            if train_dis:
                loss_discriminator.backward()
                # [p.grad.data.clamp_(-1,1) for p in net.discriminator.parameters()]
                optimizer_discriminator.step()

            # LOGGING
            progress.update(progress.value + 1, loss_nle=loss_nle_mean(nle_value.data.cpu().numpy()[0]),
                            loss_encoder=loss_encoder_mean(loss_encoder.data.cpu().numpy()[0]),
                            loss_decoder=loss_decoder_mean(loss_decoder.data.cpu().numpy()[0]),
                            loss_discriminator=loss_discriminator_mean(loss_discriminator.data.cpu().numpy()[0]),

                            epoch=i + 1)

        # EPOCH END
        lr_encoder.step()
        lr_decoder.step()
        lr_discriminator.step()
        progress.finish()

        writer.add_scalar('loss_encoder', loss_encoder_mean.measure, step_index)
        writer.add_scalar('loss_decoder', loss_decoder_mean.measure, step_index)
        writer.add_scalar('loss_discriminator', loss_discriminator_mean.measure, step_index)
        writer.add_scalar('loss_reconstruction', loss_nle_mean.measure, step_index)

        for j, data_batch in enumerate(dataloader_test):
            data_in = Variable(data_batch, requires_grad=True).float().cuda()
            out = net(data_in)
            out = out[0].data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=4)
            writer.add_image("reconstructed", out, step_index)

            net.eval()
            out = net(None, 64)
            out = out[0].data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=4)
            writer.add_image("generated", out, step_index)

            out = data_in.data.cpu()
            out = (out + 1) / 2
            out = make_grid(out, nrow=4)
            writer.add_image("original", out, step_index)
            break

        step_index += 1
