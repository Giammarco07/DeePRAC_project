import itertools
import os
import torch
from torch import nn
from torch.autograd import Variable
import utils
from utils import activation_map, keep_largest_mask_img_black
from utils import gradient_penalty
from utils import savingimage, savingimage2
from arch import define_Gen, define_Dis, set_grad
from templeatedataset import templatedataset,templatedatasetnew
import neptune
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
import numpy as np
'''
Class for CycleGAN with train() as a member function
'''
class cycleGANv0(object):
    def __init__(self, args):

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=args.lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedataset('/ldaphome/glabarbera/acgan/Database_GAN/trainA')
        dataB = templatedataset('/ldaphome/glabarbera/acgan/Database_GAN/trainB')

        print('number of slices for A is: ' + str(dataA.__len__()))
        print('number of slices for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=int(args.batch_size), shuffle=True, num_workers=8,drop_last=True)
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=int(args.batch_size), shuffle=True, num_workers=8,drop_last=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()

                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################

                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                ###saving image for epoch
                if epoch % 1 == 0 and i == 1:
                    a_real_img = savingimage(a_real)
                    b_real_img = savingimage(b_real)
                    images = savingimage(a_fake)
                    imagesB = savingimage(b_fake)
                    activation_map_A, color_plot_A = activation_map(b_real, a_fake)
                    activation_map_B, color_plot_B = activation_map(a_real, b_fake)
                    #mask = keep_largest_mask_img_black(b_real)
                    #maskB = keep_largest_mask_img_black(a_real)
                    os.makedirs(args.results_dir, exist_ok=True)
                    os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                    os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                    os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                    os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                    os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                    os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                    os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                    imgA = plt.imshow(color_plot_A, interpolation='none')
                    plt.colorbar(imgA)
                    plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                    plt.close('all')
                    os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                    imgB = plt.imshow(color_plot_B, interpolation='none')
                    plt.colorbar(imgB)
                    plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                    plt.close('all')

                    activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                    activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                    a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                    b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                    imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                    images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                    neptune.log_image('A_fake', images)
                    neptune.log_image('A_real', a_real_img)
                    neptune.log_image('B_fake', imagesB)
                    neptune.log_image('B_real', b_real_img)

                    ### saving activation maps
                    neptune.log_image('Activation map A', b_real_img)
                    neptune.log_image('Activation map A', images)
                    #neptune.log_image('Activation map A', mask)
                    neptune.log_image('Activation map A', activation_map_A)
                    neptune.log_image('Activation map B', a_real_img)
                    neptune.log_image('Activation map B', imagesB)
                    #neptune.log_image('Activation map B', maskB)
                    neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                #if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)


                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)
                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()

                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                #################################################
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))


                if not args.wsgan:
                    # Discriminator losses
                    ##################################################
                    a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                    a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                    b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                    b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                    # Total discriminators losses
                    a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                    b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                else:
                    # using WSGAN-GP for discriminator
                    gp_a = gradient_penalty(self.Da, a_real, a_fake)
                    gp_b = gradient_penalty(self.Db, b_real, b_fake)
                    a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                    b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)

class cycleGANv0_PBS(object):
    def __init__(self, args):

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        if args.wsgan:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            ck = 1
        except:
            print(' [*] No checkpoint!')
            ck = 0

        if ck == 1:
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        else:
            self.start_epoch = 0
            print('no (or problem with) load_state_dict or optimizers')

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainA') #ceCT
        dataB = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainB') #CT

        print('number of patients for A is: ' + str(dataA.__len__()))
        print('number of patients for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        if args.wsgan:
            one = torch.tensor(1, dtype=torch.float)
            mone = one * -1
            one, mone = utils.cuda([one, mone])
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real_p, b_real_p) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Functional interface
                print(min(a_real_p.shape[-1], b_real_p.shape[-1]))
                r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(args.batch_size))
                if np.argmin([a_real_p.shape[-1], b_real_p.shape[-1]]) == 0:
                    r1 = r
                    r2 = [int(rr * (b_real_p.shape[-1] - 1) / (a_real_p.shape[-1] - 1)) for rr in r]
                else:
                    r2 = r
                    r1 = [int(rr * (a_real_p.shape[-1] - 1) // (b_real_p.shape[-1] - 1)) for rr in r]
                a_real, b_real = a_real_p[..., r1], b_real_p[..., r2]
                a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,
                                                                                                 -1).unsqueeze(1)
                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################
                print(a_real.size(), b_real.size())
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                ###saving image for epoch
                if (epoch + 1) % args.batch_size == 0 and i == 1:
                    for bb in range(args.batch_size):
                        a_real_img = savingimage(a_real[bb])
                        b_real_img = savingimage(b_real[bb])
                        images = savingimage(a_fake[bb])
                        imagesB = savingimage(b_fake[bb])
                        activation_map_A, color_plot_A = activation_map(b_real[bb], a_fake[bb])
                        activation_map_B, color_plot_B = activation_map(a_real[bb], b_fake[bb])
                        mask = keep_largest_mask_img_black(b_real[bb])
                        maskB = keep_largest_mask_img_black(a_real[bb])
                        os.makedirs(args.results_dir, exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                        os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                        imgA = plt.imshow(color_plot_A, interpolation='none')
                        plt.colorbar(imgA)
                        plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                        plt.close('all')
                        os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                        imgB = plt.imshow(color_plot_B, interpolation='none')
                        plt.colorbar(imgB)
                        plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                        plt.close('all')

                        activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                        activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                        a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                        b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                        imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                        images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                        neptune.log_image('A_fake', images)
                        neptune.log_image('A_real', a_real_img)
                        neptune.log_image('B_fake', imagesB)
                        neptune.log_image('B_real', b_real_img)

                        ### saving activation maps
                        neptune.log_image('Activation map A', b_real_img)
                        neptune.log_image('Activation map A', images)
                        neptune.log_image('Activation map A', mask)
                        neptune.log_image('Activation map A', activation_map_A)
                        neptune.log_image('Activation map B', a_real_img)
                        neptune.log_image('Activation map B', imagesB)
                        neptune.log_image('Activation map B', maskB)
                        neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                # if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                    # Total generators losses
                    ###################################################
                    gen_loss = 10 * a_gen_loss + 10 * b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss
                    # gen_loss_partial = a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    # a_fake_dis.mean().backward(mone,retain_graph=True)
                    # b_fake_dis.mean().backward(mone,retain_graph=True)
                    gen_loss.backward()
                    self.g_optimizer.step()

                disc = 0
                while disc < 10:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                    b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                    a_fake, b_fake = utils.cuda([a_fake, b_fake])

                    # Forward pass through discriminators
                    #################################################
                    a_real_dis = self.Da(a_real)
                    a_fake_dis = self.Da(a_fake)
                    b_real_dis = self.Db(b_real)
                    b_fake_dis = self.Db(b_fake)
                    real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                    fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                    real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                    fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))

                    if not args.wsgan:
                        # Discriminator losses
                        ##################################################
                        a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                        a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                        b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                        b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                        # Total discriminators losses
                        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
                        # Update discriminators
                        ##################################################
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        disc = 10
                    else:
                        # using WSGAN-GP for discriminator
                        gp_a = gradient_penalty(self.Da, a_real, a_fake)
                        gp_b = gradient_penalty(self.Db, b_real, b_fake)
                        # a_real_dis.mean().backward(mone,retain_graph=True)
                        # b_real_dis.mean().backward(mone,retain_graph=True)
                        # a_fake_dis.mean().backward(one,retain_graph=True)
                        # b_fake_dis.mean().backward(one,retain_graph=True)
                        # gp_a.backward(retain_graph=True)
                        # gp_b.backward(retain_graph=True)
                        a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                        b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        # clip parameters in D
                        for p in self.Da.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        for p in self.Db.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        disc += 1

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)
            torch.cuda.empty_cache()


class cycleGANv0_affinePBS(object):
    def __init__(self, args):

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        if args.wsgan:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            ck = 1
        except:
            print(' [*] No checkpoint!')
            ck = 0

        if ck == 1:
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        else:
            self.start_epoch = 0
            print('no (or problem with) load_state_dict or optimizers')

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/trainA')  # ceCT
        dataB = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/trainB')  # CT

        print('number of patients for A is: ' + str(dataA.__len__()))
        print('number of patients for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        if args.wsgan:
            one = torch.tensor(1, dtype=torch.float)
            mone = one * -1
            one, mone = utils.cuda([one, mone])
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real_p, b_real_p) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Functional interface
                try:
                    elastixImageFilter = sitk.ElastixImageFilter()
                    elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(a_real_p[0].cpu().data.numpy()))
                    elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(b_real_p[0].cpu().data.numpy()))
                    AffineMap = sitk.GetDefaultParameterMap("affine")
                    AffineMap['DefaultPixelValue'] = ['-1']
                    elastixImageFilter.SetParameterMap(AffineMap)
                    b_real_p = torch.as_tensor(sitk.GetArrayFromImage(elastixImageFilter.Execute()),
                                               dtype=torch.float).unsqueeze(0)
                except:
                    pass

                print(min(a_real_p.shape[-1], b_real_p.shape[-1]))
                r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(args.batch_size))
                if np.argmin([a_real_p.shape[-1], b_real_p.shape[-1]]) == 0:
                    r1 = r
                    r2 = [int(rr * (b_real_p.shape[-1] - 1) / (a_real_p.shape[-1] - 1)) for rr in r]
                else:
                    r2 = r
                    r1 = [int(rr * (a_real_p.shape[-1] - 1) // (b_real_p.shape[-1] - 1)) for rr in r]
                a_real, b_real = a_real_p[..., r1], b_real_p[..., r2]
                a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,
                                                                                                 -1).unsqueeze(1)
                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################
                print(a_real.size(), b_real.size())
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                ###saving image for epoch
                if (epoch + 1) % args.batch_size == 0 and i == 1:
                    for bb in range(args.batch_size):
                        a_real_img = savingimage(a_real[bb])
                        b_real_img = savingimage(b_real[bb])
                        images = savingimage(a_fake[bb])
                        imagesB = savingimage(b_fake[bb])
                        activation_map_A, color_plot_A = activation_map(b_real[bb], a_fake[bb])
                        activation_map_B, color_plot_B = activation_map(a_real[bb], b_fake[bb])
                        mask = keep_largest_mask_img_black(b_real[bb])
                        maskB = keep_largest_mask_img_black(a_real[bb])
                        os.makedirs(args.results_dir, exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                        os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                        imgA = plt.imshow(color_plot_A, interpolation='none')
                        plt.colorbar(imgA)
                        plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                        plt.close('all')
                        os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                        imgB = plt.imshow(color_plot_B, interpolation='none')
                        plt.colorbar(imgB)
                        plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                        plt.close('all')

                        activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                        activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                        a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                        b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                        imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                        images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                        neptune.log_image('A_fake', images)
                        neptune.log_image('A_real', a_real_img)
                        neptune.log_image('B_fake', imagesB)
                        neptune.log_image('B_real', b_real_img)

                        ### saving activation maps
                        neptune.log_image('Activation map A', b_real_img)
                        neptune.log_image('Activation map A', images)
                        neptune.log_image('Activation map A', mask)
                        neptune.log_image('Activation map A', activation_map_A)
                        neptune.log_image('Activation map B', a_real_img)
                        neptune.log_image('Activation map B', imagesB)
                        neptune.log_image('Activation map B', maskB)
                        neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                # if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                    # Total generators losses
                    ###################################################
                    gen_loss = 10 * a_gen_loss + 10 * b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss
                    # gen_loss_partial = a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    # a_fake_dis.mean().backward(mone,retain_graph=True)
                    # b_fake_dis.mean().backward(mone,retain_graph=True)
                    gen_loss.backward()
                    self.g_optimizer.step()

                disc = 0
                while disc < 10:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                    b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                    a_fake, b_fake = utils.cuda([a_fake, b_fake])

                    # Forward pass through discriminators
                    #################################################
                    a_real_dis = self.Da(a_real)
                    a_fake_dis = self.Da(a_fake)
                    b_real_dis = self.Db(b_real)
                    b_fake_dis = self.Db(b_fake)
                    real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                    fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                    real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                    fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))

                    if not args.wsgan:
                        # Discriminator losses
                        ##################################################
                        a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                        a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                        b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                        b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                        # Total discriminators losses
                        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
                        # Update discriminators
                        ##################################################
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        disc = 10
                    else:
                        # using WSGAN-GP for discriminator
                        gp_a = gradient_penalty(self.Da, a_real, a_fake)
                        gp_b = gradient_penalty(self.Db, b_real, b_fake)
                        # a_real_dis.mean().backward(mone,retain_graph=True)
                        # b_real_dis.mean().backward(mone,retain_graph=True)
                        # a_fake_dis.mean().backward(one,retain_graph=True)
                        # b_fake_dis.mean().backward(one,retain_graph=True)
                        # gp_a.backward(retain_graph=True)
                        # gp_b.backward(retain_graph=True)
                        a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                        b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        # clip parameters in D
                        for p in self.Da.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        for p in self.Db.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        disc += 1

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)
            torch.cuda.empty_cache()


class cycleGANv1_ssbrselection(object):
    def __init__(self, args):
        self.patch_size = args.crop_height

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.netC = torch.load('/tsi/clusterhome/glabarbera/acgan/ssbrfinal').cuda()
        # set_grad([self.netC], False)
        self.netC.eval()

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db, self.netC], ['Gab', 'Gba', 'Da', 'Db', 'SSBR'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        if args.wsgan:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            ck = 1
        except:
            print(' [*] No checkpoint!')
            ck = 0

        if ck == 1:
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        else:
            self.start_epoch = 0
            print('no (or problem with) load_state_dict or optimizers')

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainA')  # ceCT
        dataB = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainB')  # CT

        print('number of patients for A is: ' + str(dataA.__len__()))
        print('number of patients for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=True, num_workers=0, drop_last=True)  # 8
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=True, num_workers=0, drop_last=True)  # 8

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        if args.wsgan:
            one = torch.tensor(1, dtype=torch.float)
            mone = one * -1
            one, mone = utils.cuda([one, mone])
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real_p, b_real_p) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Functional interface
                score = np.zeros((a_real_p.shape[-1]))
                for slice in range(0, a_real_p.shape[-1]):
                    with torch.no_grad():
                        sc = self.netC(a_real_p[:, :, :, slice].view(1, 1, self.patch_size, self.patch_size).cuda())
                    score[slice] = np.around(sc.cpu().detach().numpy(), decimals=2)

                score1 = np.zeros((b_real_p.shape[-1]))
                for slice in range(0, b_real_p.shape[-1]):
                    with torch.no_grad():
                        sc = self.netC(b_real_p[:, :, :, slice].view(1, 1, self.patch_size, self.patch_size).cuda())
                    score1[slice] = np.around(sc.cpu().detach().numpy(), decimals=2)

                if (max(score.min(), score1.min()) + 0.1) < min(score.max(), score1.max()):
                    r = random.sample(
                        list(np.arange(max(score.min(), score1.min()), min(score.max(), score1.max()), step=0.01)),
                        k=int(args.batch_size))
                    r = np.around(r, decimals=2)
                    a_real, b_real = torch.zeros((1, self.patch_size, self.patch_size, args.batch_size)), torch.zeros(
                        (1, self.patch_size, self.patch_size, args.batch_size))
                    for batch, slice in enumerate(r):
                        a_real[..., batch] = a_real_p[..., (np.abs(score - slice)).argmin()]
                        b_real[..., batch] = b_real_p[..., (np.abs(score1 - slice)).argmin()]
                else:
                    r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(args.batch_size))
                    a_real, b_real = a_real_p[..., r], b_real_p[..., r]

                a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,
                                                                                                 -1).unsqueeze(1)
                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################
                print(a_real.size(), b_real.size())
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                ###saving image for epoch
                if (epoch + 1) % args.batch_size == 0 and i == 1:
                    for bb in range(args.batch_size):
                        a_real_img = savingimage(a_real[bb])
                        b_real_img = savingimage(b_real[bb])
                        images = savingimage(a_fake[bb])
                        imagesB = savingimage(b_fake[bb])
                        activation_map_A, color_plot_A = activation_map(b_real[bb], a_fake[bb])
                        activation_map_B, color_plot_B = activation_map(a_real[bb], b_fake[bb])
                        mask = keep_largest_mask_img_black(b_real[bb])
                        maskB = keep_largest_mask_img_black(a_real[bb])
                        os.makedirs(args.results_dir, exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                        os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                        imgA = plt.imshow(color_plot_A, interpolation='none')
                        plt.colorbar(imgA)
                        plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                        plt.close('all')
                        os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                        imgB = plt.imshow(color_plot_B, interpolation='none')
                        plt.colorbar(imgB)
                        plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                        plt.close('all')

                        activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                        activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                        a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                        b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                        imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                        images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                        neptune.log_image('A_fake', images)
                        neptune.log_image('A_real', a_real_img)
                        neptune.log_image('B_fake', imagesB)
                        neptune.log_image('B_real', b_real_img)

                        ### saving activation maps
                        neptune.log_image('Activation map A', b_real_img)
                        neptune.log_image('Activation map A', images)
                        neptune.log_image('Activation map A', mask)
                        neptune.log_image('Activation map A', activation_map_A)
                        neptune.log_image('Activation map B', a_real_img)
                        neptune.log_image('Activation map B', imagesB)
                        neptune.log_image('Activation map B', maskB)
                        neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                # if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                    # Total generators losses
                    ###################################################
                    gen_loss = 10 * a_gen_loss + 10 * b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss
                    # gen_loss_partial = a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    # a_fake_dis.mean().backward(mone,retain_graph=True)
                    # b_fake_dis.mean().backward(mone,retain_graph=True)
                    gen_loss.backward()
                    self.g_optimizer.step()

                disc = 0
                while disc < 10:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                    b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                    a_fake, b_fake = utils.cuda([a_fake, b_fake])

                    # Forward pass through discriminators
                    #################################################
                    a_real_dis = self.Da(a_real)
                    a_fake_dis = self.Da(a_fake)
                    b_real_dis = self.Db(b_real)
                    b_fake_dis = self.Db(b_fake)
                    real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                    fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                    real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                    fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))

                    if not args.wsgan:
                        # Discriminator losses
                        ##################################################
                        a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                        a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                        b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                        b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                        # Total discriminators losses
                        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
                        # Update discriminators
                        ##################################################
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        disc = 10
                    else:
                        # using WSGAN-GP for discriminator
                        gp_a = gradient_penalty(self.Da, a_real, a_fake)
                        gp_b = gradient_penalty(self.Db, b_real, b_fake)
                        # a_real_dis.mean().backward(mone,retain_graph=True)
                        # b_real_dis.mean().backward(mone,retain_graph=True)
                        # a_fake_dis.mean().backward(one,retain_graph=True)
                        # b_fake_dis.mean().backward(one,retain_graph=True)
                        # gp_a.backward(retain_graph=True)
                        # gp_b.backward(retain_graph=True)
                        a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                        b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        # clip parameters in D
                        for p in self.Da.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        for p in self.Db.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        disc += 1

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)
            torch.cuda.empty_cache()

class cycleGANv1_ssbrloss(object):
    def __init__(self, args):

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.netC = torch.load('/tsi/clusterhome/glabarbera/acgan/ssbrfinal').cuda()
        #set_grad([self.netC], False)
        self.netC.eval()

        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db, self.netC], ['Gab', 'Gba', 'Da', 'Db','SSBR'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        if args.wsgan:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            ck = 1
        except:
            print(' [*] No checkpoint!')
            ck = 0

        if ck == 1:
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        else:
            self.start_epoch = 0
            print('no (or problem with) load_state_dict or optimizers')

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainA')  # ceCT
        dataB = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/trainB')  # CT

        print('number of patients for A is: ' + str(dataA.__len__()))
        print('number of patients for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=True, num_workers=8, drop_last=True)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        if args.wsgan:
            one = torch.tensor(1, dtype=torch.float)
            mone = one * -1
            one, mone = utils.cuda([one, mone])
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real_p, b_real_p) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Functional interface

                r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(args.batch_size))
                a_real, b_real = a_real_p[..., r], b_real_p[..., r]

                a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,
                                                                                                 -1).unsqueeze(1)
                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################
                print(a_real.size(), b_real.size())
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                a_real_ssbr = self.netC(a_real)
                b_real_ssbr = self.netC(b_real)
                a_fake_ssbr = self.netC(a_fake)
                b_fake_ssbr = self.netC(b_fake)


                ###saving image for epoch
                if (epoch + 1) % args.batch_size == 0 and i == 1:
                    for bb in range(args.batch_size):
                        a_real_img = savingimage(a_real[bb])
                        b_real_img = savingimage(b_real[bb])
                        images = savingimage(a_fake[bb])
                        imagesB = savingimage(b_fake[bb])
                        activation_map_A, color_plot_A = activation_map(b_real[bb], a_fake[bb])
                        activation_map_B, color_plot_B = activation_map(a_real[bb], b_fake[bb])
                        mask = keep_largest_mask_img_black(b_real[bb])
                        maskB = keep_largest_mask_img_black(a_real[bb])
                        os.makedirs(args.results_dir, exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                        os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                        imgA = plt.imshow(color_plot_A, interpolation='none')
                        plt.colorbar(imgA)
                        plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                        plt.close('all')
                        os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                        imgB = plt.imshow(color_plot_B, interpolation='none')
                        plt.colorbar(imgB)
                        plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                        plt.close('all')

                        activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                        activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                        a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                        b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                        imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                        images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                        neptune.log_image('A_fake', images)
                        neptune.log_image('A_real', a_real_img)
                        neptune.log_image('B_fake', imagesB)
                        neptune.log_image('B_real', b_real_img)

                        ### saving activation maps
                        neptune.log_image('Activation map A', b_real_img)
                        neptune.log_image('Activation map A', images)
                        neptune.log_image('Activation map A', mask)
                        neptune.log_image('Activation map A', activation_map_A)
                        neptune.log_image('Activation map B', a_real_img)
                        neptune.log_image('Activation map B', imagesB)
                        neptune.log_image('Activation map B', maskB)
                        neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                # if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # SSBR losses
                ###################################################
                a_ssbr_loss = self.L1(b_fake_ssbr, a_real_ssbr)
                b_ssbr_loss = self.L1(a_fake_ssbr, b_real_ssbr)

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss + a_ssbr_loss + b_ssbr_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                    # Total generators losses
                    ###################################################
                    gen_loss = 10 * a_gen_loss + 10 * b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss + a_ssbr_loss + b_ssbr_loss
                    # gen_loss_partial = a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    # a_fake_dis.mean().backward(mone,retain_graph=True)
                    # b_fake_dis.mean().backward(mone,retain_graph=True)
                    gen_loss.backward()
                    self.g_optimizer.step()

                disc = 0
                while disc < 10:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                    b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                    a_fake, b_fake = utils.cuda([a_fake, b_fake])

                    # Forward pass through discriminators
                    #################################################
                    a_real_dis = self.Da(a_real)
                    a_fake_dis = self.Da(a_fake)
                    b_real_dis = self.Db(b_real)
                    b_fake_dis = self.Db(b_fake)
                    real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                    fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                    real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                    fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))

                    if not args.wsgan:
                        # Discriminator losses
                        ##################################################
                        a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                        a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                        b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                        b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                        # Total discriminators losses
                        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
                        # Update discriminators
                        ##################################################
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        disc = 10
                    else:
                        # using WSGAN-GP for discriminator
                        gp_a = gradient_penalty(self.Da, a_real, a_fake)
                        gp_b = gradient_penalty(self.Db, b_real, b_fake)
                        # a_real_dis.mean().backward(mone,retain_graph=True)
                        # b_real_dis.mean().backward(mone,retain_graph=True)
                        # a_fake_dis.mean().backward(one,retain_graph=True)
                        # b_fake_dis.mean().backward(one,retain_graph=True)
                        # gp_a.backward(retain_graph=True)
                        # gp_b.backward(retain_graph=True)
                        a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                        b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        # clip parameters in D
                        for p in self.Da.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        for p in self.Db.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        disc += 1

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            neptune.log_metric('ssbr_loss', a_ssbr_loss + b_ssbr_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)
            torch.cuda.empty_cache()

class cycleGANv1_ssbracgan(object):
    def __init__(self, args):
        self.patch_size = args.crop_height

        # Define the network
        ###################################################### change the number of channel,also in utils
        self.Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=args.dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=1, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.netC = torch.load('/tsi/clusterhome/glabarbera/acgan/ssbrfinal').cuda()
        #set_grad([self.netC], False)
        self.netC.eval()
        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db, self.netC], ['Gab', 'Gba', 'Da', 'Db','SSBR'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        # Optimizers
        #####################################################
        if args.lr_g == args.lr:
            print('Learning Rate equals')
        if args.wsgan:
            lr = args.lr * 0.1
        else:
            lr = args.lr
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr_g,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            ck = 1
        except:
            print(' [*] No checkpoint!')
            ck = 0

        if ck == 1:
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        else:
            self.start_epoch = 0
            print('no (or problem with) load_state_dict or optimizers')

    def train(self, args):
        # For transforming the input image
        # transform = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor()])

        dataA = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/ceCT-A', norm=2)  # ceCT
        dataB = templatedatasetnew('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/CT-B', norm=2) # CT

        print('number of patients for A is: ' + str(dataA.__len__()))
        print('number of patients for B is: ' + str(dataB.__len__()))
        neptune.log_metric('lenght dataA', dataA.__len__())
        neptune.log_metric('lenght dataB', dataB.__len__())
        # Pytorch dataloader
        #a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
        #b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
        a_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        b_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        if args.wsgan:
            one = torch.tensor(1, dtype=torch.float)
            mone = one * -1
            one, mone = utils.cuda([one, mone])
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real_p, b_real_p) in enumerate(zip(a_loader, b_loader)):

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)  # freeze the discriminator
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                # Functional interface
                score = np.zeros((a_real_p.shape[-1]))
                for slice in range(0, a_real_p.shape[-1]):
                    with torch.no_grad():
                        sc = self.netC(a_real_p[:, :, :, slice].view(1, 1, self.patch_size, self.patch_size).cuda())
                    score[slice] = np.around(sc.cpu().detach().numpy(), decimals=2)

                score1 = np.zeros((b_real_p.shape[-1]))
                for slice in range(0, b_real_p.shape[-1]):
                    with torch.no_grad():
                        sc = self.netC(b_real_p[:, :, :, slice].view(1, 1, self.patch_size, self.patch_size).cuda())
                    score1[slice] = np.around(sc.cpu().detach().numpy(), decimals=2)

                if (max(score.min(),score1.min())+0.1)<min(score.max(),score1.max()):
                    r = random.sample(list(np.arange(max(score.min(),score1.min()), min(score.max(),score1.max()), step=0.01)), k=int(args.batch_size))
                    r = np.around(r, decimals=2)
                    a_real, b_real = torch.zeros((1,self.patch_size,self.patch_size,args.batch_size)),torch.zeros((1,self.patch_size,self.patch_size,args.batch_size))
                    for batch,slice in enumerate(r):
                        a_real[...,batch] = a_real_p[...,(np.abs(score - slice)).argmin()]
                        b_real[...,batch] = b_real_p[...,(np.abs(score1 - slice)).argmin()]
                else:
                    r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(args.batch_size))
                    a_real, b_real = a_real_p[..., r], b_real_p[..., r]


                a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,
                                                                                                 -1).unsqueeze(1)
                a_real, b_real = utils.cuda([a_real, b_real])
                # Forward pass through generators
                ##################################################
                print(a_real.size(), b_real.size())
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                a_real_ssbr = self.netC(a_real)
                b_real_ssbr = self.netC(b_real)
                a_fake_ssbr = self.netC(a_fake)
                b_fake_ssbr = self.netC(b_fake)


                ###saving image for epoch
                if (epoch + 1) % args.batch_size == 0 and i == 1:
                    for bb in range(args.batch_size):
                        a_real_img = savingimage(a_real[bb])
                        b_real_img = savingimage(b_real[bb])
                        images = savingimage(a_fake[bb])
                        imagesB = savingimage(b_fake[bb])
                        activation_map_A, color_plot_A = activation_map(b_real[bb], a_fake[bb])
                        activation_map_B, color_plot_B = activation_map(a_real[bb], b_fake[bb])
                        mask = keep_largest_mask_img_black(b_real[bb])
                        maskB = keep_largest_mask_img_black(a_real[bb])
                        os.makedirs(args.results_dir, exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_real', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/A_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/B_fake', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_A', exist_ok=True)
                        os.makedirs(args.results_dir + '/images/maps_B', exist_ok=True)

                        os.makedirs(args.results_dir + '/images/color_map_A', exist_ok=True)
                        imgA = plt.imshow(color_plot_A, interpolation='none')
                        plt.colorbar(imgA)
                        plt.savefig(args.results_dir + '/images/color_map_A/' + str(epoch) + '.png')
                        plt.close('all')
                        os.makedirs(args.results_dir + '/images/color_map_B', exist_ok=True)
                        imgB = plt.imshow(color_plot_B, interpolation='none')
                        plt.colorbar(imgB)
                        plt.savefig(args.results_dir + '/images/color_map_B/' + str(epoch) + '.png')
                        plt.close('all')

                        activation_map_A.save(args.results_dir + '/images/maps_A/' + str(epoch) + '.png', 'png')
                        activation_map_B.save(args.results_dir + '/images/maps_B/' + str(epoch) + '.png', 'png')
                        a_real_img.save(args.results_dir + '/images/A_real/' + str(epoch) + '.png', 'png')
                        b_real_img.save(args.results_dir + '/images/B_real/' + str(epoch) + '.png', 'png')
                        imagesB.save(args.results_dir + '/images/B_fake/' + str(epoch) + '.png', 'png')
                        images.save(args.results_dir + '/images/A_fake/' + str(epoch) + '.png', 'png')
                        neptune.log_image('A_fake', images)
                        neptune.log_image('A_real', a_real_img)
                        neptune.log_image('B_fake', imagesB)
                        neptune.log_image('B_real', b_real_img)

                        ### saving activation maps
                        neptune.log_image('Activation map A', b_real_img)
                        neptune.log_image('Activation map A', images)
                        neptune.log_image('Activation map A', mask)
                        neptune.log_image('Activation map A', activation_map_A)
                        neptune.log_image('Activation map B', a_real_img)
                        neptune.log_image('Activation map B', imagesB)
                        neptune.log_image('Activation map B', maskB)
                        neptune.log_image('Activation map B', activation_map_B)
                    ####

                ###### update pesi dell'identity  e cycle#
                # if epoch > 149 and epoch % 10 == 0 and i == 1:
                #     args.idt_coef = args.idt_coef - 0.0625
                #     args.lamda = args.lamda - 1.25
                #     print('cambiato pesi dei lambda')
                ########

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.idt_coef

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # SSBR losses
                ###################################################
                a_ssbr_loss = self.L1(b_fake_ssbr, a_real_ssbr)
                b_ssbr_loss = self.L1(a_fake_ssbr, b_real_ssbr)

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                # real_label = utils.cuda(torch.ones(a_fake_dis.size()))
                # da usare perchè ho dimensioni diverse per l'ultima iterazione del batch_size
                real_label_A = utils.cuda(torch.ones(a_fake_dis.size()))
                real_label_B = utils.cuda(torch.ones(b_fake_dis.size()))

                if not args.wsgan:
                    a_gen_loss = self.MSE(a_fake_dis, real_label_A)
                    b_gen_loss = self.MSE(b_fake_dis, real_label_B)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss + a_ssbr_loss + b_ssbr_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                else:
                    a_gen_loss = -torch.mean(a_fake_dis)
                    b_gen_loss = -torch.mean(b_fake_dis)

                    # Total generators losses
                    ###################################################
                    gen_loss = 10 * a_gen_loss + 10 * b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss + a_ssbr_loss + b_ssbr_loss
                    # gen_loss_partial = a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    # a_fake_dis.mean().backward(mone,retain_graph=True)
                    # b_fake_dis.mean().backward(mone,retain_graph=True)
                    gen_loss.backward()
                    self.g_optimizer.step()

                disc = 0
                while disc < 10:
                    # Discriminator Computations
                    #################################################

                    set_grad([self.Da, self.Db], True)  # now grad ind the discriminators are activated
                    self.g_optimizer.zero_grad()
                    self.d_optimizer.zero_grad()

                    # Sample from history of generated images
                    #################################################
                    a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                    b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                    a_fake, b_fake = utils.cuda([a_fake, b_fake])

                    # Forward pass through discriminators
                    #################################################
                    a_real_dis = self.Da(a_real)
                    a_fake_dis = self.Da(a_fake)
                    b_real_dis = self.Db(b_real)
                    b_fake_dis = self.Db(b_fake)
                    real_label_A = utils.cuda(torch.ones(a_real_dis.size()))
                    fake_label_A = utils.cuda(torch.zeros(a_fake_dis.size()))
                    real_label_B = utils.cuda(torch.ones(b_real_dis.size()))
                    fake_label_B = utils.cuda(torch.zeros(b_fake_dis.size()))

                    if not args.wsgan:
                        # Discriminator losses
                        ##################################################
                        a_dis_real_loss = self.MSE(a_real_dis, real_label_A)
                        a_dis_fake_loss = self.MSE(a_fake_dis, fake_label_A)
                        b_dis_real_loss = self.MSE(b_real_dis, real_label_B)
                        b_dis_fake_loss = self.MSE(b_fake_dis, fake_label_B)

                        # Total discriminators losses
                        a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                        b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5
                        # Update discriminators
                        ##################################################
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        disc = 10
                    else:
                        # using WSGAN-GP for discriminator
                        gp_a = gradient_penalty(self.Da, a_real, a_fake)
                        gp_b = gradient_penalty(self.Db, b_real, b_fake)
                        # a_real_dis.mean().backward(mone,retain_graph=True)
                        # b_real_dis.mean().backward(mone,retain_graph=True)
                        # a_fake_dis.mean().backward(one,retain_graph=True)
                        # b_fake_dis.mean().backward(one,retain_graph=True)
                        # gp_a.backward(retain_graph=True)
                        # gp_b.backward(retain_graph=True)
                        a_dis_loss = -(torch.mean(a_real_dis) - torch.mean(a_fake_dis)) + gp_a
                        b_dis_loss = -(torch.mean(b_real_dis) - torch.mean(b_fake_dis)) + gp_b
                        a_dis_loss.backward()
                        b_dis_loss.backward()
                        self.d_optimizer.step()
                        # clip parameters in D
                        for p in self.Da.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        for p in self.Db.parameters():
                            p.data.clamp_(-0.01, 0.01)
                        disc += 1

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            neptune.log_metric('gen_loss', gen_loss)
            neptune.log_metric('discriminator_loss', a_dis_loss + b_dis_loss)
            neptune.log_metric('cycle_loss', a_cycle_loss + b_cycle_loss)
            neptune.log_metric('identity_loss', a_idt_loss + b_idt_loss)
            neptune.log_metric('gn_general_loss', a_gen_loss + b_gen_loss)
            neptune.log_metric('ssbr_loss', a_ssbr_loss + b_ssbr_loss)
            if not args.wsgan:
                neptune.log_metric('D_real_loss', a_dis_real_loss + b_dis_real_loss)
                neptune.log_metric('D_fake_loss', a_dis_fake_loss + b_dis_fake_loss)

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
            neptune.log_metric('lr', lr)
            torch.cuda.empty_cache()



