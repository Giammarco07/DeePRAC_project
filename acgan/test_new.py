import os
import torch
import neptune
from torch.autograd import Variable
from PIL import Image
import utils
from arch import define_Gen, define_Dis
from templeatedataset import templatedataset, templatedatasetnew, templatedatasetnew_test, templatedataset_paired, templatedataset_nii
from templeatedatasetA_bis import templatedatasetA_bis
from templeatedatasetB_bis import templatedatasetB_bis
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import sys
ee = sys.float_info.epsilon
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib

def np_to_img0(image,mode):
    I8 = (((image - image.min()) / (image.max() - image.min() + ee)) * 255.0).astype(np.uint8)
    img = Image.fromarray(I8, mode)
    return img

def np_to_img(image,mode):
    image = np.clip(image, -1.0, 1.0)
    image = (image + 1.0) / 2.0
    image = (image * 255.0).astype(np.uint8)
    img = Image.fromarray(image, mode)
    return img
    

def activation_map_test(x, y):
    x = np.clip(x, -1.0, 1.0)
    y = np.clip(y, -1.0, 1.0)
    image = abs(x - y)
    difference_plot = x-y
    image[image >= 0.2] = 1
    image[image < 1] = 0
    image_final = np_to_img0(image, 'L')
    return image_final, difference_plot

def test_nii(args):
    dataA = templatedataset_nii('/tsi/clusterhome/glabarbera/unet3d/nnUNet_raw_data_base/nnUNet_raw_data/Task408_NECKER/imagesTr') #ceCT
    dataB = templatedataset_nii('/tsi/clusterhome/glabarbera/unet3d/nnUNet_raw_data_base/nnUNet_raw_data/Task708_NECKER/imagesTr') #CT
    resultsA = '/tsi/clusterhome/glabarbera/unet3d/nnUNet_raw_data_base/nnUNet_raw_data/Task828_NECKER/imagesTr'
    resultsB = '/tsi/clusterhome/glabarbera/unet3d/nnUNet_raw_data_base/nnUNet_raw_data/Task928_NECKER/imagesTr'
    
    a_test_loader = torch.utils.data.DataLoader(dataA, batch_size=args.batch_size, shuffle=False)
    b_test_loader = torch.utils.data.DataLoader(dataB, batch_size=args.batch_size, shuffle=False)

    Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab, Gba], ['Gab', 'Gba'])

    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])  # load from the training


    """ run """
    Gab.eval()  # necessary to indicate at dropout and batch norm that they are now in evaluation mode.
    Gba.eval()
    print('Start real testing')
    for j, (a_real, b_real) in enumerate(zip(a_test_loader, b_test_loader)):
        a_real_test = a_real[0]
        b_real_test = b_real[0]

        name_a = a_real[1][0]
        name_b = b_real[1][0]
        hdr_a = nib.load(a_real[4][0]).header
        hdr_b = nib.load(b_real[4][0]).header
        aff_a = nib.load(a_real[4][0]).affine
        aff_b = nib.load(b_real[4][0]).affine

        minimum_a,maximum_a = a_real[2][0].detach().cpu().numpy(), a_real[3][0].detach().cpu().numpy()
        minimum_b,maximum_b = b_real[2][0].detach().cpu().numpy(), b_real[3][0].detach().cpu().numpy()

        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
        b_fake_test = a_real_test.clone()
        b_mask_test = a_real_test.clone()
        a_fake_test = b_real_test.clone()
        a_mask_test = b_real_test.clone()
        with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up.
            for k in range(b_real_test.size()[-1]):
                #a_fake_test[:,:,:,:,k],a_mask_test[:,:,:,:,k] = Gab(b_real_test[:,:,:,:,k])
                a_fake_test[:,:,:,:,k],_ = Gab(b_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()
            for k in range(a_real_test.size()[-1]):
                #b_fake_test[:,:,:,:,k],b_mask_test[:,:,:,:,k] = Gba(a_real_test[:,:,:,:,k])
                b_fake_test[:,:,:,:,k],_ = Gba(a_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()

        if not os.path.isdir(args.results_dir):  # make directory for result
            os.makedirs(args.results_dir)

        ####saving numpy array
        os.makedirs(args.results_dir + '/A_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/A_testpaired', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testpaired', exist_ok=True)

        a_fake = a_fake_test[0,0].detach().cpu().numpy()
        a_real = a_real_test[0,0].detach().cpu().numpy()
        #a_mask = 1 - a_mask_test[0,0].detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1
        b_fake = b_fake_test[0,0].detach().cpu().numpy()
        b_real = b_real_test[0,0].detach().cpu().numpy()        
        #b_mask = 1 - b_mask_test[0,0].detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1

        a_fake = (((a_fake - a_fake.min()) / (a_fake.max() - a_fake.min())) * (maximum_a-minimum_a)) + minimum_a
        b_fake = (((b_fake - b_fake.min()) / (b_fake.max() - b_fake.min())) * (maximum_b-minimum_b)) + minimum_b
        a_real = (((a_real - a_real.min()) / (a_real.max() - a_real.min())) * (maximum_a-minimum_a)) + minimum_a
        b_real = (((b_real - b_real.min()) / (b_real.max() - b_real.min())) * (maximum_b-minimum_b)) + minimum_b

        img_ar = nib.Nifti1Image(a_real, aff_a, hdr_a)
        nib.save(img_ar, os.path.join(resultsA,name_a+'_0000_0000.nii.gz'))
        img_bf = nib.Nifti1Image(b_fake, aff_a, hdr_a)
        nib.save(img_bf, os.path.join(resultsA,name_a+'_0000_0001.nii.gz'))        
        
        img_br = nib.Nifti1Image(b_real, aff_b, hdr_b)
        nib.save(img_br, os.path.join(resultsB,name_b+'_0000_0001.nii.gz'))
        img_af = nib.Nifti1Image(a_fake, aff_b, hdr_b)
        nib.save(img_af, os.path.join(resultsB,name_b+'_0000_0000.nii.gz') )   

        a_real = a_real_test.detach().cpu().numpy()
        a_fake = a_fake_test.detach().cpu().numpy()
        #a_mask = 1 - a_mask_test.detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1
        
        #a_fake = (((a_fake - a_fake.min()) / (a_fake.max() - a_fake.min())) * (maximum_a-minimum_a)) + minimum_a

        b_real = b_real_test.detach().cpu().numpy()
        b_fake = b_fake_test.detach().cpu().numpy()
        #b_mask = 1 - b_mask_test.detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1
        
        #b_fake = (((b_fake - b_fake.min()) / (b_fake.max() - b_fake.min())) * (maximum_b-minimum_b)) + minimum_b


        dima = a_real.shape[-1] // 2
        dimb = b_real.shape[-1] // 2

        a_real = np.array(a_real[0, 0, :,:, dima])
        b_fake = np.array(b_fake[0, 0, :, :, dima])

        b_real = np.array(b_real[0, 0, :, :, dimb])
        a_fake = np.array(a_fake[0, 0, :,:, dimb])



        images_a = np_to_img(a_fake, 'L')
        images_b = np_to_img(b_fake, 'L')
        images_a_real = np_to_img(a_real, 'L')
        images_b_real = np_to_img(b_real, 'L')
        activation_A, color_plot_A = activation_map_test(a_fake, b_real)
        activation_B, color_plot_B = activation_map_test(b_fake, a_real)
        os.makedirs(args.results_dir, exist_ok=True)
        images_a_real.save(args.results_dir + '/A_testNew/realA' + str(j) + '.png', 'png')
        images_a.save(args.results_dir + '/A_testNew/fakeA' + str(j) + '.png', 'png')
        images_b_real.save(args.results_dir + '/B_testNew/realB' + str(j) + '.png', 'png')
        images_b.save(args.results_dir + '/B_testNew/fakeB' + str(j) + '.png', 'png')
        activation_B.save(args.results_dir + '/B_testNew/activationB' + str(j) + '.png', 'png')
        activation_A.save(args.results_dir + '/A_testNew/activationA' + str(j) + '.png', 'png')

        os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
        imgA = plt.imshow(color_plot_A, interpolation='none')
        plt.colorbar(imgA)
        plt.savefig(args.results_dir + '/color_map_test/A' + str(j) + '.png')
        plt.close('all')
        imgB = plt.imshow(color_plot_B, interpolation='none')
        plt.colorbar(imgB)
        plt.savefig(args.results_dir + '/color_map_test/B' + str(j) + '.png')
        plt.close('all')

        neptune.log_image('A_real', images_a_real)
        neptune.log_image('A_fake', images_a)
        neptune.log_image('B_fake', images_b)
        neptune.log_image('B_reale', images_b_real)
        neptune.log_image('Activatione_A', images_b_real)
        neptune.log_image('Activatione_A', activation_A)
        neptune.log_image('Activatione_A', images_a)
        neptune.log_image('Activatione_B', images_a_real)
        neptune.log_image('Activatione_B', activation_B)
        neptune.log_image('Activatione_B', images_b)


def test(args):
    #dataA = templatedataset('/tsi/clusterhome/glabarbera/acgan/Database_GAN/ceCT-A')
    #dataB = templatedataset('/tsi/clusterhome/glabarbera/acgan/Database_GAN/CT-B')
    #dataA = templatedataset('/media/glabarbera/Donnees/ceCT-CT/ceCT/preprocessed')
    #dataB = templatedataset('/media/glabarbera/Donnees/ceCT-CT/CT/preprocessed')
    #dataA = templatedataset('/media/glabarbera/Donnees/nnUNet_preprocessed/Task202_NECKER_bis/nnUNetData_plans_v2.1_stage1')
    #dataB = templatedataset('/media/glabarbera/Donnees/nnUNet_preprocessed/Task202_NECKER_bis/nnUNetData_plans_v2.1_stage1')
    dataA = templatedataset('/tsi/clusterhome/glabarbera/unet3d/nnUNet_preprocessed/Task308_NECKER/nnUNetData_plans_v2.1_stage1_affine')
    dataB = templatedataset('/tsi/clusterhome/glabarbera/unet3d/nnUNet_preprocessed/Task308_NECKER/nnUNetData_plans_v2.1_stage1_affine')


    a_test_loader = torch.utils.data.DataLoader(dataA, batch_size=args.batch_size, shuffle=False)
    b_test_loader = torch.utils.data.DataLoader(dataB, batch_size=args.batch_size, shuffle=False)

    Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab, Gba], ['Gab', 'Gba'])

    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])  # load from the training


    """ run """
    Gab.eval()  # necessary to indicate at dropout and batch norm that they are now in evaluation mode.
    Gba.eval()
    print('Start real testing')
    for j, (a_real, b_real) in enumerate(zip(a_test_loader, b_test_loader)):
        a_real_test = a_real[0]
        b_real_test = b_real[0]

        seg_a = a_real[1][0].detach().cpu().numpy()
        seg_b = b_real[1][0].detach().cpu().numpy()

        name_a = a_real[2][0]
        name_b = b_real[2][0]

        shape_a = a_real[3][0].detach().cpu().numpy()
        shape_b = b_real[3][0].detach().cpu().numpy()

        minimum_a,maximum_a = a_real[4][0].detach().cpu().numpy(), a_real[5][0].detach().cpu().numpy()
        minimum_b,maximum_b = b_real[4][0].detach().cpu().numpy(), b_real[5][0].detach().cpu().numpy()

        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
        b_fake_test = a_real_test.clone()
        b_mask_test = a_real_test.clone()
        a_fake_test = b_real_test.clone()
        a_mask_test = b_real_test.clone()
        with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up.
            for k in range(b_real_test.size()[-1]):
                #a_fake_test[:,:,:,:,k],a_mask_test[:,:,:,:,k] = Gab(b_real_test[:,:,:,:,k])
                a_fake_test[:,:,:,:,k],_ = Gab(b_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()
            for k in range(a_real_test.size()[-1]):
                #b_fake_test[:,:,:,:,k],b_mask_test[:,:,:,:,k] = Gba(a_real_test[:,:,:,:,k])
                b_fake_test[:,:,:,:,k],_ = Gba(a_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()

        if not os.path.isdir(args.results_dir):  # make directory for result
            os.makedirs(args.results_dir)

        ####saving numpy array
        os.makedirs(args.results_dir + '/A_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/A_testpaired', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testpaired', exist_ok=True)

        a_fake = a_fake_test[0,0].detach().cpu().numpy()
        #a_mask = 1 - a_mask_test[0,0].detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1
        a_fake = skimage.transform.resize(a_fake, shape_a)
        a_fake = np.rollaxis(a_fake, 2, 0)
        b_fake = b_fake_test[0,0].detach().cpu().numpy()
        #b_mask = 1 - b_mask_test[0,0].detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1
        b_fake = skimage.transform.resize(b_fake, shape_b)
        b_fake = np.rollaxis(b_fake, 2, 0)

        a_fake = (((a_fake - a_fake.min()) / (a_fake.max() - a_fake.min())) * (maximum_a-minimum_a)) + minimum_a
        b_fake = (((b_fake - b_fake.min()) / (b_fake.max() - b_fake.min())) * (maximum_b-minimum_b)) + minimum_b

        np.savez_compressed(name_a[:-4] + '_fake_wc.npz',
                 np.asarray([a_fake, seg_a], dtype=np.float32))
        np.savez_compressed(name_b[:-4] + '_fake_nc.npz',
                 np.asarray([b_fake, seg_b], dtype=np.float32))

        a_real = a_real_test.detach().cpu().numpy()
        a_fake = a_fake_test.detach().cpu().numpy()
        #a_mask = 1 - a_mask_test.detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1
        a_fake = (((a_fake - a_fake.min()) / (a_fake.max() - a_fake.min())) * (maximum_a-minimum_a)) + minimum_a

        b_real = b_real_test.detach().cpu().numpy()
        b_fake = b_fake_test.detach().cpu().numpy()
        #b_mask = 1 - b_mask_test.detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1
        b_fake = (((b_fake - b_fake.min()) / (b_fake.max() - b_fake.min())) * (maximum_b-minimum_b)) + minimum_b



        dima = a_real.shape[-1] // 2
        dimb = b_real.shape[-1] // 2

        a_real = np.array(a_real[0, 0, :,:, dima])
        b_fake = np.array(b_fake[0, 0, :, :, dima])

        b_real = np.array(b_real[0, 0, :, :, dimb])
        a_fake = np.array(a_fake[0, 0, :,:, dimb])



        images_a = np_to_img(a_fake, 'L')
        images_b = np_to_img(b_fake, 'L')
        images_a_real = np_to_img(a_real, 'L')
        images_b_real = np_to_img(b_real, 'L')
        activation_A, color_plot_A = activation_map_test(a_fake, b_real)
        activation_B, color_plot_B = activation_map_test(b_fake, a_real)
        os.makedirs(args.results_dir, exist_ok=True)
        images_a_real.save(args.results_dir + '/A_testNew/realA' + str(j) + '.png', 'png')
        images_a.save(args.results_dir + '/A_testNew/fakeA' + str(j) + '.png', 'png')
        images_b_real.save(args.results_dir + '/B_testNew/realB' + str(j) + '.png', 'png')
        images_b.save(args.results_dir + '/B_testNew/fakeB' + str(j) + '.png', 'png')
        activation_B.save(args.results_dir + '/B_testNew/activationB' + str(j) + '.png', 'png')
        activation_A.save(args.results_dir + '/A_testNew/activationA' + str(j) + '.png', 'png')

        os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
        imgA = plt.imshow(color_plot_A, interpolation='none')
        plt.colorbar(imgA)
        plt.savefig(args.results_dir + '/color_map_test/A' + str(j) + '.png')
        plt.close('all')
        imgB = plt.imshow(color_plot_B, interpolation='none')
        plt.colorbar(imgB)
        plt.savefig(args.results_dir + '/color_map_test/B' + str(j) + '.png')
        plt.close('all')

        neptune.log_image('A_real', images_a_real)
        neptune.log_image('A_fake', images_a)
        neptune.log_image('B_fake', images_b)
        neptune.log_image('B_reale', images_b_real)
        neptune.log_image('Activatione_A', images_b_real)
        neptune.log_image('Activatione_A', activation_A)
        neptune.log_image('Activatione_A', images_a)
        neptune.log_image('Activatione_B', images_a_real)
        neptune.log_image('Activatione_B', activation_B)
        neptune.log_image('Activatione_B', images_b)


def test_paired(args):
    dataA = templatedataset_paired('/tsi/clusterhome/glabarbera/acgan/Database_GAN/pairedceCT-CT',data=0)
    dataB = templatedataset_paired('/tsi/clusterhome/glabarbera/acgan/Database_GAN/pairedceCT-CT',data=1)



    a_test_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=False)
    b_test_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=False)

    Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab, Gba], ['Gab', 'Gba'])

    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])  # load from the training


    """ run """
    Gab.eval()  # necessary to indicate at dropout and batch norm that they are now in evaluation mode.
    Gba.eval()

    MSE=np.zeros((9,2))
    SSIM=np.zeros((9,2))
    PSNR=np.zeros((9,2))

    print('Start real testing')
    for j, (a_real_test, b_real_test) in enumerate(zip(a_test_loader, b_test_loader)):

        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
        b_fake_test = a_real_test.clone()
        b_mask_test = a_real_test.clone()
        a_fake_test = b_real_test.clone()
        a_mask_test = b_real_test.clone()
        with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up.
            for k in range(b_real_test.size()[-1]):
                #a_fake_test[:,:,:,:,k],a_mask_test[:,:,:,:,k] = Gab(b_real_test[:,:,:,:,k])
                a_fake_test[:,:,:,:,k] = Gab(b_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()
            for k in range(a_real_test.size()[-1]):
                #b_fake_test[:,:,:,:,k],b_mask_test[:,:,:,:,k] = Gba(a_real_test[:,:,:,:,k])
                b_fake_test[:,:,:,:,k] = Gba(a_real_test[:,:,:,:,k])
            torch.cuda.empty_cache()

        if not os.path.isdir(args.results_dir):  # make directory for result
            os.makedirs(args.results_dir)

        ####saving numpy array
        os.makedirs(args.results_dir + '/A_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testNew', exist_ok=True)
        os.makedirs(args.results_dir + '/A_testpaired', exist_ok=True)
        os.makedirs(args.results_dir + '/B_testpaired', exist_ok=True)

        a_fake = a_fake_test[0,0].detach().cpu().numpy()
        #a_mask = 1 - a_mask_test[0,0].detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1
        a_real = a_real_test[0,0].detach().cpu().numpy()

        b_fake = b_fake_test[0,0].detach().cpu().numpy()
        #b_mask = 1 - b_mask_test[0,0].detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1
        b_real = b_real_test[0,0].detach().cpu().numpy()

        MSE[j,0]=mse(a_real,a_fake)
        MSE[j,1]=mse(b_real,b_fake)
        SSIM[j,0]=ssim(a_real,a_fake,data_range=2)
        SSIM[j,1]=ssim(b_real,b_fake,data_range=2)
        PSNR[j,0]=psnr(a_real,a_fake,data_range=2)
        PSNR[j,1]=psnr(b_real,b_fake,data_range=2)

        a_real = a_real_test.detach().cpu().numpy()
        a_fake = a_fake_test.detach().cpu().numpy()
        #a_mask = 1 - a_mask_test.detach().cpu().numpy()
        #a_fake[a_mask.astype(bool)] = -1


        b_real = b_real_test.detach().cpu().numpy()
        b_fake = b_fake_test.detach().cpu().numpy()
        #b_mask = 1 - b_mask_test.detach().cpu().numpy()
        #b_fake[b_mask.astype(bool)] = -1


        dima = a_real.shape[-1] // 2
        dimb = b_real.shape[-1] // 2

        a_real = np.array(a_real[0, 0, :,:, dima])
        b_fake = np.array(b_fake[0, 0, :, :, dima])

        b_real = np.array(b_real[0, 0, :, :, dimb])
        a_fake = np.array(a_fake[0, 0, :,:, dimb])



        images_a = np_to_img(a_fake, 'L')
        images_b = np_to_img(b_fake, 'L')
        images_a_real = np_to_img(a_real, 'L')
        images_b_real = np_to_img(b_real, 'L')
        activation_A, color_plot_A = activation_map_test(a_fake, b_real)
        activation_B, color_plot_B = activation_map_test(b_fake, a_real)
        os.makedirs(args.results_dir, exist_ok=True)
        images_a_real.save(args.results_dir + '/A_testNew/realA' + str(j) + '.png', 'png')
        images_a.save(args.results_dir + '/A_testNew/fakeA' + str(j) + '.png', 'png')
        images_b_real.save(args.results_dir + '/B_testNew/realB' + str(j) + '.png', 'png')
        images_b.save(args.results_dir + '/B_testNew/fakeB' + str(j) + '.png', 'png')
        activation_B.save(args.results_dir + '/B_testNew/activationB' + str(j) + '.png', 'png')
        activation_A.save(args.results_dir + '/A_testNew/activationA' + str(j) + '.png', 'png')

        os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
        imgA = plt.imshow(color_plot_A, interpolation='none')
        plt.colorbar(imgA)
        plt.savefig(args.results_dir + '/color_map_test/A' + str(j) + '.png')
        plt.close('all')
        imgB = plt.imshow(color_plot_B, interpolation='none')
        plt.colorbar(imgB)
        plt.savefig(args.results_dir + '/color_map_test/B' + str(j) + '.png')
        plt.close('all')

        neptune.log_image('A_real', images_a_real)
        neptune.log_image('A_fake', images_a)
        neptune.log_image('B_fake', images_b)
        neptune.log_image('B_reale', images_b_real)
        neptune.log_image('Activatione_A', images_b_real)
        neptune.log_image('Activatione_A', activation_A)
        neptune.log_image('Activatione_A', images_a)
        neptune.log_image('Activatione_B', images_a_real)
        neptune.log_image('Activatione_B', activation_B)
        neptune.log_image('Activatione_B', images_b)

    print(MSE)
    print(SSIM)
    print(PSNR)

def test_old(args):
    print('Preparing data...')
    norm = 3
    #dataA = templatedatasetnew_test('/tsi/clusterhome/glabarbera/acgan/Database_GAN/testA')  # ceCT
    #dataB = templatedatasetnew_test('/tsi/clusterhome/glabarbera/acgan/Database_GAN/testB')  # CT
    dataA = templatedatasetnew_test('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/ceCT-A',norm=norm)
    dataB = templatedatasetnew_test('/tsi/clusterhome/glabarbera/acgan/Database_GAN/Database_GAN_512/CT-B',norm=norm)

    batch_size = int(args.batch_size*10)
    if not os.path.isdir(args.results_dir):  # make directory for result
        os.makedirs(args.results_dir)

    ####saving numpy array
    os.makedirs(args.results_dir + '/A_testNew', exist_ok=True)
    os.makedirs(args.results_dir + '/B_testNew', exist_ok=True)

    a_test_loader = torch.utils.data.DataLoader(dataA, batch_size=1, shuffle=False)
    b_test_loader = torch.utils.data.DataLoader(dataB, batch_size=1, shuffle=False)

    Gab = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=1, output_nc=1, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                     use_dropout=args.dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab, Gba], ['Gab', 'Gba'])


    ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
    Gab.load_state_dict(ckpt['Gab'])
    Gba.load_state_dict(ckpt['Gba'])  # load from the training

    Gab.eval()  # necessary to indicate at dropout and batch norm that they are now in evaluation mode.
    Gba.eval()
    print('Start real testing')

    """ run """
    for i, (a_real_p, b_real_p) in enumerate(zip(a_test_loader, b_test_loader)):

        a_realp, b_realp = torch.transpose(a_real_p[0], 0, -1).unsqueeze(1), torch.transpose(b_real_p[0], 0, -1).unsqueeze(1)
        print(a_realp.shape, b_realp.shape)
        j = 0
        size = a_realp.size()[0]
        while (j<size):
            a_real_test = utils.cuda(a_realp[j:j+batch_size])

            with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up.

                b_fake_test, b_mask_test = Gba(a_real_test)
                #b_fake_test = Gba(a_real_test)


            a_real = np.array(a_real_test[0, 0, :, :].detach().cpu().numpy())
            b_fake = np.array(b_fake_test[0, 0, :, :].detach().cpu().numpy())

            b_mask = 1 - b_mask_test[0, 0].detach().cpu().numpy()
            b_fake[b_mask.astype(bool)] = -1 #b_fake.min()

            images_b = np_to_img(b_fake,'L')
            images_a_real = np_to_img(a_real, 'L')
            activation_B, color_plot_B = activation_map_test(b_fake, a_real)
            os.makedirs(args.results_dir, exist_ok=True)
            images_a_real.save(args.results_dir + '/A_testNew/realA' + str(i) + str(j) + '.png', 'png')
            images_b.save(args.results_dir + '/B_testNew/fakeB' + str(i) + str(j) + '.png', 'png')
            activation_B.save(args.results_dir + '/B_testNew/activationB' + str(i) + str(j) + '.png', 'png')
            os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
            imgB = plt.imshow(color_plot_B, interpolation='none')
            plt.colorbar(imgB)
            plt.savefig(args.results_dir + '/color_map_test/B' + str(i) + str(j) + '.png')
            plt.close('all')
            neptune.log_image('A_real', images_a_real)
            neptune.log_image('B_fake', images_b)
            neptune.log_image('Activatione_B', images_a_real)
            neptune.log_image('Activatione_B', activation_B)
            neptune.log_image('Activatione_B', images_b)

            if norm!=1 and j==0:
                for k in range(1,a_real_test.size()[0]):
                    a_real = np.array(a_real_test[k, 0, :, :].detach().cpu().numpy())
                    b_fake = np.array(b_fake_test[k, 0, :, :].detach().cpu().numpy())

                    b_mask = 1 - b_mask_test[k, 0].detach().cpu().numpy()
                    b_fake[b_mask.astype(bool)] = -1 #b_fake.min()

                    images_b = np_to_img(b_fake, 'L')
                    images_a_real = np_to_img(a_real, 'L')
                    activation_B, color_plot_B = activation_map_test(b_fake, a_real)
                    os.makedirs(args.results_dir, exist_ok=True)
                    images_a_real.save(args.results_dir + '/A_testNew/realA' + str(i) + str(j) + str(k) + '.png', 'png')
                    images_b.save(args.results_dir + '/B_testNew/fakeB' + str(i) + str(j) + str(k) + '.png', 'png')
                    activation_B.save(args.results_dir + '/B_testNew/activationB' + str(i) + str(j) + str(k) + '.png', 'png')
                    os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
                    imgB = plt.imshow(color_plot_B, interpolation='none')
                    plt.colorbar(imgB)
                    plt.savefig(args.results_dir + '/color_map_test/B' + str(i) + str(j) + str(k) + '.png')
                    plt.close('all')
                    neptune.log_image('A_real', images_a_real)
                    neptune.log_image('B_fake', images_b)
                    neptune.log_image('Activatione_B', images_a_real)
                    neptune.log_image('Activatione_B', activation_B)
                    neptune.log_image('Activatione_B', images_b)

            j += batch_size

        j = 0
        size = b_realp.size()[0]
        while (j < size):
            b_real_test = utils.cuda(b_realp[j:j + batch_size])
            with torch.no_grad():  # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up.


                a_fake_test,a_mask_test = Gab(b_real_test)
                #a_fake_test = Gab(b_real_test)

            b_real = np.array(b_real_test[0, 0, :, :].detach().cpu().numpy())
            a_fake = np.array(a_fake_test[0, 0, :, :].detach().cpu().numpy())

            a_mask = 1 - a_mask_test[0, 0].detach().cpu().numpy()
            a_fake[a_mask.astype(bool)] = -1 #a_fake.min()

            images_a = np_to_img(a_fake, 'L')
            images_b_real = np_to_img(b_real, 'L')
            activation_A, color_plot_A = activation_map_test(a_fake, b_real)
            os.makedirs(args.results_dir, exist_ok=True)
            images_b_real.save(args.results_dir + '/B_testNew/realB' + str(i) + str(j) + '.png', 'png')
            images_a.save(args.results_dir + '/A_testNew/fakeA' + str(i) + str(j) + '.png', 'png')
            activation_A.save(args.results_dir + '/A_testNew/activationA' + str(i) + str(j) + '.png', 'png')
            os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
            imgA = plt.imshow(color_plot_A, interpolation='none')
            plt.colorbar(imgA)
            plt.savefig(args.results_dir + '/color_map_test/A' + str(i) + str(j) + '.png')
            plt.close('all')
            neptune.log_image('B_real', images_b_real)
            neptune.log_image('A_fake', images_a)
            neptune.log_image('Activatione_A', images_b_real)
            neptune.log_image('Activatione_A', activation_A)
            neptune.log_image('Activatione_A', images_a)

            if norm!=1 and j==0:
                for k in range(1,b_real_test.size()[0]):
                    b_real = np.array(b_real_test[k, 0, :, :].detach().cpu().numpy())
                    a_fake = np.array(a_fake_test[k, 0, :, :].detach().cpu().numpy())

                    a_mask = 1 - a_mask_test[k, 0].detach().cpu().numpy()
                    a_fake[a_mask.astype(bool)] = -1 #a_fake.min()

                    images_a = np_to_img(a_fake, 'L')
                    images_b_real = np_to_img(b_real, 'L')
                    activation_A, color_plot_A = activation_map_test(a_fake, b_real)
                    os.makedirs(args.results_dir, exist_ok=True)
                    images_b_real.save(args.results_dir + '/B_testNew/realB' + str(i) + str(j) + str(k) + '.png', 'png')
                    images_a.save(args.results_dir + '/A_testNew/fakeA' + str(i) + str(j) + str(k) + '.png', 'png')
                    activation_A.save(args.results_dir + '/A_testNew/activationA' + str(i) + str(j) + str(k) + '.png', 'png')
                    os.makedirs(args.results_dir + '/color_map_test/', exist_ok=True)
                    imgA = plt.imshow(color_plot_A, interpolation='none')
                    plt.colorbar(imgA)
                    plt.savefig(args.results_dir + '/color_map_test/A' + str(i) + str(j) + str(k) + '.png')
                    plt.close('all')
                    neptune.log_image('B_real', images_b_real)
                    neptune.log_image('A_fake', images_a)
                    neptune.log_image('Activatione_A', images_b_real)
                    neptune.log_image('Activatione_A', activation_A)
                    neptune.log_image('Activatione_A', images_a)

            j += batch_size

