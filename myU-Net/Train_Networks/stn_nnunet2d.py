from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
th.backends.cudnn.deterministic = False
th.backends.cudnn.benchmark = True
#the cudnn heuristics to pick the fastest algorithms for your workload.
#Note that the first iteration for each new input shape will be slow, as cudnn is benchmarking the kernels,
#so you should profile the model after a few warmup iterations.
import numpy as np
import matplotlib
matplotlib.use('agg')
import pylab as plt
import time
import sys
ee = sys.float_info.epsilon
from skimage.transform import resize
from utils.utils import referencef, inv_affine, keep_largest, referencef2
from utils.figures import ensemble, np_to_img
from utils.losses import L1, L2, ce, soft_dice_loss, dice_loss, dice_loss_val, soft_dice_loss_old

def train_stncrop_nnunet2d(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    net2 = nets[1]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    if start_epoch>=50:
        lr = init_lr * ((1 - ((start_epoch - 50) / num_epochs)) ** 0.9)
        optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        optimizer2 = optim.SGD(net2.parameters(), lr=init_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    valdices = []
    valdicesstn = []
    vals1 = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    if channel_dim > 2:
        vals2 = []
        train_loader2 = train_loader[2]
        val_loader2 = val_loader[2]
    if channel_dim > 3:
        vals3 = []
        train_loader3 = train_loader[3]
        val_loader3 = val_loader[3]
    mloss1 = []
    mloss2 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
        G_losses2 = []
        torch.cuda.synchronize()
        step = time.time()
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            # print(i)
            torch.cuda.synchronize()
            step1 = time.time()
            if mode=='debug':
                if i==0 :
                    print("Load images (s):", step1- step)
                else:
                    print("Load images (s):", step1 - step6)

            image0, target0 = data0
            image1, target1 = data1
            image = torch.cat((image0, image1), 0)
            target = torch.cat((target0, target1), 0)
            rand = torch.randperm(image.size()[0])
            image = image[rand]
            target = target[rand]
            image = image.to(device)
            target = target.to(device)

            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if epoch < 50:
                if (i % 1000) == 0:
                    print('Iteration: ', i)
                net1.train()
                net2.eval()

                optimizer1.zero_grad()

                image_crop, theta_crop = referencef(image, ensemble(target, channel_dim).squeeze(1))

                x_affine, theta = net1(image)

                torch.cuda.synchronize()
                step3 = time.time()
                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint1 = L1(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop))
                #loss_G_joint1 = torch.sqrt(L2(theta, theta_crop))

                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint1.backward()
                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer1.step()


                # Save Losses for plotting later
                G_losses1.append(loss_G_joint1.item())
                torch.cuda.synchronize()
                step6 = time.time()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del theta_crop
                del image_crop

                # ---------END TRAIN FIRST NET---------START TRAIN SECOND NET------
            else:
                if (i % 1000) == 0:
                    print('Iteration: ', i)
                net2.train()
                net1.eval()
                if epoch == 50 and i==0:
                    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
                #seg = ensemble(target, channel_dim).squeeze(1)
                seg = torch.argmax(target, dim=1)
                optimizer2.zero_grad()
                with torch.no_grad():
                    x_affine, theta = net1(image)
                output_affine = net2(x_affine)


                torch.cuda.synchronize()
                step3 = time.time()
                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                if supervision == 'deep':
                    w = (32 / 63)
                    for p in range(6):
                        if p ==0:
                            output = inv_affine(output_affine[p], theta)

                            zero = output[:,0,:,:]
                            zero[zero == 0] = 1
                            output[:,0,:,:] = zero

                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output + 1e-20), seg)
                            loss_G_joint2 = w*(1/(2**p))*(loss_dice + loss_ce)
                        else:
                            output = inv_affine(output_affine[p], theta)

                            zero = output[:, 0, :, :]
                            zero[zero == 0] = 1
                            output[:, 0, :, :] = zero

                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output + 1e-20), seg)
                            loss_G_joint2 += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                else:
                    output = inv_affine(output_affine[0], theta)
                    output = output.clone() + 1e-20
                    loss_dice = -torch.log(soft_dice_loss(output, target))
                    loss_ce = ce(torch.log(output), seg)
                    loss_G_joint2 = loss_dice + loss_ce
                    del output


                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint2.backward()
                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer2.step()

                # Save Losses for plotting later
                G_losses2.append(loss_G_joint2.item())
                torch.cuda.synchronize()
                step6 = time.time()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del seg

            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)
        temp_meanG2 = np.mean(G_losses2)
        mloss2.append(temp_meanG2)
        if epoch < 50:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
                % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))
        else:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G2 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G2: %.4f'
                % (G_losses2[0], loss_G_joint2.item(), temp_meanG2))

        del G_losses1
        del G_losses2

        print('starting validation...')
        net1.eval()
        net2.eval()

        dices_stn = []
        dices = []
        dices_s1 = []
        if channel_dim>2:
            dices_s2 = []
        if channel_dim>3:
            dices_s3 = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            val_target = val_target.to(device)
            image_crop, theta_crop = referencef(val_image, ensemble(val_target, channel_dim).squeeze(1))
            with torch.no_grad():
                x_affine, theta = net1(val_image)
                output_affine= net2(x_affine)
                pred = inv_affine(output_affine[0], theta)
                #pred = output_affine[0]

                zero = pred[:, 0, :, :]
                zero[zero == 0] = 1
                pred[:, 0, :, :] = zero

            if epoch < 50:
                loss_stn = 1 - (L1(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop)))
                #loss_stn = 1 - (torch.sqrt(L2(theta, theta_crop)))
                dices_stn.append(loss_stn.item())
            else:
                val_dice = dice_loss(pred, val_target)
                dices.append(val_dice.item())
                val_s1 = dice_loss_val(pred[6:12, 1, :, :], val_target[6:12, 1, :, :])
                dices_s1.append(val_s1.item())
                if channel_dim > 2:
                    val_s2 = dice_loss_val(pred[:, 2, :, :], val_target[:, 2, :, :])
                    dices_s2.append(val_s2.item())
                if channel_dim > 3:
                    val_s3 = dice_loss_val(pred[:, 3, :, :], val_target[:, 3, :, :])
                    dices_s3.append(val_s3.item())

        if epoch < 50:
            mdicestn = np.mean(dices_stn, axis=0)
            valdicesstn.append(mdicestn.item())
            print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
                  % (mdicestn))
            if mdicestn > best_dicestn:
                best_dicestn = mdicestn
                torch.save(net1.state_dict(), data_results + '/net_1.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dicestn = [mdicestn]
                np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

            del dices_stn


        else:

            mdice = np.mean(dices, axis=0)
            valdices.append(mdice.item())
            s1dice = np.mean(dices_s1, axis=0)
            vals1.append(s1dice.item())
            if channel_dim > 2:
                s2dice = np.mean(dices_s2, axis=0)
                vals2.append(s2dice.item())
            if channel_dim > 3:
                s3dice = np.mean(dices_s3, axis=0)
                vals3.append(s3dice.item())

            print('VALIDATION (BEST AT 1) -->Average of batches of Dice loss: %.4f'
                  % (mdice))
            if channel_dim == 2:
                print('Structure 1: %.4f'
                      % (s1dice))
            if channel_dim == 3:
                print('Structure 1: %.4f, Structure 2: %.4f'
                      % (s1dice, s2dice))
            if channel_dim == 4:
                print('Structure 1: %.4f, Structure 2: %.4f, Structure 3: %.4f'
                      % (s1dice, s2dice, s3dice))

            if mdice > best_dice:
                best_dice = mdice
                torch.save(net2.state_dict(), data_results + '/net_2.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dice = [mdice]
                np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
                no_improvement = 0
            else:
                no_improvement += 1

            del dices

        print('THETA: ', theta[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            #val_target_ = ensemble(val_target, channel_dim)
            #pred_ = ensemble(pred, channel_dim)
            #output_affine_ = ensemble(output_affine[0], channel_dim)
            val_target_ = torch.argmax(val_target, dim=1, keepdim=True)
            pred_ = torch.argmax(pred, dim=1, keepdim=True)
            output_affine_ = torch.argmax(output_affine[0],dim=1, keepdim=True)
            ncols = 6  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                prd_affine = np_to_img(output_affine_[q, 0, :, :].data.cpu().numpy(), 'target')
                prd = np_to_img(pred_[q, 0, :, :].data.cpu().numpy(), 'target')
                tgt = np_to_img(val_target_[q, 0, :, :].data.cpu().numpy(), 'target')
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Cropped Predicted")
                    axes[2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Predicted")
                    axes[3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[4].set_title("Reference")
                    axes[4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[5].set_title("Reference Crop")
                    axes[5].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Cropped Predicted")
                    axes[q, 2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 3].set_title("Predicted")
                    axes[q, 3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 4].set_title("Reference")
                    axes[q, 4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 5].set_title("Reference Crop")
                    axes[q, 5].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            val_step += 1

            del pred_
            del val_target_
            del output_affine_

        del val_image
        del val_target

        # poly learning rate policy
        if epoch>=50:
            # poly learning rate policy
            for g in optimizer2.param_groups:
                g['lr'] = init_lr * ((1 - (epoch / num_epochs)) ** 0.9)
            # lr = init_lr * ((1 - (epoch/num_epochs))**0.9)
            # del optimizer2
            # optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
            # if learning == 'apex':
            #    net2, optimizer2 = amp.initialize(net2, optimizer2, opt_level="O1")

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(mloss2, label="Training Loss (best at 0)")
    plt.plot(valdices, label="Validation Loss (best at 1)")
    plt.plot(vals1, label="Val Structure 1")
    if channel_dim > 2:
        plt.plot(vals2, label="Val Structure 2")
    if channel_dim > 3:
        plt.plot(vals3, label="Val Structure 3")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_unet_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training UNet Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 1)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_stnpose_nnunet2d(learning, path, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, optimizers, init_lr,channel_dim, train_loader, val_loader, patch_size, data_results, massimo, minimo, supervision, device,  asnn, mode=None, log=False):
    net1 = nets[0]
    net2 = nets[1]
    optimizer1 = optimizers[0]
    optimizer2 = optimizers[1]
    num_p = sum(p.numel() for p in net1.parameters() if p.requires_grad)
    print('Number of model parameters stn:', num_p)
    num_p2 = sum(p.numel() for p in net2.parameters() if p.requires_grad)
    print('Number of model parameters unet:', num_p2)
    print('Dice loss log: ', log)
    if learning == 'autocast':
        import torch.cuda.amp as amp2
        scaler = amp2.GradScaler()
    # Lists to keep track of progress
    valdices = []
    valdicesstn = []
    vals1 = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    if channel_dim > 2:
        vals2 = []
    if channel_dim > 3:
        vals3 = []
    mloss1 = []
    mloss2 = []
    no_improvement = 0


    print('Upload reference image for pose and shape...')
    ref_path = path + '/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/NECKER_071_1601_311.npz'
    ref_img = np.load(ref_path)['arr_0'][0, :, :].astype(np.float32)
    ref_image = resize(ref_img, (patch_size), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
    ref = keep_largest(ref_image)


    torch.cuda.synchronize()
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
        G_losses2 = []
        torch.cuda.synchronize()
        step = time.time()

        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
                # print(i)
                torch.cuda.synchronize()
                step1 = time.time()
                if mode == 'debug':
                    if i == 0:
                        print("Load images (s):", step1 - step)
                    else:
                        print("Load images (s):", step1 - step6)

                image0, target0 = data0
                image1, target1 = data1
                image = torch.cat((image0, image1), 0)
                target = torch.cat((target0, target1), 0)
                rand = torch.randperm(image.size()[0])
                image = image[rand]
                target = target[rand]
                image = image.type(th.float).to(device)
                target = target.type(th.float).to(device)

                white = torch.ones((image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
                black = torch.zeros((image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
                pose = image.clone()
                pose = torch.where(pose >= (image[0].min() + 0.01),
                                   torch.where(pose < (image[0].min() + 0.01), pose, white), black)

                torch.cuda.synchronize()
                step2 = time.time()
                if mode=='debug':
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                if epoch<1000:

                    net1.train()
                    net2.eval()

                    optimizer1.zero_grad()

                    reference = th.as_tensor(ref, dtype=th.float).unsqueeze(0).repeat(image.size()[0], 1, 1, 1).to(
                        "cuda").to(device)

                    x_affine, theta = net1(torch.cat((image,pose), dim=1))

                    torch.cuda.synchronize()
                    step3 = time.clock()

                    if mode == 'debug':
                        print("forward step (s): ", step3 - step2)
                        print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                        print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')


                    loss_G_joint1 = soft_dice_loss_old(x_affine, reference)

                    torch.cuda.synchronize()
                    step4 = time.clock()
                    if mode == 'debug':
                        print("Loss step (s): ", step4 - step3)
                        print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                        print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                    loss_G_joint1.backward()
                    torch.cuda.synchronize()
                    step5 = time.clock()
                    if mode == 'debug':
                        print("backward step (s): ", step5 - step4)
                        print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                        print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                    optimizer1.step()

                    # Save Losses for plotting later
                    G_losses1.append(loss_G_joint1.item())
                    torch.cuda.synchronize()
                    step6 = time.clock()
                    if mode == 'debug':
                        print("optimizer step (s): ", step6 - step5)

                    del reference


                    #---------END TRAIN FIRST NET---------START TRAIN SECOND NET------
                else:

                    net2.train()
                    net1.eval()

                    if epoch==1000 and i==0:
                        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

                    seg = torch.argmax(target, dim=1)
                    optimizer2.zero_grad()
                    with torch.no_grad():
                        x_affine, theta = net1(torch.cat((image, pose), dim=1))
                    if learning == 'autocast':
                        with amp2.autocast():
                            output_affine = net2(x_affine[:,0,:,:].unsqueeze(1))

                            torch.cuda.synchronize()
                            step3 = time.clock()

                            if mode == 'debug':
                                print("forward step (s): ", step3 - step2)
                                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                        if supervision == 'deep':
                            w = (32 / 63)
                            for p in range(6):

                                output = inv_affine(output_affine[p], theta)

                                zero = output[:, 0, :, :]
                                zero[zero == 0] = 1
                                output[:, 0, :, :] = zero
                                if log:
                                    loss_dice = -torch.log(soft_dice_loss(output, target))
                                else:
                                    loss_dice = soft_dice_loss_old(output, target)
                                loss_ce = ce(torch.log(output + 1e-20), seg)
                                if p == 0:
                                    loss_G_joint2 = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                                else:
                                    loss_G_joint2 += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                        else:
                            output = inv_affine(output_affine[0], theta)
                            output = output.clone() + 1e-20
                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output), seg)
                            loss_G_joint2 = loss_dice + loss_ce
                            del output

                            torch.cuda.synchronize()
                            step4 = time.clock()
                            if mode == 'debug':
                                print("Loss step (s): ", step4 - step3)
                                print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                                print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                        with amp2.autocast():
                            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                            # Backward passes under autocast are not recommended.
                            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                            scaler.scale(loss_G_joint2).backward()

                            # scaler.step() first unscales the gradients of the optimizer's assigned params.
                            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                            # otherwise, optimizer.step() is skipped.
                            scaler.step(optimizer2)

                            # Updates the scale for next iteration.
                            scaler.update()

                            torch.cuda.synchronize()
                            step5 = time.time()
                            if mode == 'debug':
                                print("backward step (s): ", step5 - step4)
                                print(
                                    f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                                print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                    else:
                        output_affine = net2(x_affine[:, 0, :, :].unsqueeze(1))

                        torch.cuda.synchronize()
                        step3 = time.clock()

                        if mode == 'debug':
                            print("forward step (s): ", step3 - step2)
                            print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                            print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                        if supervision == 'deep':
                            w = (32 / 63)
                            for p in range(6):

                                output = inv_affine(output_affine[p], theta)

                                zero = output[:, 0, :, :]
                                zero[zero == 0] = 1
                                output[:, 0, :, :] = zero
                                if log:
                                    loss_dice = -torch.log(soft_dice_loss(output, target))
                                else:
                                    loss_dice = soft_dice_loss_old(output, target)
                                loss_ce = ce(torch.log(output + 1e-20), seg)
                                if p == 0:
                                    loss_G_joint2 = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                                else:
                                    loss_G_joint2 += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                        else:
                            output = inv_affine(output_affine[0], theta)
                            output = output.clone() + 1e-20
                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output), seg)
                            loss_G_joint2 = loss_dice + loss_ce
                            del output

                        loss_G_joint2.backward()

                        torch.cuda.synchronize()
                        step5 = time.clock()
                        if mode == 'debug':
                            print("backward step (s): ", step5 - step4)
                            print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                            print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                        optimizer2.step()

                    # Save Losses for plotting later
                    G_losses2.append(loss_G_joint2.item())
                    torch.cuda.synchronize()
                    step6 = time.clock()
                    if mode == 'debug':
                        print("optimizer step (s): ", step6 - step5)

                    del seg, output


                del image, image0, image1
                del target, target0, target1
                del pose, white, black

                if asnn:
                    if (i + 1) == 250:
                        break

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step)/60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)
        temp_meanG2 = np.mean(G_losses2)
        mloss2.append(temp_meanG2)
        if epoch<1000:
            print('[%d/%d]\t'
                    % ((epoch + 1), num_epochs))
            print('Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
                    % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))
        else:
            print('[%d/%d]\t'
                    % ((epoch + 1), num_epochs))
            print('Loss_G2 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G2: %.4f'
                % (G_losses2[0], loss_G_joint2.item(), temp_meanG2))

        del G_losses1
        del G_losses2

        print('starting validation...')
        net1.eval()
        net2.eval()

        dices_stn=[]
        dices = []
        dices_s1 = []
        if channel_dim>2:
            dices_s2 = []
        if channel_dim>3:
            dices_s3 = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            val_target = val_target.type(th.float).to(device)

            reference = th.as_tensor(ref, dtype=th.float).unsqueeze(0).repeat(val_image.size()[0], 1, 1, 1).to("cuda").to(device)
            white = torch.ones((val_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            black = torch.zeros((val_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            pose = val_image.clone()
            pose = torch.where(pose >= (val_image[0].min() + 0.01), torch.where(pose < (val_image[0].min() + 0.01), pose, white), black)
            with torch.no_grad():
                x_affine, theta = net1(torch.cat((val_image,pose), dim=1))
                output_affine = net2(x_affine[:,0,:,:].unsqueeze(1))
            pred = inv_affine(output_affine[0],theta)
            zero = pred[:, 0, :, :]
            zero[zero == 0] = 1
            pred[:, 0, :, :] = zero


            if epoch<1000:
                loss_stn=dice_loss_val(x_affine[:,1,:,:],reference[:,0,:,:])
                dices_stn.append(loss_stn.item())
            else:
                val_dice = dice_loss(pred, val_target)
                dices.append(val_dice.item())
                val_s1 = dice_loss_val(pred[6:12, 1, :, :], val_target[6:12, 1, :, :])
                dices_s1.append(val_s1.item())
                if channel_dim > 2:
                    val_s2 = dice_loss_val(pred[6:12, 2, :, :], val_target[6:12, 2, :, :])
                    dices_s2.append(val_s2.item())
                if channel_dim > 3:
                    val_s3 = dice_loss_val(pred[6:12, 3, :, :], val_target[6:12, 3, :, :])
                    dices_s3.append(val_s3.item())

        if epoch<1000:
            mdicestn = np.mean(dices_stn, axis=0)
            valdicesstn.append(mdicestn.item())
            print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
                  % (mdicestn))
            if mdicestn > best_dicestn:
                best_dicestn = mdicestn
                torch.save(net1.state_dict(), data_results + '/net_1.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dicestn = [mdicestn]
                np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

            del dices_stn


        else:
            mdice = np.mean(dices, axis=0)
            valdices.append(mdice.item())
            s1dice = np.mean(dices_s1, axis=0)
            vals1.append(s1dice.item())
            if channel_dim > 2:
                s2dice = np.mean(dices_s2, axis=0)
                vals2.append(s2dice.item())
            if channel_dim > 3:
                s3dice = np.mean(dices_s3, axis=0)
                vals3.append(s3dice.item())

            print('VALIDATION (BEST AT 1) -->Average of batches of Dice loss: %.4f'
                  % (mdice))
            if channel_dim == 2:
                print('Structure 1: %.4f'
                      % (s1dice))
            if channel_dim == 3:
                print('Structure 1: %.4f, Structure 2: %.4f'
                      % (s1dice, s2dice))
            if channel_dim == 4:
                print('Structure 1: %.4f, Structure 2: %.4f, Structure 3: %.4f'
                      % (s1dice, s2dice, s3dice))

            if mdice > best_dice:
                best_dice = mdice
                torch.save(optimizer2.state_dict(), data_results + '/optimizer.pth')
                torch.save(net2.state_dict(), data_results + '/net_2.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dice = [mdice]
                np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
                no_improvement = 0
            else:
                no_improvement += 1

            del dices


        print('THETA: ', theta[0].data.cpu().numpy())


        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            val_target_ = torch.argmax(val_target, dim=1, keepdim=True)
            pred_ = torch.argmax(pred, dim=1, keepdim=True)
            output_affine_ = torch.argmax(output_affine[0], dim=1, keepdim=True)

            ncols = 6  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                prd_affine = np_to_img(output_affine_[q, 0, :, :].data.cpu().numpy(), 'target')
                prd = np_to_img(pred_[q, 0, :, :].data.cpu().numpy(), 'target')
                tgt = np_to_img(val_target_[q, 0, :, :].data.cpu().numpy(), 'target')
                ref_ = np_to_img(reference[0,0,:,:].data.cpu().numpy(), 'image')
                # 0.0.11 Store results
                if nrows == 1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Affine Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Affine Predicted")
                    axes[2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Predicted")
                    axes[3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[4].set_title("Reference")
                    axes[4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[5].set_title("Reference Pose")
                    axes[5].imshow(ref_, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Affine Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Affine Predicted")
                    axes[q, 2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 3].set_title("Predicted")
                    axes[q, 3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 4].set_title("Reference")
                    axes[q, 4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[q,5].set_title("Reference Pose")
                    axes[q,5].imshow(ref_, cmap='gray', vmin=0, vmax=255.)

            f.savefig(data_results + '/images_' + str(val_step*25) + '_val.png', bbox_inches='tight')
            val_step += 1

            del pred_
            del val_target_
            del output_affine_

        del val_image
        del val_target
        del reference


        # poly learning rate policy
        if epoch>=1000:
            # poly learning rate policy
            for g in optimizer2.param_groups:
                g['lr'] = init_lr * ((1 - ((epoch-1000) / (num_epochs-1000))) ** 0.9)
                print('new lr:', g['lr'])

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue


    elapsed = time.clock()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(mloss2, label="Training Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(data_results + '/train_unet_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("Validation Loss")
    plt.plot(valdices, label="Validation Loss (best at 1)")
    plt.plot(vals1, label="Val Structure 1")
    if channel_dim>2:
        plt.plot(vals2, label="Val Structure 2")
    if channel_dim>3:
        plt.plot(vals3, label="Val Structure 3")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(data_results + '/val_unet_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("STNpose - Training Loss")
    plt.plot(mloss1, label="Training UNet Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_stnpose_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("STNpose - Validation Loss")
    plt.plot(valdicesstn, label="Validation Loss (best at 1)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/val_stnpose_plot.png')

    torch.cuda.empty_cache()

def train_stnpose_stncrop_nnunet2d(learning, path, start_epoch, num_epochs, best_dice, best_dicestn1, best_dicestn2, val_step, early_stopping, nets, init_lr,channel_dim, train_loader, val_loader, patch_size, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    net2 = nets[1]
    net3 = nets[2]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    optimizer2 = optim.SGD(net2.parameters(), lr=init_lr)
    if start_epoch>=100:
        lr = init_lr * ((1 - ((start_epoch - 100) / num_epochs)) ** 0.9)
        optimizer3 = optim.SGD(net3.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        optimizer3 = optim.SGD(net3.parameters(), lr=init_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)

    # Lists to keep track of progress
    valdices = []
    valdicesstn1 = []
    valdicesstn2 = []
    vals1 = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    if channel_dim > 2:
        vals2 = []
        train_loader2 = train_loader[2]
        val_loader2 = val_loader[2]
    if channel_dim > 3:
        vals3 = []
        train_loader3 = train_loader[3]
        val_loader3 = val_loader[3]
    mloss1 = []
    mloss2 = []
    mloss3 = []
    no_improvement = 0

    print('Upload reference image for pose and shape...')
    ref_path = path + '/nnUNet_preprocessed/Task200_NECKER/nnUNetData_plans_v2.1_2D_stage0/Slices/NECKER_071_1601_311.npz'
    ref_img = np.load(ref_path)['arr_0'][0, :, :].astype(np.float32)
    ref_image = resize(ref_img, (patch_size), order=1, mode='constant', cval=ref_img.min(), anti_aliasing=False)
    ref = keep_largest(ref_image)

    torch.cuda.synchronize()
    start_time = time.time()
    # For each epoch
    for epoch in range(start_epoch,num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
        G_losses2 = []
        G_losses3 = []
        torch.cuda.synchronize()
        step = time.time()
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            # print(i)
            torch.cuda.synchronize()
            step1 = time.time()
            if mode == 'debug':
                if i == 0:
                    print("Load images (s):", step1 - step)
                else:
                    print("Load images (s):", step1 - step6)

            image0, target0 = data0
            image1, target1 = data1
            image = torch.cat((image0, image1), 0)
            target = torch.cat((target0, target1), 0)
            rand = torch.randperm(image.size()[0])
            image = image[rand]
            target = target[rand]
            image = image.to(device)
            target = target.to(device)


            white = torch.ones((image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            black = torch.zeros((image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            pose = image.clone()
            pose = torch.where(pose >= (image[0].min() + 0.01),
                               torch.where(pose < (image[0].min() + 0.01), pose, white), black)

            torch.cuda.synchronize()
            step2 = time.time()
            if mode == 'debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if epoch<50:
                if (i % 1000) == 0:
                    print('Iteration: ', i)

                net1.train()
                net2.eval()
                net3.eval()

                optimizer1.zero_grad()

                reference = th.as_tensor(ref, dtype=th.float).unsqueeze(0).repeat(image.size()[0], 1, 1, 1).to(
                    "cuda").to(device)

                x_affine, theta = net1(torch.cat((image, pose), dim=1))

                torch.cuda.synchronize()
                step3 = time.time()

                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint1 = soft_dice_loss_old(x_affine[:, 1, :, :].unsqueeze(1), reference)

                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint1.backward()
                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer1.step()

                # Save Losses for plotting later
                G_losses1.append(loss_G_joint1.item())
                torch.cuda.synchronize()
                step6 = time.clock()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

            elif epoch>=50 and epoch<100:
                if (i % 1000) == 0:
                    print('Iteration: ', i)

                net1.eval()
                net2.train()
                net3.eval()

                optimizer2.zero_grad()

                seg = torch.argmax(target, dim=1)
                if epoch == 50 and i==0:
                    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

                with torch.no_grad():
                    x_affine, theta_affine= net1(torch.cat((image,pose), dim=1))
                    seg_clone = seg.clone().unsqueeze(1).type(torch.FloatTensor).to(device)
                    grid_cropped = F.affine_grid(theta_affine, seg_clone.size(), align_corners=False)
                    seg_affine = F.grid_sample(seg_clone, grid_cropped,
                                               align_corners=False, mode='nearest')  # , padding_mode="border"

                image_crop, theta_crop_ref = referencef(x_affine[:,0,:,:].unsqueeze(1), seg_affine.squeeze(1))

                x_crop, theta_crop = net2(x_affine[:,0,:,:].unsqueeze(1))

                torch.cuda.synchronize()
                step3 = time.clock()

                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                #loss_G_joint2 = L1(x_crop, image_crop) + torch.sqrt(L2(theta_crop, theta_crop_ref))
                loss_G_joint2 = torch.sqrt(L2(theta_crop, theta_crop_ref))

                torch.cuda.synchronize()
                step4 = time.clock()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')


                loss_G_joint2.backward()
                torch.cuda.synchronize()
                step5 = time.clock()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer2.step()

                # Save Losses for plotting later
                G_losses2.append(loss_G_joint2.item())
                torch.cuda.synchronize()
                step6 = time.clock()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del seg

            #---------END TRAIN FIRST NET---------START TRAIN SECOND NET------
            else:
                if (i % 1000) == 0:
                    print('Iteration: ', i)

                net1.eval()
                net2.eval()
                net3.train()

                optimizer3.zero_grad()

                seg = torch.argmax(target, dim=1)
                if epoch == 100 and i==0:
                    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
                    net2.load_state_dict(torch.load(data_results + '/net_2.pth'))



                with torch.no_grad():
                    x_affine, theta = net1(torch.cat((image,pose), dim=1))
                    x_crop, theta_crop = net2(x_affine[:, 0, :, :].unsqueeze(1))

                output_crop = net3(x_crop)

                output_affine = []
                for i in range(6):
                    output = inv_affine(output_crop[i], theta_crop)
                    zero = output[:, 0, :, :]
                    zero[zero == 0] = 1
                    output[:, 0, :, :] = zero
                    output_affine.append(output)
                del output

                torch.cuda.synchronize()
                step3 = time.clock()

                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                if supervision == 'deep':
                    w = (32 / 63)
                    for p in range(6):
                        if p == 0:
                            output = inv_affine(output_affine[p], theta)

                            zero = output[:, 0, :, :]
                            zero[zero == 0] = 1
                            output[:, 0, :, :] = zero

                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output + 1e-20), seg)
                            loss_G_joint3 = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                        else:
                            output = inv_affine(output_affine[p], theta)

                            zero = output[:, 0, :, :]
                            zero[zero == 0] = 1
                            output[:, 0, :, :] = zero

                            loss_dice = -torch.log(soft_dice_loss(output, target))
                            loss_ce = ce(torch.log(output + 1e-20), seg)
                            loss_G_joint3 += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                else:
                    output = inv_affine(output_affine[0], theta)
                    output = output.clone() + 1e-20
                    loss_dice = -torch.log(soft_dice_loss(output, target))
                    loss_ce = ce(torch.log(output), seg)
                    loss_G_joint3 = loss_dice + loss_ce
                    del output

                torch.cuda.synchronize()
                step4 = time.clock()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint3.backward()

                torch.cuda.synchronize()
                step5 = time.clock()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer3.step()

                # Save Losses for plotting later
                G_losses3.append(loss_G_joint3.item())
                torch.cuda.synchronize()
                step6 = time.clock()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del seg

            del image
            del target
            del pose
            del white
            del black

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step)/60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)
        temp_meanG2 = np.mean(G_losses2)
        mloss2.append(temp_meanG2)
        temp_meanG3 = np.mean(G_losses3)
        mloss3.append(temp_meanG3)

        if epoch < 50:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
                % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))
        elif epoch>=50 and epoch<100:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G2 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G2: %.4f'
                % (G_losses2[0], loss_G_joint2.item(), temp_meanG2))
        else:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G2 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G2: %.4f'
                % (G_losses3[0], loss_G_joint3.item(), temp_meanG3))
        del G_losses1
        del G_losses2
        del G_losses3

        print('starting validation...')
        net1.eval()
        net2.eval()
        net3.eval()

        dices_stn1=[]
        dices_stn2 = []
        dices = []
        dices_s1 = []
        if channel_dim>2:
            dices_s2 = []
        if channel_dim>3:
            dices_s3 = []

        # for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            val_target = val_target.to(device)

            val_seg = torch.argmax(val_target, dim=1)
            reference = th.as_tensor(ref, dtype=th.float).unsqueeze(0).repeat(val_image.size()[0], 1, 1, 1).to(
                "cuda").to(device)
            white = torch.ones((val_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            black = torch.zeros((val_image.size()[0], 1, patch_size[0], patch_size[1]), requires_grad=True).to(device)
            pose = val_image.clone()
            pose = torch.where(pose >= (val_image[0].min() + 0.01),
                               torch.where(pose < (val_image[0].min() + 0.01), pose, white), black)

            with torch.no_grad():
                x_affine,theta = net1(torch.cat((val_image,pose), dim=1))
                x_crop,theta_crop = net2(x_affine[:,0,:,:].unsqueeze(1))
                output_crop = net3(x_crop)
            output_affine = inv_affine(output_crop[0],theta_crop)
            zero = output_affine[:, 0, :, :]
            zero[zero == 0] = 1
            output_affine[:, 0, :, :] = zero

            pred = inv_affine(output_affine, theta)
            zero = pred[:, 0, :, :]
            zero[zero == 0] = 1
            pred[:, 0, :, :] = zero

            #pred[pred < 0.5] = 0
            #pred[pred >= 0.5] = 1
            seg_clone = val_seg.clone().unsqueeze(1).type(torch.FloatTensor).to(device)
            grid_cropped = F.affine_grid(theta, seg_clone.size(), align_corners=False)
            seg_affine = F.grid_sample(seg_clone, grid_cropped,
                                       align_corners=False, mode='nearest')  # , padding_mode="border"
            image_crop, theta_crop_ref = referencef(x_affine[:,0,:,:].unsqueeze(1), seg_affine.squeeze(1))
            if epoch<50:
                loss_stn1=dice_loss_val(x_affine[:,1,:,:],reference[:,0,:,:])
                dices_stn1.append(loss_stn1.item())
            elif epoch>=50 and epoch<100:
                #loss_stn2 = 1 - (L1(x_crop, image_crop) + torch.sqrt(L2(theta_crop, theta_crop_ref)))
                loss_stn2 = 1 - (torch.sqrt(L2(theta_crop, theta_crop_ref)))
                dices_stn2.append(loss_stn2.item())
            else:
                val_dice = dice_loss(pred, val_target)
                dices.append(val_dice.item())
                val_s1 = dice_loss_val(pred[6:12, 1, :, :], val_target[6:12, 1, :, :])
                dices_s1.append(val_s1.item())
                if channel_dim > 2:
                    val_s2 = dice_loss_val(pred[:, 2, :, :], val_target[:, 2, :, :])
                    dices_s2.append(val_s2.item())
                if channel_dim > 3:
                    val_s3 = dice_loss_val(pred[:, 3, :, :], val_target[:, 3, :, :])
                    dices_s3.append(val_s3.item())

        if epoch<50:
            mdicestn1 = np.mean(dices_stn1, axis=0)
            valdicesstn1.append(mdicestn1.item())
            print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
                  % (mdicestn1))
            if mdicestn1 > best_dicestn1:
                best_dicestn1 = mdicestn1
                torch.save(net1.state_dict(), data_results + '/net_1.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dicestn = [mdicestn1]
                np.save(data_results + '/dice_valstn1.npy', np.asarray(total_dicestn, dtype=np.float32))

            del dices_stn1

        elif epoch>=50 and epoch<100:
            mdicestn2 = np.mean(dices_stn2, axis=0)
            valdicesstn2.append(mdicestn2.item())
            print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
                  % (mdicestn2))
            if mdicestn2 > best_dicestn2:
                best_dicestn2 = mdicestn2
                torch.save(net2.state_dict(), data_results + '/net_2.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dicestn = [mdicestn2]
                np.save(data_results + '/dice_valstn2.npy', np.asarray(total_dicestn, dtype=np.float32))

            del dices_stn2


        else:
            mdice = np.mean(dices, axis=0)
            valdices.append(mdice.item())
            s1dice = np.mean(dices_s1, axis=0)
            vals1.append(s1dice.item())
            if channel_dim > 2:
                s2dice = np.mean(dices_s2, axis=0)
                vals2.append(s2dice.item())
            if channel_dim > 3:
                s3dice = np.mean(dices_s3, axis=0)
                vals3.append(s3dice.item())

            print('VALIDATION (BEST AT 1) -->Average of batches of Dice loss: %.4f'
                  % (mdice))
            if channel_dim == 2:
                print('Structure 1: %.4f'
                      % (s1dice))
            if channel_dim == 3:
                print('Structure 1: %.4f, Structure 2: %.4f'
                      % (s1dice, s2dice))
            if channel_dim == 4:
                print('Structure 1: %.4f, Structure 2: %.4f, Structure 3: %.4f'
                      % (s1dice, s2dice, s3dice))

            if mdice > best_dice:
                    best_dice = mdice
                    torch.save(net3.state_dict(), data_results + '/net_3.pth')
                    print('IMPROVEMENT IN VALIDATION')
                    total_dice = [mdice]
                    np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
                    no_improvement = 0
            else:
                    no_improvement += 1

            del dices

        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA CROP: ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            #val_target_ = ensemble(val_target, channel_dim)
            #pred_ = ensemble(pred, channel_dim)
            #output_affine_ = ensemble(output_affine, channel_dim)
            #output_crop_ = ensemble(output_crop[0], channel_dim)

            val_target_ = torch.argmax(val_target, dim=1, keepdim=True)
            pred_ = torch.argmax(pred, dim=1, keepdim=True)
            output_affine_ = torch.argmax(output_affine, dim=1, keepdim=True)
            output_crop_ = torch.argmax(output_crop[0], dim=1, keepdim=True)

            ncols = 7  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(x_crop[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                prd_crop = np_to_img(output_crop_[q, 0, :, :].data.cpu().numpy(), 'target')
                prd_affine = np_to_img(output_affine_[q, 0, :, :].data.cpu().numpy(), 'target')
                prd = np_to_img(pred_[q, 0, :, :].data.cpu().numpy(), 'target')
                tgt = np_to_img(val_target_[q, 0, :, :].data.cpu().numpy(), 'target')
                # 0.0.11 Store results
                axes[q, 0].set_title("Original Test Image")
                axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[q, 1].set_title("Affine Test Image")
                axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 2].set_title("Crop Test Image")
                axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                axes[q, 3].set_title("Crop Predicted")
                axes[q, 3].imshow(prd_crop, cmap='gray', vmin=0, vmax=255.)
                axes[q, 4].set_title("Affine Predicted")
                axes[q, 4].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 5].set_title("Predicted")
                axes[q, 5].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                axes[q, 6].set_title("Reference")
                axes[q, 6].imshow(tgt, cmap='gray', vmin=0, vmax=255.)

            f.savefig(data_results + '/images_' + str(val_step*25) + '_val.png', bbox_inches='tight')
            val_step += 1

            del pred_
            del val_target_
            del output_affine_
            del output_crop_

        del val_image
        del val_target
        del val_seg
        del reference
        del pose

        # poly learning rate policy
        if epoch >= 100:
            # poly learning rate policy
            for g in optimizer3.param_groups:
                g['lr'] = init_lr * ((1 - (epoch / num_epochs)) ** 0.9)
            # lr = init_lr * ((1 - (epoch/num_epochs))**0.9)
            # del optimizer3
            # optimizer3 = optim.SGD(net3.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
            # if learning == 'apex':
            #    net3, optimizer3 = amp.initialize(net3, optimizer3, opt_level="O1")

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue


    elapsed = time.clock()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(mloss3, label="Training Loss (best at 0)")
    plt.plot(valdices, label="Validation Loss (best at 1)")
    plt.plot(vals1, label="Val Structure 1")
    if channel_dim>2:
        plt.plot(vals2, label="Val Structure 2")
    if channel_dim>3:
        plt.plot(vals3, label="Val Structure 3")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_unet_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("STNpose - Training and Validation Loss")
    plt.plot(mloss1, label="Training UNet Loss (best at 0)")
    plt.plot(valdicesstn1, label="Validation Loss (best at 1)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stnpose_plot.png')

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss2, label="Training UNet Loss (best at 0)")
    plt.plot(valdicesstn2, label="Validation Loss (best at 1)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_stncrop_3_nnunet2d(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    net2 = nets[1]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    if start_epoch>=50:
        lr = init_lr * ((1 - ((start_epoch - 50) / num_epochs)) ** 0.9)
        optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        optimizer2 = optim.SGD(net2.parameters(), lr=init_lr, weight_decay=3e-5, momentum=0.99, nesterov=True)

    valdices = []
    valdicesstn = []
    vals1 = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    if channel_dim > 2:
        vals2 = []
        train_loader2 = train_loader[2]
        val_loader2 = val_loader[2]
    if channel_dim > 3:
        vals3 = []
        train_loader3 = train_loader[3]
        val_loader3 = val_loader[3]
    mloss1 = []
    mloss2 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
        G_losses2 = []
        torch.cuda.synchronize()
        step = time.time()
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            # print(i)
            torch.cuda.synchronize()
            step1 = time.time()
            if mode=='debug':
                if i==0 :
                    print("Load images (s):", step1- step)
                else:
                    print("Load images (s):", step1 - step6)

            image0, target0 = data0
            image1, target1 = data1
            image = torch.cat((image0, image1), 0)
            target = torch.cat((target0, target1), 0)
            rand = torch.randperm(image.size()[0])
            image = image[rand]
            target = target[rand]
            image = image.to(device)
            target = target.to(device)

            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if epoch < 50:
                net1.train()
                net2.eval()

                optimizer1.zero_grad()

                image_crop, theta_crop = referencef2(image, ensemble(target, channel_dim).squeeze(1))

                x_affine, theta = net1(image)

                torch.cuda.synchronize()
                step3 = time.time()
                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                #loss_G_joint1 = L2(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop))
                loss_G_joint1 = torch.sqrt(L2(theta, theta_crop))

                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint1.backward()
                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer1.step()


                # Save Losses for plotting later
                G_losses1.append(loss_G_joint1.item())
                torch.cuda.synchronize()
                step6 = time.time()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del theta_crop
                del image_crop

                # ---------END TRAIN FIRST NET---------START TRAIN SECOND NET------
            else:
                if (i % 1000) == 0:
                    print('Iteration: ', i)
                net2.train()
                net1.eval()
                if epoch == 50 and i==0:
                    net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
                #seg = ensemble(target, channel_dim).squeeze(1)
                seg = torch.argmax(target, dim=1)
                optimizer2.zero_grad()
                with torch.no_grad():
                    x_affine, theta = net1(image)
                output = net2(x_affine)

                torch.cuda.synchronize()
                step3 = time.time()
                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                if supervision == 'deep':
                    w = (32 / 63)
                    for p in range(6):
                        if p == 0:
                            loss_dice = -torch.log(soft_dice_loss(output[p], target))
                            loss_ce = ce(torch.log(output[p] + 1e-20), seg)
                            loss_G_joint2 = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                        else:
                            loss_dice = -torch.log(soft_dice_loss(output[p], target))
                            loss_ce = ce(torch.log(output[p] + 1e-20), seg)
                            loss_G_joint2 += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                else:
                    loss_dice = -torch.log(soft_dice_loss(output[0], target))
                    loss_ce = ce(torch.log(output[0]+ 1e-20), seg)
                    loss_G_joint2 = loss_dice + loss_ce
                    del output


                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint2.backward()
                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer2.step()

                # Save Losses for plotting later
                G_losses2.append(loss_G_joint2.item())
                torch.cuda.synchronize()
                step6 = time.time()
                if mode == 'debug':
                    print("optimizer step (s): ", step6 - step5)

                del seg

            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)
        temp_meanG2 = np.mean(G_losses2)
        mloss2.append(temp_meanG2)
        if epoch < 50:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
                % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))
        else:
            print('[%d/%d]\t'
                  % ((epoch + 1), num_epochs))
            print(
                'Loss_G2 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G2: %.4f'
                % (G_losses2[0], loss_G_joint2.item(), temp_meanG2))

        del G_losses1
        del G_losses2

        print('starting validation...')
        net1.eval()
        net2.eval()

        dices_stn = []
        dices = []
        dices_s1 = []
        if channel_dim>2:
            dices_s2 = []
        if channel_dim>3:
            dices_s3 = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            val_target = val_target.to(device)
            image_crop, theta_crop = referencef2(val_image, ensemble(val_target, channel_dim).squeeze(1))
            with torch.no_grad():
                x_affine, theta = net1(val_image)
                output_affine= net2(x_affine)
                pred = output_affine[0]

            if epoch < 50:
                #loss_stn = 1 - (L2(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop)))
                loss_stn = 1 - (torch.sqrt(L2(theta, theta_crop)))
                dices_stn.append(loss_stn.item())
            else:
                val_dice = dice_loss(pred, val_target)
                dices.append(val_dice.item())
                val_s1 = dice_loss_val(pred[6:12, 1, :, :], val_target[6:12, 1, :, :])
                dices_s1.append(val_s1.item())
                if channel_dim > 2:
                    val_s2 = dice_loss_val(pred[:, 2, :, :], val_target[:, 2, :, :])
                    dices_s2.append(val_s2.item())
                if channel_dim > 3:
                    val_s3 = dice_loss_val(pred[:, 3, :, :], val_target[:, 3, :, :])
                    dices_s3.append(val_s3.item())

        if epoch < 50:
            mdicestn = np.mean(dices_stn, axis=0)
            valdicesstn.append(mdicestn.item())
            print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
                  % (mdicestn))
            if mdicestn > best_dicestn:
                best_dicestn = mdicestn
                torch.save(net1.state_dict(), data_results + '/net_1.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dicestn = [mdicestn]
                np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

            del dices_stn


        else:

            mdice = np.mean(dices, axis=0)
            valdices.append(mdice.item())
            s1dice = np.mean(dices_s1, axis=0)
            vals1.append(s1dice.item())
            if channel_dim > 2:
                s2dice = np.mean(dices_s2, axis=0)
                vals2.append(s2dice.item())
            if channel_dim > 3:
                s3dice = np.mean(dices_s3, axis=0)
                vals3.append(s3dice.item())

            print('VALIDATION (BEST AT 1) -->Average of batches of Dice loss: %.4f'
                  % (mdice))
            if channel_dim == 2:
                print('Structure 1: %.4f'
                      % (s1dice))
            if channel_dim == 3:
                print('Structure 1: %.4f, Structure 2: %.4f'
                      % (s1dice, s2dice))
            if channel_dim == 4:
                print('Structure 1: %.4f, Structure 2: %.4f, Structure 3: %.4f'
                      % (s1dice, s2dice, s3dice))

            if mdice > best_dice:
                best_dice = mdice
                torch.save(net2.state_dict(), data_results + '/net_2.pth')
                print('IMPROVEMENT IN VALIDATION')
                total_dice = [mdice]
                np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
                no_improvement = 0
            else:
                no_improvement += 1

            del dices

        print('THETA: ', theta[-1].data.cpu().numpy())
        print('THETA TRUE: ', theta_crop[-1].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            #val_target_ = ensemble(val_target, channel_dim)
            #pred_ = ensemble(pred, channel_dim)
            #output_affine_ = ensemble(output_affine[0], channel_dim)
            val_target_ = torch.argmax(val_target, dim=1, keepdim=True)
            pred_ = torch.argmax(pred, dim=1, keepdim=True)
            output_affine_ = torch.argmax(output_affine[0],dim=1, keepdim=True)
            ncols = 6  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                prd_affine = np_to_img(output_affine_[q, 0, :, :].data.cpu().numpy(), 'target')
                prd = np_to_img(pred_[q, 0, :, :].data.cpu().numpy(), 'target')
                tgt = np_to_img(val_target_[q, 0, :, :].data.cpu().numpy(), 'target')
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Cropped Predicted")
                    axes[2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[3].set_title("Predicted")
                    axes[3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[4].set_title("Reference")
                    axes[4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[5].set_title("Reference Crop")
                    axes[5].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Cropped Predicted")
                    axes[q, 2].imshow(prd_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 3].set_title("Predicted")
                    axes[q, 3].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 4].set_title("Reference")
                    axes[q, 4].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 5].set_title("Reference Crop")
                    axes[q, 5].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            val_step += 1

            del pred_
            del val_target_
            del output_affine_

        del val_image
        del val_target

        # poly learning rate policy
        if epoch>=50:
            # poly learning rate policy
            for g in optimizer2.param_groups:
                g['lr'] = init_lr * ((1 - (epoch / num_epochs)) ** 0.9)
            # lr = init_lr * ((1 - (epoch/num_epochs))**0.9)
            # del optimizer2
            # optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
            # if learning == 'apex':
            #    net2, optimizer2 = amp.initialize(net2, optimizer2, opt_level="O1")

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    torch.cuda.empty_cache()

