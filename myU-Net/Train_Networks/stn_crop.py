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
from utils.utils import referencef3, inv_affine, keep_largest, referencef2, referencef, keep_largest_mask_torch, referencef3D
from utils.figures import ensemble, np_to_img
from utils.losses import L1, L2, ce, soft_dice_loss, dice_loss, dice_loss_val, soft_dice_loss_old

def train_stncrop3d(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)

    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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

            if (i % 1000) == 0:
                print('Iteration: ', i)
            net1.train()

            optimizer1.zero_grad()

            image_crop, theta_crop = referencef3(image, ensemble(target, channel_dim).squeeze(1))

            x_affine, theta = net1(image)

            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            loss_G_joint1 = L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop))

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
                
            if (epoch%25==0) or (epoch == (num_epochs - 1)):
            	ncols = 3  # number of columns in final grid of images
            	nrows = image.size()[0]  # looking at all images takes some time
            	f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            	for axis in axes.flatten():
            		axis.set_axis_off()
            		axis.set_aspect('equal')
            	for q in range(nrows):
            		img = np_to_img(image[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
            		img_crop = np_to_img(image_crop[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
            		img_affine = np_to_img(x_affine[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
            		# 0.0.11 Store results
            		if nrows==1:
            			axes[0].set_title("Original Test Image")
            			axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[1].set_title("Cropped Test Image")
            			axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[2].set_title("Reference Crop")
            			axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            		else:
            			axes[q, 0].set_title("Original Test Image")
            			axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 1].set_title("Cropped Test Image")
            			axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 2].set_title("Reference Crop")
            			axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            	f.savefig(data_results + '/images_train_view1_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            
            	ncols = 3  # number of columns in final grid of images
            	nrows = image.size()[0]  # looking at all images takes some time
            	f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            	for axis in axes.flatten():
            		axis.set_axis_off()
            		axis.set_aspect('equal')
            	for q in range(nrows):
            		img = np_to_img(image[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
            		img_crop = np_to_img(image_crop[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
            		img_affine = np_to_img(x_affine[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
            		# 0.0.11 Store results
            		if nrows==1:
            			axes[0].set_title("Original Test Image")
            			axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[1].set_title("Cropped Test Image")
            			axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[2].set_title("Reference Crop")
            			axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            		else:
            			axes[q, 0].set_title("Original Test Image")
            			axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 1].set_title("Cropped Test Image")
            			axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 2].set_title("Reference Crop")
            			axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            	f.savefig(data_results + '/images_train_view2_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            
            	ncols = 3  # number of columns in final grid of images
            	nrows = image.size()[0]  # looking at all images takes some time
            	f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            	for axis in axes.flatten():
            		axis.set_axis_off()
            		axis.set_aspect('equal')
            	for q in range(nrows):
            		img = np_to_img(image[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
            		img_crop = np_to_img(image_crop[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
            		img_affine = np_to_img(x_affine[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
            		# 0.0.11 Store results
            		if nrows==1:
            			axes[0].set_title("Original Test Image")
            			axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[1].set_title("Cropped Test Image")
            			axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[2].set_title("Reference Crop")
            			axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            		else:
            			axes[q, 0].set_title("Original Test Image")
            			axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 1].set_title("Cropped Test Image")
            			axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
            			axes[q, 2].set_title("Reference Crop")
            			axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            	f.savefig(data_results + '/images_train_view3_' + str(val_step * 25) + '_val.png', bbox_inches='tight')


            del theta_crop
            del image_crop
            del image
            del target
            
            

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            val_target = val_target.to(device)
            image_crop, theta_crop = referencef3(val_image, ensemble(val_target, channel_dim).squeeze(1))
            with torch.no_grad():
                x_affine, theta = net1(val_image)

            loss_stn = 1 - (L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop)))
            dices_stn.append(loss_stn.item())

        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))
        else:
            torch.save(net1.state_dict(), data_results + '/net_train_1.pth')       
        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real : ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, 64, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_view1_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, 64, :].data.cpu().numpy(), 'image',massimo, minimo)
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_view2_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :, 64].data.cpu().numpy(), 'image',massimo, minimo)
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_view3_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            
            val_step += 1


        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_stnposecrop3d(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[1]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net.eval()
    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            target_old = torch.cat((target0, target1), 0)
            rand = torch.randperm(image.size()[0])
            image = image[rand]
            target_old = target_old[rand]
            image = image.type(th.float).to(device)
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            net1.train()

            optimizer1.zero_grad()
            pose = keep_largest_mask_torch(image)
            target_old = target_old.type(th.float).to(device)
            with torch.no_grad():
                imagepso, theta1 = net(torch.cat((image, pose), dim=1))
                imageps = imagepso[:, :1, :, :, :]
                grid = F.affine_grid(theta1, target_old.size(), align_corners=False)
                target = F.grid_sample(target_old, grid, align_corners=False, mode='nearest',
                                           padding_mode='zeros')

            image_crop, theta_crop = referencef3D(imageps, ensemble(target, channel_dim).squeeze(1))

            x_affine, theta = net1(image)

            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            loss_G_joint1 = L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop))
            print(L1(x_affine, image_crop).item())
            print(10*torch.sqrt(L2(theta, theta_crop)).item())
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

            grid_cropped = F.affine_grid(theta_crop, target.size(), align_corners=False)
            v_crop = F.grid_sample(target.type(th.float), grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            if (((epoch % 10) == 0) or (epoch == (num_epochs - 1))) and i==0:
                ncols = 4  # number of columns in final grid of images
                nrows = 3  # looking at all images takes some time
                v_c = ensemble(v_crop, channel_dim)
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                img0 = np_to_img(imageps[0, 0, 64, :, :].data.cpu().numpy(), 'image')
                img_pose0 = np_to_img(image_crop[0, 0, 64, :, :].data.cpu().numpy(), 'image')
                v_c0 = np_to_img(v_c[0, 0, 64, :, :].data.cpu().numpy(), 'target')
                img_affine0 = np_to_img(x_affine[0, 0, 64, :, :].data.cpu().numpy(), 'image')

                img1 = np_to_img(imageps[0, 0, :, 64, :].data.cpu().numpy(), 'image')
                img_pose1 = np_to_img(image_crop[0, 0, :, 64, :].data.cpu().numpy(), 'image')
                v_c1 = np_to_img(v_c[0, 0, :, 64, :].data.cpu().numpy(), 'target')
                img_affine1 = np_to_img(x_affine[0, 0, :, 64, :].data.cpu().numpy(), 'image')

                img2 = np_to_img(imageps[0, 0, :, :, 64].data.cpu().numpy(), 'image')
                img_pose2 = np_to_img(image_crop[0, 0, :, :, 64].data.cpu().numpy(), 'image')
                v_c2 = np_to_img(v_c[0, 0, :, :, 64].data.cpu().numpy(), 'target')
                img_affine2 = np_to_img(x_affine[0, 0, :, :, 64].data.cpu().numpy(), 'image')

                # 0.0.11 Store results
                axes[0, 0].set_title("Original Test Image")
                axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
                axes[0, 1].set_title("Target")
                axes[0, 1].imshow(img_pose0, cmap='gray', vmin=0, vmax=255.)
                axes[0, 2].set_title("Pred")
                axes[0, 2].imshow(v_c0, cmap='gray', vmin=0, vmax=255.)
                axes[0, 3].set_title("Pred")
                axes[0, 3].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)

                axes[1, 0].set_title("Original Test Image")
                axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[1, 1].set_title("Target")
                axes[1, 1].imshow(img_pose1, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[1, 2].set_title("Pred")
                axes[1, 2].imshow(v_c1, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[1, 3].set_title("Pred")
                axes[1, 3].imshow(img_affine1, cmap='gray', vmin=0, vmax=255., origin='lower')

                axes[2, 0].set_title("Original Test Image")
                axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[2, 1].set_title("Target")
                axes[2, 1].imshow(img_pose2, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[2, 2].set_title("Pred")
                axes[2, 2].imshow(v_c2, cmap='gray', vmin=0, vmax=255., origin='lower')
                axes[2, 3].set_title("Pred")
                axes[2, 3].imshow(img_affine2, cmap='gray', vmin=0, vmax=255., origin='lower')

                f.savefig(data_results + '/images_' + str(epoch) + '_train.png', bbox_inches='tight')
                # 0.0.11 Store results

            del theta_crop
            del image_crop
            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target_old = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            pose = keep_largest_mask_torch(val_image)
            val_target_old = val_target_old.type(th.float).to(device)
            with torch.no_grad():
                val_imagepso, theta1 = net(torch.cat((val_image, pose), dim=1))
                val_imageps = val_imagepso[:, :1, :, :, :]
                grid = F.affine_grid(theta1, val_target_old.size(), align_corners=False)
                val_target = F.grid_sample(val_target_old, grid, align_corners=False, mode='nearest',
                                           padding_mode='zeros')
                image_crop, theta_crop = referencef3D(val_image, ensemble(val_target, channel_dim).squeeze(1))
                x_affine, theta = net1(val_imageps)

            grid_cropped = F.affine_grid(theta_crop, val_target.size(), align_corners=False)
            v_crop = F.grid_sample(val_target.type(th.float), grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            loss_stn = 2 - (L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(10*torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())

        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(2 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn

        print('THETA TRUE: ', theta_crop[0].data.cpu().numpy())
        print('THETA: ', theta[0].data.cpu().numpy())

        if ((val_step * 10) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 4  # number of columns in final grid of images
            nrows = 3  # looking at all images takes some time
            v_c = ensemble(v_crop, channel_dim)
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            img0 = np_to_img(val_imageps[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_pose0 = np_to_img(image_crop[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            v_c0 = np_to_img(v_c[0, 0, 64, :, :].data.cpu().numpy(), 'target')
            img_affine0 = np_to_img(x_affine[0, 0, 64, :, :].data.cpu().numpy(), 'image')

            img1 = np_to_img(val_imageps[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_pose1 = np_to_img(image_crop[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            v_c1 = np_to_img(v_c[0, 0, :, 64, :].data.cpu().numpy(), 'target')
            img_affine1 = np_to_img(x_affine[0, 0, :, 64, :].data.cpu().numpy(), 'image')

            img2 = np_to_img(val_imageps[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_pose2 = np_to_img(image_crop[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            v_c2 = np_to_img(v_c[0, 0, :, :, 64].data.cpu().numpy(), 'target')
            img_affine2 = np_to_img(x_affine[0, 0, :, :, 64].data.cpu().numpy(), 'image')

            # 0.0.11 Store results
            axes[0, 0].set_title("Original Test Image")
            axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 1].set_title("Target")
            axes[0, 1].imshow(img_pose0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 2].set_title("Pred")
            axes[0, 2].imshow(v_c0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 3].set_title("Pred")
            axes[0, 3].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)

            axes[1, 0].set_title("Original Test Image")
            axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 1].set_title("Target")
            axes[1, 1].imshow(img_pose1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 2].set_title("Pred")
            axes[1, 2].imshow(v_c1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 3].set_title("Pred")
            axes[1, 3].imshow(img_affine1, cmap='gray', vmin=0, vmax=255., origin='lower')

            axes[2, 0].set_title("Original Test Image")
            axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 1].set_title("Target")
            axes[2, 1].imshow(img_pose2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 2].set_title("Pred")
            axes[2, 2].imshow(v_c2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 3].set_title("Pred")
            axes[2, 3].imshow(img_affine2, cmap='gray', vmin=0, vmax=255., origin='lower')

            f.savefig(data_results + '/images_' + str(epoch) + '_val.png', bbox_inches='tight')
            # 0.0.11 Store results
            val_step += 1


        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_stncrop(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)

    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            image = image.type(th.float).to(device)
            target = target.to(device)
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            print(image.size(), target.size())
            net1.train()

            optimizer1.zero_grad()

            image_crop, theta_crop = referencef(image, ensemble(target, channel_dim).squeeze(1))

            x_affine, theta = net1(image)

            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            loss_G_joint1 = L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop))
            print(L1(x_affine, image_crop).item())
            print(10*torch.sqrt(L2(theta, theta_crop)).item())
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

            if i==0:
                ncols = 3  # number of columns in final grid of images
                nrows = image.size()[0]  # looking at all images takes some time
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                for q in range(nrows):
                    img = np_to_img(image[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    # 0.0.11 Store results
                    if nrows == 1:
                        axes[0].set_title("Original Test Image")
                        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[1].set_title("Cropped Test Image")
                        axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[2].set_title("Reference Crop")
                        axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    else:
                        axes[q, 0].set_title("Original Test Image")
                        axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 1].set_title("Cropped Test Image")
                        axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 2].set_title("Reference Crop")
                        axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)

                f.savefig(data_results + '/images_' + str(epoch) + '_train.png', bbox_inches='tight')

            del theta_crop
            del image_crop
            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            val_target = val_target.to(device)
            image_crop, theta_crop = referencef(val_image, ensemble(val_target, channel_dim).squeeze(1))
            with torch.no_grad():
                x_affine, theta = net1(val_image)

            loss_stn = 1 - (L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())

        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real: ', theta_crop[0].data.cpu().numpy())

        ncols = 3  # number of columns in final grid of images
        nrows = val_image.size()[0]  # looking at all images takes some time
        f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
        for axis in axes.flatten():
            axis.set_axis_off()
            axis.set_aspect('equal')
        for q in range(nrows):
            img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
            img_crop = np_to_img(image_crop[q, 0, :,  :].data.cpu().numpy(), 'image',massimo, minimo)
            img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
            # 0.0.11 Store results
            if nrows==1:
                axes[0].set_title("Original Test Image")
                axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[1].set_title("Cropped Test Image")
                axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[2].set_title("Reference Crop")
                axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
            else:
                axes[q, 0].set_title("Original Test Image")
                axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                axes[q, 1].set_title("Cropped Test Image")
                axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                axes[q, 2].set_title("Reference Crop")
                axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


        f.savefig(data_results + '/images_' + str(epoch) + '_val.png', bbox_inches='tight')

        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_stnpose_stncrop(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[1]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net.eval()
    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            target_old = target[rand]
            image = image.type(th.float).to(device)
            pose = keep_largest_mask_torch(image)
            target_old = target_old.type(th.float).to(device)
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            print(image.size(), target.size())
            net1.train()

            optimizer1.zero_grad()
            with torch.no_grad():
                imagepso,theta1 = net(torch.cat((image,pose), dim=1))
                imageps = imagepso[:, :1,:,:]
                grid = F.affine_grid(theta1, target_old.size(), align_corners=False)
                target = F.grid_sample(target_old, grid, align_corners=False, mode='nearest', padding_mode='zeros')
            image_crop, theta_crop = referencef(imageps, ensemble(target, channel_dim).squeeze(1))

            x_affine, theta = net1(imageps)

            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            loss_G_joint1 = L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop))
            print(L1(x_affine, image_crop).item())
            print(10*torch.sqrt(L2(theta, theta_crop)).item())
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

            if ((epoch%25==0) or (epoch == (num_epochs - 1))) and i==0:
                ncols = 3  # number of columns in final grid of images
                nrows = image.size()[0]  # looking at all images takes some time
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                for q in range(nrows):
                    img = np_to_img(imageps[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    img_crop = np_to_img(image_crop[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image', massimo, minimo)
                    # 0.0.11 Store results
                    if nrows == 1:
                        axes[0].set_title("Original Test Image")
                        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[1].set_title("Cropped Test Image")
                        axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[2].set_title("Reference Crop")
                        axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                    else:
                        axes[q, 0].set_title("Original Test Image")
                        axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 1].set_title("Cropped Test Image")
                        axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 2].set_title("Reference Crop")
                        axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)

                f.savefig(data_results + '/images_' + str(epoch) + '_train.png', bbox_inches='tight')

            del theta_crop
            del image_crop
            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target_old = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            pose = keep_largest_mask_torch(val_image)
            val_target_old = val_target_old.type(th.float).to(device)
            with torch.no_grad():
                val_imagepso,theta1 = net(torch.cat((val_image,pose), dim=1))
                val_imageps = val_imagepso[:, :1,:,:]
                grid = F.affine_grid(theta1, val_target_old.size(), align_corners=False)
                val_target = F.grid_sample(val_target_old, grid, align_corners=False, mode='nearest', padding_mode='zeros')
            image_crop, theta_crop = referencef(val_imageps, ensemble(val_target, channel_dim).squeeze(1))
            with torch.no_grad():
                x_affine, theta = net1(val_imageps)

            loss_stn = 1 - (L1(x_affine, image_crop) + 10*torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(10*torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())

        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real: ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_imageps[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                img_crop = np_to_img(image_crop[q, 0, :,  :].data.cpu().numpy(), 'image',massimo, minimo)
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image',massimo, minimo)
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            val_step += 1


        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()

def train_faster_rcnn(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)

    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            images = image.type(th.float).to(device)
            images = list(image for image in images)
            target = target.to(device)
            targets = []
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            print(image.size(), target.size())
            net1.train()

            optimizer1.zero_grad()
            seg = ensemble(target, channel_dim).squeeze(1)
            for ii in range(len(images)):
                seg_crop = torch.nonzero(seg[ii], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes = torch.tensor([0,0,512,512]).unsqueeze(0).to(device)
                    labels = torch.zeros(1,dtype=torch.int64).to(device)
                else:
                    boxes = torch.tensor([x.min(),y.min(),x.max(),y.max()]).unsqueeze(0).to(device)
                    labels = torch.ones(1,dtype=torch.int64).to(device)
                d = {}
                d['boxes'] = boxes
                d['labels'] = labels
                targets.append(d)


            output = net1(images,targets)
            print(output)
            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            for key, value in output.items():
                if key=='loss_classifier':
                    loss_G_joint1 = value
                else:
                    loss_G_joint1 += value
            print(loss_G_joint1.item())
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

            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            print(val_image.min(),val_image.mean(),val_image.max())
            val_images = list(image for image in val_image)
            val_target = val_target.to(device)
            seg = ensemble(val_target, channel_dim).squeeze(1)
            boxes, labels = torch.zeros(val_target.size()[0],4),  torch.zeros(val_target.size()[0])
            preds, pred_labels = torch.zeros(val_target.size()[0],4),  torch.zeros(val_target.size()[0])
            theta_crop, theta = torch.zeros((val_target.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((val_target.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda"))

            for jj in range(len(val_images)):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes[jj] = torch.tensor([0, 0, 512, 512]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 2] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 2] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max()]).unsqueeze(0).to(device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((x.max() - x.min()) / (val_target.size()[2] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 2] = ((x.max() + x.min()) / (val_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((y.max() - y.min()) / (val_target.size()[3] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 2] = ((y.max() + y.min()) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred = net1(val_images)
            for ii in range(val_image.size()[0]):
                if pred[ii]['boxes'].nelement() != 0 and pred[ii]['labels'][0]==1:
                    preds[ii] = pred[ii]['boxes'][0]
                    pred_labels[ii] = pred[ii]['labels'][0]
                    theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (val_target.size()[2] * 1.0))  # x2-x1/w
                    theta[ii, 0, 2] = ((preds[ii][2] + preds[ii][0]) / (val_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (val_target.size()[3] * 1.0))  # y2-y1/h
                    theta[ii, 1, 2] = ((preds[ii][3] + preds[ii][1]) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 2] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 2] = 0
            print(preds, pred_labels,theta)

            grid_cropped = F.affine_grid(theta_crop, val_image.size(), align_corners=False)
            image_crop = F.grid_sample(val_image, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta, val_image.size(), align_corners=False)
            x_affine = F.grid_sample(val_image, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            loss_stn = 1 - (L1(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())


        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real: ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_image[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :,  :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            val_step += 1


        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()


def train_stnpose_faster_rcnn(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[1]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)
    net = nets[0]
    net.load_state_dict(torch.load(data_results + '/net_pose.pth'))
    net.eval()
    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            target_old = target[rand]
            image = image.type(th.float).to(device)
            pose = keep_largest_mask_torch(image)
            target_old = target_old.type(th.float).to(device)
            targets = []
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            print(image.size(), target_old.size())
            net1.train()
            with torch.no_grad():
                imagepso,theta1 = net(torch.cat((image,pose), dim=1))
                imageps = imagepso[:, :1,:,:]
                grid = F.affine_grid(theta1, target_old.size(), align_corners=False)
                target = F.grid_sample(target_old, grid, align_corners=False, mode='nearest', padding_mode='zeros')
                images = list(image for image in imageps)

            optimizer1.zero_grad()

            seg = ensemble(target, channel_dim).squeeze(1)
            for ii in range(target.size()[0]):
                seg_crop = torch.nonzero(seg[ii], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes = torch.tensor([0,0,512,512]).unsqueeze(0).to(device)
                    labels = torch.zeros(1,dtype=torch.int64).to(device)
                else:
                    boxes = torch.tensor([x.min(),y.min(),x.max(),y.max()]).unsqueeze(0).to(device)
                    labels = torch.ones(1,dtype=torch.int64).to(device)
                d = {}
                d['boxes'] = boxes
                d['labels'] = labels
                targets.append(d)

            output = net1(images,targets)
            print(output)
            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            for key, value in output.items():
                if key=='loss_classifier':
                    loss_G_joint1 = value
                else:
                    loss_G_joint1 += value
            print(loss_G_joint1.item())
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

            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target_old = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            print(val_image.min(),val_image.mean(),val_image.max())
            pose = keep_largest_mask_torch(val_image)
            val_target_old = val_target_old.type(th.float).to(device)
            boxes, labels = torch.zeros(val_target_old.size()[0],4),  torch.zeros(val_target_old.size()[0])
            preds, pred_labels = torch.zeros(val_target_old.size()[0],4),  torch.zeros(val_target_old.size()[0])
            theta_crop, theta = torch.zeros((val_target_old.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((val_target_old.size()[0], 2, 3), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda"))

            with torch.no_grad():
                val_imagepso,theta1 = net(torch.cat((val_image,pose), dim=1))
                val_imageps = val_imagepso[:, :1,:,:]
                grid = F.affine_grid(theta1, val_target_old.size(), align_corners=False)
                val_target = F.grid_sample(val_target_old, grid, align_corners=False, mode='nearest', padding_mode='zeros')
                val_images = list(image for image in val_imageps)

            seg = ensemble(val_target, channel_dim).squeeze(1)
            for jj in range(val_target.size()[0]):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]
                x = seg_crop[1]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max():
                    boxes[jj] = torch.tensor([0, 0, 512, 512]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 2] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 2] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max()]).unsqueeze(0).to(device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((x.max() - x.min()) / (val_target.size()[2] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 2] = ((x.max() + x.min()) / (val_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((y.max() - y.min()) / (val_target.size()[3] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 2] = ((y.max() + y.min()) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred = net1(val_images)
            for ii in range(val_image.size()[0]):
                if pred[ii]['boxes'].nelement() != 0 and pred[ii]['labels'][0]==1:
                    preds[ii] = pred[ii]['boxes'][0]
                    pred_labels[ii] = pred[ii]['labels'][0]
                    theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (val_target.size()[2] * 1.0))  # x2-x1/w
                    theta[ii, 0, 2] = ((preds[ii][2] + preds[ii][0]) / (val_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                    theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (val_target.size()[3] * 1.0))  # y2-y1/h
                    theta[ii, 1, 2] = ((preds[ii][3] + preds[ii][1]) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 2] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 2] = 0
            print(preds, pred_labels,theta)

            grid_cropped = F.affine_grid(theta_crop, val_imageps.size(), align_corners=False)
            image_crop = F.grid_sample(val_imageps, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta, val_imageps.size(), align_corners=False)
            x_affine = F.grid_sample(val_imageps, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            loss_stn = 1 - (L1(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())


        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        if mdicestn > best_dicestn:
            best_dicestn = mdicestn
            torch.save(net1.state_dict(), data_results + '/net_1.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dicestn = [mdicestn]
            np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real: ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 25) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 3  # number of columns in final grid of images
            nrows = val_image.size()[0]  # looking at all images takes some time
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            for q in range(nrows):
                img = np_to_img(val_imageps[q, 0, :, :].data.cpu().numpy(), 'image')
                img_crop = np_to_img(image_crop[q, 0, :,  :].data.cpu().numpy(), 'image')
                img_affine = np_to_img(x_affine[q, 0, :, :].data.cpu().numpy(), 'image')
                # 0.0.11 Store results
                if nrows==1:
                    axes[0].set_title("Original Test Image")
                    axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[1].set_title("Cropped Test Image")
                    axes[1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[2].set_title("Reference Crop")
                    axes[2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)
                else:
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Cropped Test Image")
                    axes[q, 1].imshow(img_affine, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference Crop")
                    axes[q, 2].imshow(img_crop, cmap='gray', vmin=0, vmax=255.)


            f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
            val_step += 1


        del val_image
        del val_target

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()


def train_nnDet(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode):
    net1 = nets[0]
    optimizer1 = optim.SGD(net1.parameters(), lr=init_lr)

    valdicesstn = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss1 = []
    no_improvement = 0


    torch.cuda.synchronize()
    start_time = time.time()


    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        G_losses1 = []
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
            images = image.type(th.float).to(device)
            target = target.to(device)
            batch = {}
            batch['data'] = images
            torch.cuda.synchronize()
            step2 = time.time()
            if mode=='debug':
                print("Move images in gpu (s): ", step2 - step1)

                print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if (i % 1000) == 0:
                print('Iteration: ', i)
            print(image.size(), target.size())
            net1.train()

            optimizer1.zero_grad()
            seg = ensemble(target, channel_dim).squeeze(1)
            segg = seg > 0
            print(segg.size())
            batch['seg'] = segg.unsqueeze(1).data.cpu().numpy()
            boxs =[]
            lbls = []
            for ii in range(images.size()[0]):
                seg_crop = torch.nonzero(seg[ii], as_tuple=True)
                y = seg_crop[1]
                x = seg_crop[2]
                z = seg_crop[0]
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max() or z.min()==z.max():
                    boxes = torch.tensor([0,0,128,128,0,128]).unsqueeze(0).to(device)
                    labels = torch.zeros(1,dtype=torch.int64).to(device)
                else:
                    boxes = torch.tensor([x.min(),y.min(),x.max(),y.max(),z.min(),z.max()]).unsqueeze(0).to(device)
                    labels = torch.ones(1,dtype=torch.int64).to(device)
                boxs.append(boxes.data.cpu().numpy())
                lbls.append(labels.data.cpu().numpy())
            batch['bb_target'] = boxs
            batch['roi_labels'] = lbls

            loss_G_joint1 = net1.train_forward(batch)
            torch.cuda.synchronize()
            step3 = time.time()
            if mode == 'debug':
                print("forward step (s): ", step3 - step2)
                print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            print(loss_G_joint1.item())
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

            del image
            del target

        # ---------END TRAIN SECOND NET---
        torch.cuda.synchronize()
        stepepoch = time.time()
        print("Epoch time (min): ", (stepepoch - step) / 60)
        temp_meanG1 = np.mean(G_losses1)
        mloss1.append(temp_meanG1)

        print('[%d/%d]\t'
              % ((epoch + 1), num_epochs))
        print(
            'Loss_G1 - TRAINING (BEST AT 0) --> first batch:\t%.4f and last batch:\t%.4f --> Average of batches of\t Loss_G1: %.4f'
            % (G_losses1[0], loss_G_joint1.item(), temp_meanG1))


        del G_losses1

        print('starting validation...')
        net1.eval()

        dices_stn = []

        #for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.type(th.float).to(device)
            print(val_image.min(),val_image.mean(),val_image.max())
            batch = {}
            batch['data'] = val_image
            val_target = val_target.type(th.float).to(device)
            seg = ensemble(val_target, channel_dim).squeeze(1)
            boxes, labels = torch.zeros(val_target.size()[0],6).to("cuda"),  torch.zeros(val_target.size()[0]).to("cuda")
            theta_crop, theta = torch.zeros((val_target.size()[0], 3, 4), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda")), torch.zeros((val_target.size()[0], 3, 4), requires_grad=False, dtype=torch.float).to(
                torch.device("cuda"))

            for jj in range(val_image.size()[0]):
                seg_crop = torch.nonzero(seg[jj], as_tuple=True)
                y = seg_crop[0]  # depth: axis z
                x = seg_crop[1]  # row: axis y
                z = seg_crop[2]  # col: axis x
                if x.nelement() == 0 or x.min() == x.max() or y.min() == y.max() or z.min()==z.max():
                    boxes[jj] = torch.tensor([0, 0, 128, 128,0,128]).unsqueeze(0).to(device)
                    labels[jj] = torch.zeros(1, dtype=torch.int64).to(device)
                    theta_crop[jj, 0, 0] = 1
                    theta_crop[jj, 0, 3] = 0
                    theta_crop[jj, 1, 1] = 1
                    theta_crop[jj, 1, 3] = 0
                    theta_crop[jj, 2, 2] = 1
                    theta_crop[jj, 2, 3] = 0
                else:
                    boxes[jj] = torch.tensor([x.min(), y.min(), x.max(), y.max(),z.min(),z.max()]).unsqueeze(0).to(device)
                    labels[jj] = torch.ones(1, dtype=torch.int64).to(device)

                    theta_crop[jj, 0, 0] = ((z.max() - z.min()) / (val_target.size()[3] * 1.0))  # x2-x1/w
                    theta_crop[jj, 0, 3] = ((z.max() + z.min()) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 1, 1] = ((x.max() - x.min()) / (val_target.size()[4] * 1.0))  # y2-y1/h
                    theta_crop[jj, 1, 3] = ((x.max() + x.min()) / (val_target.size()[4] * 1.0)) - 1  # x2+x1/w - 1
                    theta_crop[jj, 2, 2] = ((y.max() - y.min()) / (val_target.size()[2] * 1.0))  # z2-z1/h
                    theta_crop[jj, 2, 3] = ((y.max() + y.min()) / (val_target.size()[2] * 1.0)) - 1  # z2+z1/w - 1
            print(boxes, labels, theta_crop)
            with torch.no_grad():
                pred_labels, preds = net1.test_forward(batch)
                volume = (preds[:,2] - preds[:,0])*(preds[:,3] - preds[:,1])*(preds[:,5] - preds[:,4])
                #print(pred['boxes'][0])
            for ii in range(val_image.size()[0]):
                if volume[ii]>0:
                #if pred['boxes'][ii]:
                #    if pred['boxes'][0][ii]['box_pred_class_id']==1:
                #        print(pred['boxes'][0][ii]['box_coords'])
                #        preds[ii] =  torch.from_numpy(pred['boxes'][0][ii]['box_coords']).to("cuda")
                #        pred_labels[ii] =  torch.tensor(pred['boxes'][0][ii]['box_pred_class_id']).to("cuda")
                        theta[ii, 0, 0] = ((preds[ii][2] - preds[ii][0]) / (val_target.size()[3] * 1.0))  # x2-x1/w
                        theta[ii, 0, 3] = ((preds[ii][2] + preds[ii][0]) / (val_target.size()[3] * 1.0)) - 1  # x2+x1/w - 1
                        theta[ii, 1, 1] = ((preds[ii][3] - preds[ii][1]) / (val_target.size()[4] * 1.0))  # y2-y1/h
                        theta[ii, 1, 3] = ((preds[ii][3] + preds[ii][1]) / (val_target.size()[4] * 1.0)) - 1  # x2+x1/w - 1
                        theta[ii, 2, 2] = ((preds[ii][5] - preds[ii][4]) / (val_target.size()[2] * 1.0))  # y2-y1/h
                        theta[ii, 2, 3] = ((preds[ii][5] + preds[ii][4]) / (val_target.size()[2] * 1.0)) - 1  # x2+x1/w - 1
                else:
                    theta[ii, 0, 0] = 1
                    theta[ii, 0, 3] = 0
                    theta[ii, 1, 1] = 1
                    theta[ii, 1, 3] = 0
                    theta[ii, 2, 2] = 1
                    theta[ii, 2, 3] = 0
            print(preds, pred_labels,theta)

            grid_cropped = F.affine_grid(theta_crop, val_image.size(), align_corners=False)
            image_crop = F.grid_sample(val_image, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta_crop, val_target.size(), align_corners=False)
            v_crop = F.grid_sample(val_target, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            grid_cropped = F.affine_grid(theta, val_image.size(), align_corners=False)
            x_affine = F.grid_sample(val_image, grid_cropped, align_corners=False, mode='bilinear')  # , padding_mode="border"

            loss_stn = 1 - (L1(x_affine, image_crop) + torch.sqrt(L2(theta, theta_crop)))
            print(L1(x_affine, image_crop).item())
            print(torch.sqrt(L2(theta, theta_crop)).item())
            dices_stn.append(loss_stn.item())


        mdicestn = np.mean(dices_stn, axis=0)
        valdicesstn.append(1 - mdicestn.item())
        print('VALIDATION (BEST AT 1) -->Average of batches of L2 loss: %.4f '
              % (mdicestn))
        #if mdicestn > best_dicestn:
        best_dicestn = mdicestn
        torch.save(net1.state_dict(), data_results + '/net_1.pth')
        print('IMPROVEMENT IN VALIDATION')
        total_dicestn = [mdicestn]
        np.save(data_results + '/dice_valstn.npy', np.asarray(total_dicestn, dtype=np.float32))

        del dices_stn


        print('THETA: ', theta[0].data.cpu().numpy())
        print('THETA real: ', theta_crop[0].data.cpu().numpy())

        if ((val_step * 10) == epoch) or (epoch == (num_epochs - 1)):
            ncols = 4  # number of columns in final grid of images
            nrows = 3  # looking at all images takes some time
            v_c = ensemble(v_crop, channel_dim)
            f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
            for axis in axes.flatten():
                axis.set_axis_off()
                axis.set_aspect('equal')
            img0 = np_to_img(val_image[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            img_pose0 = np_to_img(image_crop[0, 0, 64, :, :].data.cpu().numpy(), 'image')
            v_c0 = np_to_img(v_c[0, 0, 64, :, :].data.cpu().numpy(), 'target')
            img_affine0 = np_to_img(x_affine[0, 0, 64, :, :].data.cpu().numpy(), 'image')

            img1 = np_to_img(val_image[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            img_pose1 = np_to_img(image_crop[0, 0, :, 64, :].data.cpu().numpy(), 'image')
            v_c1 = np_to_img(v_c[0, 0, :, 64, :].data.cpu().numpy(), 'target')
            img_affine1 = np_to_img(x_affine[0, 0, :, 64, :].data.cpu().numpy(), 'image')

            img2 = np_to_img(val_image[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            img_pose2 = np_to_img(image_crop[0, 0, :, :, 64].data.cpu().numpy(), 'image')
            v_c2 = np_to_img(v_c[0, 0, :, :, 64].data.cpu().numpy(), 'target')
            img_affine2 = np_to_img(x_affine[0, 0, :, :, 64].data.cpu().numpy(), 'image')

            # 0.0.11 Store results
            axes[0, 0].set_title("Original Test Image")
            axes[0, 0].imshow(img0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 1].set_title("Target")
            axes[0, 1].imshow(img_pose0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 2].set_title("Pred")
            axes[0, 2].imshow(v_c0, cmap='gray', vmin=0, vmax=255.)
            axes[0, 3].set_title("Pred")
            axes[0, 3].imshow(img_affine0, cmap='gray', vmin=0, vmax=255.)

            axes[1, 0].set_title("Original Test Image")
            axes[1, 0].imshow(img1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 1].set_title("Target")
            axes[1, 1].imshow(img_pose1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 2].set_title("Pred")
            axes[1, 2].imshow(v_c1, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[1, 3].set_title("Pred")
            axes[1, 3].imshow(img_affine1, cmap='gray', vmin=0, vmax=255., origin='lower')

            axes[2, 0].set_title("Original Test Image")
            axes[2, 0].imshow(img2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 1].set_title("Target")
            axes[2, 1].imshow(img_pose2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 2].set_title("Pred")
            axes[2, 2].imshow(v_c2, cmap='gray', vmin=0, vmax=255., origin='lower')
            axes[2, 3].set_title("Pred")
            axes[2, 3].imshow(img_affine2, cmap='gray', vmin=0, vmax=255., origin='lower')

            f.savefig(data_results + '/images_' + str(epoch) + '_val.png', bbox_inches='tight')
            # 0.0.11 Store results
            val_step += 1

        exit()
        del val_image
        del val_target
        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("STNcrop - Training and Validation Loss")
    plt.plot(mloss1, label="Training Loss (best at 0)")
    plt.plot(valdicesstn, label="Validation Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_stncrop_plot.png')

    torch.cuda.empty_cache()
