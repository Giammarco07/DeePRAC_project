from __future__ import print_function
import torch as th
import torch.nn.parallel
import torch.utils.data
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
from utils.figures import np_to_img
from utils.losses import ce, dice_loss, dice_loss_val_new

def train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, init_lr, channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnn, mode=None, log=True):
    net = nets[0]
    optimizer = optimizers[0]
    num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of model parameters:', num_p)
    print('Dice loss log: ', log)
    if learning == 'autocast':
        import torch.cuda.amp as amp2
        scaler = amp2.GradScaler()
    # Lists to keep track of progress
    valdices = []
    vals1 = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    if channel_dim > 2:
        vals2 = []
    if channel_dim > 3:
        vals3 = []
    mloss = []
    y_val = []
    no_improvement = 0

    print("starting Training Loop...")
    start_time = time.time()
    # For each epoch

    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        net.train()
        G_losses = []

        torch.cuda.synchronize()
        step = time.time()
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            if learning == 'autocast':
                if (i % 249) == 0:
                    print('Iteration: ', i)
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

                torch.cuda.synchronize()
                step2 = time.time()
                if mode == 'debug':
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer.zero_grad()
                with amp2.autocast():
                    output = net(image)

                    torch.cuda.synchronize()
                    step3 = time.time()

                    if mode == 'debug':
                        print("forward step (s): ", step3 - step2)
                        print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                        print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                    out = output.clone() + 1e-20
                    loss_G_joint = ce(torch.log(out), target[:,1,:,:])

                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss_G_joint).backward()

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')
            else:
                if (i % 1000) == 0:
                    print('Iteration: ', i)
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


                torch.cuda.synchronize()
                step2 = time.time()
                if mode == 'debug':
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer.zero_grad()

                output = net(image)

                torch.cuda.synchronize()
                step3 = time.time()

                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint = ce(torch.log(output+ 1e-20), target[:, 1, :, :])

                torch.cuda.synchronize()
                step4 = time.time()
                if mode == 'debug':
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                loss_G_joint.backward()

                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer.step()

            if i == 0 and (((val_step * 25) == epoch) or (epoch == (num_epochs - 1))):
                pred_t = torch.argmax(output,dim=1, keepdim=True)
                ncols = 3  # number of columns in final grid of images
                nrows = image.size()[0]  # looking at all images takes some time
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                for q in range(nrows):
                    img = np_to_img(image[q, 1, :, :].type(torch.float16).data.cpu().numpy(), 'image')
                    prd = np_to_img(pred_t[q, 0, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(target[q, 1, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[q, 0].set_title("Original Test Image")
                    axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 1].set_title("Predicted")
                    axes[q, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[q, 2].set_title("Reference")
                    axes[q, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                f.savefig(data_results + '/images_train.png', bbox_inches='tight')

                del pred_t
                del img
                del prd
                del tgt

            del image, image0, image1
            del target, target0, target1
            del output

            # Save Losses for plotting later
            G_losses.append(loss_G_joint.item())
            torch.cuda.synchronize()
            step6 = time.time()
            if mode == 'debug':
                print("optimizer step (s): ", step6 - step5)

            if asnn:
                if (i+1)==115:
                    break

        print("Epoch time (s): ", step6 - step)
        temp_meanG = np.mean(G_losses)
        mloss.append(temp_meanG)

        print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f'
              % ((epoch + 1), num_epochs, G_losses[0], loss_G_joint.item()))
        print('TRAINING (BEST AT 0) --> Average of batches of\t Loss_G: %.4f'
              % (temp_meanG))

        del G_losses

        print('starting validation...')
        net.eval()

        dices = []
        dices_s1 = []
        if channel_dim > 2:
            dices_s2 = []
        if channel_dim > 3:
            dices_s3 = []


        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            if (j % 500) == 0:
                print('Iteration: ', j)
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_t = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            all_seg_labels = np.unique(val_t[:,1,:,:].data.cpu().numpy())
            val_target = torch.as_tensor(np.zeros((val_t.shape[0],len(all_seg_labels), val_t.shape[2],val_t.shape[3])), dtype=torch.long).to(device)
            for i, l in enumerate(all_seg_labels):
                val_target[:,i,:,:][val_t[:,1,:,:] == l] = 1

            if learning == 'autocast':
                with amp2.autocast():
                    with torch.no_grad():
                        pred = net(val_image)
            else:
                with torch.no_grad():
                    pred = net(val_image)

            val_dice = dice_loss(pred, val_target)
            dices.append(val_dice.item())
            val_s1 = dice_loss_val_new(pred[6:12, 1, :, :], val_target[6:12, 1, :, :])
            dices_s1.append(val_s1.item())
            if channel_dim > 2:
                val_s2 = dice_loss_val_new(pred[6:12, 2, :, :], val_target[6:12, 2, :, :])
                dices_s2.append(val_s2.item())
            if channel_dim > 3:
                val_s3 = dice_loss_val_new(pred[6:12, 3, :, :], val_target[6:12, 3, :, :])
                dices_s3.append(val_s3.item())

            if (((val_step * 25) == epoch) or (epoch == (num_epochs - 1))) and j == 0:
                pred_ = torch.argmax(pred, dim=1, keepdim=True)
                ncols = 3  # number of columns in final grid of images
                nrows = val_image.size()[0]  # looking at all images takes some time
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                for q in range(nrows):
                    img = np_to_img(val_image[q, 1, :, :].type(torch.float16).data.cpu().numpy(), 'image',massimo, minimo)
                    prd = np_to_img(pred_[q, 0, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(val_target[q, 1, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    if nrows > 1:
                        axes[q, 0].set_title("Original Test Image")
                        axes[q, 0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 1].set_title("Predicted")
                        axes[q, 1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                        axes[q, 2].set_title("Reference")
                        axes[q, 2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    else:
                        axes[0].set_title("Original Test Image")
                        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        axes[1].set_title("Predicted")
                        axes[1].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                        axes[2].set_title("Reference")
                        axes[2].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                f.savefig(data_results + '/images_' + str(val_step * 25) + '_val.png', bbox_inches='tight')
                val_step += 1

                del pred_

            del pred
            del val_image, image0, image1
            del val_target, target0, target1

        net.train()
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

        y_val.append(epoch)

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
            torch.save(optimizer.state_dict(), data_results + '/optimizer.pth')
            torch.save(net.state_dict(), data_results + '/net.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dice = [mdice]
            np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
            no_improvement = 0
        else:
            no_improvement += 1

        del dices

        #poly learning rate policy
        for g in optimizer.param_groups:
            g['lr'] = init_lr * ((1 - (epoch/num_epochs))**0.9)
            print('new lr:', g['lr'])

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))

        # early stopping
        if (no_improvement >= 15) and (epoch >= early_stopping):
            break
        else:
            continue

    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(mloss, label="Training Loss (best at 0)")
    plt.plot(valdices, label="Validation Loss (best at 1)")
    plt.plot(vals1, label="Val Structure 1")
    if channel_dim > 2:
        plt.plot(vals2, label="Val Structure 2")
    if channel_dim > 3:
        plt.plot(vals3, label="Val Structure 3")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_net_plot.png')

