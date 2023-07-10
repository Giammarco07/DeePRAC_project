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
from skimage.morphology import skeletonize
from utils.figures import np_to_img
from utils.losses import fnr,ce, soft_dice_loss, dice_loss, soft_dice_loss_old,soft_dice_loss_batch, soft_dice_loss_old_new, dice_loss_val_new, compute_dtm, compute_ddt, L1, bce, general_dice_loss_batch, c_loss, top_loss
from utils.vesselness_torch import vesselness_frangi_ssvmd, vesselness_jerman_ssvmd, msloss, fvloss
from Test_Networks.nnunet3d import val

ce_w = torch.nn.NLLLoss(torch.tensor([0.3,0.7]),reduction='none').to(device='cuda')

def log_gaussian(x, mu, logvar):
    PI = mu.new([np.pi])

    x = x.view(x.shape[0], -1)
    mu = mu.view(x.shape[0], -1)
    logvar = logvar.view(x.shape[0], -1)

    N, D = x.shape

    log_norm = (-1 / 2) * (D * torch.log(2 * PI) +
                           logvar.sum(dim=1) +
                           (((x - mu) ** 2) / (logvar.exp())).sum(dim=1))

    return log_norm

def train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, init_lr,
                   channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnn,
                   mode=None, log=True, valsw = False, others = None):
    net1 = nets[0]
    net1.load_state_dict(torch.load(data_results + '/net.pth'))
    net1.eval()
    net2 = nets[1]

    channel_dim -= 1
    print('channel dim:',channel_dim)

    #net.convblock1[0].weight.register_hook(lambda x: print('grad accumulated in convblock1'))
    optimizer = optimizers[0]
    num_p = sum(p.numel() for p in net2.parameters() if p.requires_grad)
    print('Number of model parameters:', num_p)
    if learning == 'autocast':
        import torch.cuda.amp as amp2
        scaler = amp2.GradScaler()
        # Lists to keep track of progress
    valdices = []
    vals = np.zeros((num_epochs, channel_dim - 1))
    train_loader0 = train_loader[0]
    train_loader1 = train_loader[1]


    mloss = []
    vloss = []
    y_val = []
    no_improvement = 0

    start_time = time.time()
    # For each epoch

    if valsw:
        input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, val_label_path, val_data_path, data_results, massimo, minimo, tta, res, preprocessing, device = others
    else:
        val_loader0 = val_loader[0]
        val_loader1 = val_loader[1]

    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        net2.train()
        G_losses = []
        V_losses = []
        torch.cuda.synchronize()
        step = time.time()
        # for i, data in enumerate(train_loader, 0):
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            if learning == 'autocast':
                if (i % 1000) == 0:
                    print('Iteration: ', i)

                if mode == 'debug':
                    torch.cuda.synchronize()
                    step1 = time.time()
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
                    
                seg = torch.argmax(target, dim=1)
                seg_vessels = seg.bool().long()
                batch = image.size()[0]                    


                if mode == 'debug':
                    torch.cuda.synchronize()
                    step2 = time.time()
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer.zero_grad()
                with amp2.autocast():
                    with torch.no_grad():
                        output = net1(image)[0]
                        output[:,1] = output[:,1] + output[:,2]
                        output = output[:,:2]
                    out, mu, logvar, z = net2(output, 'training')
                    ref = out.clone() + 1e-20

                    log_q_z_x = log_gaussian(z, mu, logvar)
                    log_p_z = log_gaussian(z, z.new_zeros(z.shape), z.new_zeros(z.shape))
                    log_p_x_z = (-ce_w(torch.log(ref), seg_vessels)).view(batch,-1).mean(1)

                    loss_G_joint = torch.mean(1e-3 * (log_q_z_x - log_p_z) - 100*log_p_x_z)


                if mode == 'debug':
                    torch.cuda.synchronize()
                    step4 = time.time()
                    print("Loss step (s): ", step4 - step3)
                    print(f'Before backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss_G_joint).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net2.parameters(), 12)

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()


                if mode == 'debug':
                    torch.cuda.synchronize()
                    step5 = time.time()
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
                seg = torch.argmax(target, dim=1)

                torch.cuda.synchronize()
                step2 = time.time()
                if mode == 'debug':
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                optimizer.zero_grad()

                with torch.no_grad():
                    output = net1(image)
                out, mu, logvar, z = net2(output[0],'training')
                ref = out.clone() + 1e-20

                torch.cuda.synchronize()
                step3 = time.time()

                if mode == 'debug':
                    print("forward step (s): ", step3 - step2)
                    print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                log_q_z_x = log_gaussian(z, mu, logvar)
                log_p_z = log_gaussian(z, z.new_zeros(z.shape), z.new_zeros(z.shape))
                log_p_x_z = (-ce_w(torch.log(ref), seg)).view(batch,-1).mean(1)

                loss_G_joint = torch.mean(1e-3 * (log_q_z_x - log_p_z) - log_p_x_z)

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

            if i == 0 and (((val_step * 15) == epoch) or (epoch == (num_epochs - 1))):
                target_ = torch.argmax(target, dim=1, keepdim=True)
                pred_t = torch.argmax(output, dim=1, keepdim=True)
                ncols = 2 + image.size()[1]  # number of columns in final grid of images
                nrows = image.size()[0]  # looking at all images takes some time
                f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                for axis in axes.flatten():
                    axis.set_axis_off()
                    axis.set_aspect('equal')
                for q in range(nrows):
                    for ch in range(image.size()[1]):
                        img = np_to_img(image[q, ch, image.size()[2]//2, :, :].type(torch.float16).data.cpu().numpy(), 'image')
                        axes[q, ch].set_title("Original Test Image channel " + str(ch))
                        axes[q, ch].imshow(img, cmap='gray', vmin=0, vmax=255.)
                    prd = np_to_img(pred_t[q, 0, image.size()[2]//2, :, :].data.cpu().numpy(), 'target')
                    tgt = np_to_img(target_[q, 0, image.size()[2]//2, :, :].data.cpu().numpy(), 'target')
                    # 0.0.11 Store results
                    axes[q, -2].set_title("Predicted")
                    axes[q, -2].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                    axes[q, -1].set_title("Reference")
                    axes[q, -1].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                f.savefig(data_results + '/images_train.png', bbox_inches='tight')
                del f
                del target_
                del pred_t
                del img
                del prd
                del tgt

            del image, image0, image1
            del target, target0, target1
            del seg
            del output

            # Save Losses for plotting later
            G_losses.append(loss_G_joint.item())
            d = torch.mean(log_q_z_x - log_p_z).item()
            c = torch.mean(-log_p_x_z).item()
            del loss_G_joint, log_p_x_z, log_p_z, log_q_z_x
            torch.cuda.synchronize()
            step6 = time.time()
            if mode == 'debug':
                print("optimizer step (s): ", step6 - step5)

            if asnn:
                if (i + 1) == (500//batch):
                    break

        print("Epoch time (s): ", step6 - step)
        temp_meanG = np.mean(G_losses)
        mloss.append(temp_meanG)
        vloss.append(np.mean(V_losses))

        print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f (ce: \t%.4f + d: \t%.4f)'
              % ((epoch + 1), num_epochs, G_losses[0], G_losses[-1], c, d))
        print('TRAINING (BEST AT 0) --> Average of batches of\t Loss_G: %.4f'
              % (temp_meanG))

        del G_losses,V_losses

        print('starting validation...')
        torch.cuda.synchronize()
        valstep = time.time()
        if valsw:
            dices, hds = val(input_folder, patch_size, batch_size, workers, net2, channel_dim, in_c, val_label_path, val_data_path,
         data_results, massimo, minimo, tta, res, preprocessing, device, epoch, do_seg=True)
            mdice = np.mean(dices)
            dices_s = np.mean(dices, axis=0)
        else:
            net2.eval()
            dices = []
            dices_s = np.zeros((channel_dim - 1))
            # for j, data in enumerate(val_loader, 0):
            for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
                torch.cuda.synchronize()
                valstep0 = time.time()
                if (j % 500) == 0:
                    print('Iteration: ', j)
                image0, target0 = data0
                image1, target1 = data1
                val_image = torch.cat((image0, image1), 0)
                val_target = torch.cat((target0, target1), 0)
                val_image = val_image.to(device)
                val_target = val_target.to(device)
                val_seg = torch.argmax(val_target, dim=1)
                val_seg_vessels = val_seg.bool().long()
                val_target = torch.moveaxis(torch.nn.functional.one_hot(val_seg_vessels, num_classes=2), -1, 1)

                torch.cuda.synchronize()    
                valstep1 = time.time()
                #print("Loading time (s): ", valstep1 - valstep0)
                if learning == 'autocast':
                    with amp2.autocast():
                        with torch.no_grad():
                            predict = net1(val_image)[0]
                            predict[:, 1] = predict[:, 1] + predict[:, 2]
                            predict = predict[:, :2]
                            pred, mu, logvar, z = net2(predict,'test')

                else:
                    with torch.no_grad():
                        predict = net1(val_image)
                        pred, mu, logvar, z = net2(predict[0],'test')

                torch.cuda.synchronize()    
                valstep2 = time.time()
                #print("Inference time (s): ", valstep2 - valstep1)

                val_dice = 1 - soft_dice_loss_batch(pred, val_target)
                dices.append(val_dice.item())
                
                for k in range(1, channel_dim):
                        val_s = dice_loss_val_new(pred[batch // 2:batch, k, :, :, :], val_target[batch // 2:batch, k, :, :, :])
                        dices_s[k - 1] += val_s.item()


                torch.cuda.synchronize()    
                valstep3 = time.time()
                #print("Loss time (s): ", valstep3 - valstep2)
                
                if (((val_step * 15) == epoch) or (epoch == (num_epochs - 1))) and j == 0:
                    val_target_ = torch.argmax(val_target, dim=1, keepdim=True)
                    pred_ = torch.argmax(pred, dim=1, keepdim=True)
                    ncols = 2 + val_image.size()[1] # number of columns in final grid of images
                    nrows = val_image.size()[0]  # looking at all images takes some time
                    f, axes = plt.subplots(nrows, ncols, figsize=(20, 20 * nrows / ncols))
                    for axis in axes.flatten():
                        axis.set_axis_off()
                        axis.set_aspect('equal')
                    for q in range(nrows):
                        for ch in range(val_image.size()[1]):
                            img = np_to_img(val_image[q, ch, val_image.size()[2]//2, :, :].type(torch.float16).data.cpu().numpy(),
                                        'image')  # ,massimo, minimo)
                            if nrows > 1:
                                axes[q, ch].set_title("Original Test Image channel " + str(ch))
                                axes[q, ch].imshow(img, cmap='gray', vmin=0, vmax=255.)
                            else:
                                axes[ch].set_title("Original Test Image channel " + str(ch))
                                axes[ch].imshow(img, cmap='gray', vmin=0, vmax=255.)
                        prd = np_to_img(pred_[q, 0, val_image.size()[2]//2, :, :].data.cpu().numpy(), 'target')
                        tgt = np_to_img(val_target_[q, 0, val_image.size()[2]//2, :, :].data.cpu().numpy(), 'target')
                        # 0.0.11 Store results
                        if nrows > 1:
                            axes[q, -2].set_title("Predicted")
                            axes[q, -2].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                            axes[q, -1].set_title("Reference")
                            axes[q, -1].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                        else:
                            axes[-2].set_title("Predicted")
                            axes[-2].imshow(prd, cmap='gray', vmin=0, vmax=255.)
                            axes[-1].set_title("Reference")
                            axes[-1].imshow(tgt, cmap='gray', vmin=0, vmax=255.)
                    f.savefig(data_results + '/images_' + str(val_step * 15) + '_val.png', bbox_inches='tight')
                    del f
                    val_step += 1

                    del val_target_
                    del pred_
                torch.cuda.synchronize()    
                valstep4 = time.time()
                #print("Figure time (s): ", valstep4 - valstep3)
                
                del pred
                del val_image
                del val_target
                
                if asnn:
                    if (j + 1) == (500//batch):
                            break
                torch.cuda.synchronize()    
                valstep5 = time.time()
                #print("Del time (s): ", valstep5 - valstep4)
                
            batch_val = j + 1
            dices_s /= batch_val
            mdice = np.mean(dices, axis=0)

        net2.train()
        valdices.append(1 - mdice.item())
        print('Dice Loss: ',mdice)
        for k in range(1, channel_dim):
            vals[epoch, k - 1] = 1 - dices_s[k - 1].item()
            print('Dice score (BEST AT 1) Structure' + str(k) + ': %.4f'
                  % (dices_s[k - 1]))
            np.savetxt('vals.csv', vals, fmt='%d', delimiter=',')

        y_val.append(epoch)
        print('VALIDATION (BEST AT 1) -->Average of batches of validation loss: %.4f'
              % (mdice))




        if mdice > best_dice:
            best_dice = mdice
            torch.save(net2.state_dict(), data_results + '/net_1.pth')
            torch.save(optimizer.state_dict(), data_results + '/optimizer.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dice = [mdice]
            np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
            no_improvement = 0
        else:
            no_improvement += 1

        del dices

        # poly learning rate policy
        for g in optimizer.param_groups:
            g['lr'] = init_lr * ((1 - (epoch / num_epochs)) ** 0.9)
            print('new lr: ', g['lr'])

        np.save(data_results + '/last_epoch.npy', np.asarray(epoch, dtype=np.float32))
        torch.cuda.synchronize()    
        valstep6 = time.time()
        print("Validation time (s): ", valstep6 - valstep)
        # if there are no improvements in validation loss after at least 1/3 epochs, there is an early stopping
        if (no_improvement > 15) and (epoch > early_stopping):
            break
        else:
            continue


    elapsed = time.time()
    print('TIME (s):', elapsed - start_time)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(mloss, label="Training Total Loss (best at 0)")
    plt.plot(valdices, label="Validation Dice Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_net_3d_plot.png')

    np.savez(data_results + '/training.npz', name1=mloss, name2=valdices)
