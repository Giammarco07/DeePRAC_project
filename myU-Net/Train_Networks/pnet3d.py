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
from utils.losses import L1, L2, sig_bce


def train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, init_lr,
                 train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnn,
                 mode=None, log=True):
    net = nets[0]
    optimizer = optimizers[0]
    num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of model parameters:', num_p)
    k = 23
    gamma = 0.5 * 10e-7
    if learning == 'autocast':
        import torch.cuda.amp as amp2
        scaler = amp2.GradScaler()
        # Lists to keep track of progress
    valdices = []
    train_loader0 = train_loader[0]
    val_loader0 = val_loader[0]
    train_loader1 = train_loader[1]
    val_loader1 = val_loader[1]
    mloss = []
    y_val = []
    no_improvement = 0

    start_time = time.time()
    # For each epoch

    for epoch in range(start_epoch, num_epochs):
        print("starting epoch ", epoch + 1)
        # For each batch in the dataloader
        net.train()
        G_losses = []

        torch.cuda.synchronize()
        step = time.time()
        # for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        for i, (data0, data1) in enumerate(zip(train_loader0, train_loader1)):
            if learning == 'autocast':
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
                batch = image.size()[0]

                target_patches = torch.zeros((batch, 1), dtype=torch.float).to(device)
                for bb in range(batch):
                    if torch.is_nonzero(torch.sum(seg[bb])):
                        target_patches[bb] = 1

                torch.cuda.synchronize()
                step2 = time.time()
                if mode == 'debug':
                    print("Move images in gpu (s): ", step2 - step1)

                    print(f'Before forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'Before forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                with amp2.autocast():
                    output = net(image)

                    torch.cuda.synchronize()
                    step3 = time.time()

                    if mode == 'debug':
                        print("forward step (s): ", step3 - step2)
                        print(f'After forward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                        print(f'After forward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

                    loss_G_joint = sig_bce(output, target_patches)

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

                if (i + 1) % 10 == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 12)

                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)

                    # Updates the scale for next iteration.
                    scaler.update()
                    optimizer.zero_grad()
                    #print('Mean of first layer gradient of iterations from ', str(i + 1 - 10), 'to ', str(i + 1),
                    #      torch.mean(list(net.parameters())[0].grad))
                    #print('Mean of last layer gradient of iterations from ', str(i + 1 - 10), 'to ', str(i + 1),
                    #      torch.mean(list(net.parameters())[-1].grad))

                # loss_G_joint.backward()
                # optimizer.step()

                torch.cuda.synchronize()
                step5 = time.time()
                if mode == 'debug':
                    print("backward step (s): ", step5 - step4)
                    print(f'After backward pass - Cuda memory allocated: {torch.cuda.memory_allocated() / 1e9}')
                    print(f'After backward pass - Cuda memory cached: {torch.cuda.memory_cached() / 1e9}')

            if i == 0 and (((val_step * 15) == epoch) or (epoch == (num_epochs - 1))):
                print('prediction:', output)
                print('target', target_patches)

            del image, image0, image1
            del target, target0, target1
            del seg
            del output

            # Save Losses for plotting later
            G_losses.append(loss_G_joint.item())
            del loss_G_joint

            torch.cuda.synchronize()
            step6 = time.time()
            if mode == 'debug':
                print("optimizer step (s): ", step6 - step5)

            if asnn:
                if (i + 1) == (500 // batch):
                    break

        print("Epoch time (s): ", step6 - step)
        temp_meanG = np.mean(G_losses)
        mloss.append(temp_meanG)

        print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f'
              % ((epoch + 1), num_epochs, G_losses[0], G_losses[-1]))
        print('TRAINING (BEST AT 0) --> Average of batches of\t Loss_G: %.4f'
              % (temp_meanG))

        del G_losses

        print('starting validation...')
        net.eval()

        dices = []
        tp, tn, fp, fn = 0, 0, 0, 0
        # for j, data in enumerate(val_loader, 0):
        for j, (data0, data1) in enumerate(zip(val_loader0, val_loader1)):
            if (j % 500) == 0:
                print('Iteration: ', j)
            image0, target0 = data0
            image1, target1 = data1
            val_image = torch.cat((image0, image1), 0)
            val_target = torch.cat((target0, target1), 0)
            val_image = val_image.to(device)
            val_target = val_target.to(device)
            val_seg = torch.argmax(val_target, dim=1)
            batch = val_image.size()[0]
            val_target_patches = torch.zeros((batch, 1), dtype=torch.float).to(device)
            for bb in range(batch):
                if torch.is_nonzero(torch.sum(val_seg[bb])):
                    val_target_patches[bb] = 1

            if learning == 'autocast':
                with amp2.autocast():
                    with torch.no_grad():
                        pred = torch.sigmoid(net(val_image))
                        pred[pred < 0.5] = 0
                        pred[pred >= 0.5] = 1


            for bb in range(batch):
                if pred[bb] == 0 and val_target_patches[bb] == 0:
                    tn += 1
                if pred[bb] == 1 and val_target_patches[bb] == 1:
                    tp += 1
                if pred[bb] == 1 and val_target_patches[bb] == 0:
                    fp += 1
                if pred[bb] == 0 and val_target_patches[bb] == 1:
                    fn += 1


            if (((val_step * 15) == epoch) or (epoch == (num_epochs - 1))) and j == 0:
                print('prediction:', pred)
                print('target', val_target_patches)
                val_step += 1

            del pred
            del val_image
            del val_target

        print('tp:', tp)
        print('tn:', tn)
        print('fp:', fp)
        print('fn:', fn)
        p = tp / (tp + fp + ee)
        r = tp / (tp + fn + ee)
        mdice = 2 * (p * r) / (p + r + ee)

        net.train()
        print('VALIDATION (BEST AT 1) -->Average of batches of Dice loss: %.4f'
              % (mdice))
        valdices.append(1 - mdice)
        y_val.append(epoch)

        if mdice > best_dice:
            best_dice = mdice
            torch.save(net.state_dict(), data_results + '/pnet_3d.pth')
            torch.save(optimizer.state_dict(), data_results + '/optimizer.pth')
            print('IMPROVEMENT IN VALIDATION')
            total_dice = [mdice]
            np.save(data_results + '/dice_val.npy', np.asarray(total_dice, dtype=np.float32))
            no_improvement = 0
        else:
            no_improvement += 1

        del dices

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
    plt.plot(mloss, label="Training Total Loss (best at 0)")
    plt.plot(valdices, label="Validation Dice Loss (best at 0)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_net_3d_plot.png')
