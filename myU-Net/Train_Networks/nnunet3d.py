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
from utils.losses import fnr,ce, soft_dice_loss, dice_loss, soft_dice_loss_old,soft_dice_loss_batch, soft_dice_loss_old_new, dice_loss_val_new, compute_dtm, compute_ddt, L1, bce, general_dice_loss_batch
from utils.vesselness_torch import vesselness_frangi_ssvmd, vesselness_jerman_ssvmd, msloss, fvloss
from Test_Networks.nnunet3d import val

def train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, init_lr,
                   channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnn,
                   mode=None, log=True, valsw = False, others = None):
    net = nets[0]
    #net.convblock1[0].weight.register_hook(lambda x: print('grad accumulated in convblock1'))
    optimizer = optimizers[0]
    num_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Number of model parameters:', num_p)
    k = 23
    if supervision=='deepvesselgt' or supervision=='deepvesselgtnew':
        wv = 0.01 #basic 0.01 #then 0.05
    elif supervision=='deepvesselgtdeep' or supervision=='deepvesselgtdeepselffrangi':
        wv = 0.05
    elif supervision=='deepvessel' or supervision=='deepvesseltotselffrangi':
        wv = 0.1
    elif supervision=='deepvesselgtfrangi' or supervision=='deepvesselgtjerman':
        wv = 1
    elif supervision=='deepvesselgtselffrangi' or supervision=='deepvesselgtnewselffrangi':
        wv = 0.05
    else:
        wv = 0
    print('wv:',wv)
    wd = 1
    gamma = 0.5 * 10e-7
    if learning == 'autocast':
        import torch.cuda.amp as amp2
        scaler = amp2.GradScaler()
        # Lists to keep track of progress
    valdices = []
    vals = np.zeros((num_epochs, channel_dim - 1))
    if supervision[0:10]=='deepvessel' or supervision=='fnr':
        valv = np.zeros((num_epochs, channel_dim - 1))
        valvs = []   
        valtotal = []           
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
        net.train()
        G_losses = []
        V_losses = []
        torch.cuda.synchronize()
        step = time.time()
        # for i, data in enumerate(train_loader, 0):
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

                    if supervision == 'deep':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            if log:
                                loss_dice = -torch.log(soft_dice_loss(out, target))
                            else:
                                loss_dice = soft_dice_loss_old(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                    elif supervision == 'ce':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_ce)
                    elif supervision == 'deepbatch':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                    elif supervision == 'gdl':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = general_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                    elif supervision == 'gdlandbl':
                        w = (8 / 15)
                        with torch.no_grad():
                            gt_dis = compute_dtm(target[:,1:].cpu().numpy(), target[:,1:].shape)
                            gt_dis = torch.as_tensor((gt_dis-1.), dtype=torch.long).to(device)
                            gt_dis_n = compute_dtm((1-target[:,1:]).cpu().numpy(), target[:,1:].shape)
                            gt_dis_n = torch.as_tensor((gt_dis_n), dtype=torch.long).to(device)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = general_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            loss_bl = torch.mean(out[:,1:]*(-gt_dis)+out[:,1:]*(gt_dis_n))
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce + loss_bl)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce + loss_bl)
                    elif supervision == 'deepvesselmsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss([output[0]],target,sigma=25,gt=False)
                        loss_G_joint += wv*loss_vessel
                        V_losses.append(wv * loss_vessel.item())

                    elif supervision == 'deepvesseltsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss([output[0]], target, sigma=25, gt=False)
                        loss_vessel_self = fvloss([output[0]], target, sigma=25)
                        loss_G_joint += (wv * loss_vessel + loss_vessel_self)
                        V_losses.append(wv * loss_vessel.item()+loss_vessel_self.item())

                    elif supervision == 'deepvesselgtmsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:   
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss([output[0]], target, sigma=25)
                        loss_G_joint += wv * loss_vessel
                        V_losses.append(wv * loss_vessel.item())

                    elif supervision == 'deepvesselgttsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:   
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss([output[0]], target, sigma=25)
                        loss_vessel_self = fvloss([output[0]], target, sigma=25)
                        loss_G_joint += (wv * loss_vessel + loss_vessel_self)
                        V_losses.append(wv * loss_vessel.item()+loss_vessel_self.item())

                    elif supervision == 'deepvesselgtdeepmsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss(output, target, sigma=25)
                        loss_G_joint += wv * loss_vessel
                        V_losses.append(wv * loss_vessel.item())

                    elif supervision == 'deepvesselgtdeeptsloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel = msloss(output, target, sigma=25)
                        loss_vessel_self = fvloss(output, target, sigma=25)
                        loss_G_joint += (wv * loss_vessel + loss_vessel_self)
                        V_losses.append(wv * loss_vessel.item()+loss_vessel_self.item())

                    elif supervision == 'deepvesselfvloss':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:   
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                        loss_vessel_self = fvloss([output[0]], target, sigma=25)
                        loss_G_joint += loss_vessel_self
                        V_losses.append(wv * loss_vessel.item()+loss_vessel_self.item())

                    elif supervision == 'deepvesselgtfrangissvmd':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:   
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                                kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
                                gt_dil = torch.clamp(
                                    torch.nn.functional.conv3d(target.type(torch.float16), kernel, padding=1, groups=channel_dim), 0, 1)
                                for v in range(1, channel_dim):
                                    gt = gt_dil[:,v, :, :, :]
                                    gt_dilate = gt.detach()
                                    ww = torch.sum(target[:,v]) 
                                    wt = ww.type(torch.uint8).detach()
                                    if wt!= 0:
                                        eigenv_tr = vesselness_true(target[:,v], gt_dilate,sigma=3)
                                        eigenv_true = eigenv_tr.detach()
                                        loss_vessel = vesselness_frangi_ssvmd(out[:,v], gt_dilate, eigenv_true,sigma=3)
                                        loss_G_joint += wv*loss_vessel
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                    elif supervision == 'deepvesselgtjermanssvmd':
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            if p == 0:   
                                loss_G_joint = w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)
                                kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
                                gt_dil = torch.clamp(
                                    torch.nn.functional.conv3d(target.type(torch.float16), kernel, padding=1, groups=channel_dim), 0, 1)
                                for v in range(1, channel_dim):
                                    gt = gt_dil[:,v, :, :, :]
                                    gt_dilate = gt.detach()
                                    ww = torch.sum(target[:,v]) 
                                    wt = ww.type(torch.uint8).detach()
                                    if wt!= 0:
                                        eigenv_tr = vesselness_true(target[:,v], gt_dilate,sigma=3)
                                        eigenv_true = eigenv_tr.detach()
                                        loss_vessel = vesselness_jerman_ssvmd(out[:,v], gt_dilate, eigenv_true,sigma=3)
                                        loss_G_joint += wv*loss_vessel
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (wd*loss_dice + wd*loss_ce)

                    elif supervision == 'dist':
                        with torch.no_grad():
                            gt_dis = compute_dtm(target[:, 1:].cpu().numpy(), output[1].shape)
                            gt_dis = torch.from_numpy(gt_dis).float().to(device)
                        # CE
                        out = output[0].clone() + 1e-20
                        loss_ce = ce(torch.log(out), seg)
                        # compute L1 Loss
                        loss_dist = torch.norm(output[1] - gt_dis, 1) / torch.numel(output[1])

                        loss_G_joint = loss_ce + loss_dist
                    elif supervision == 'fnr':
                        p1, p2 = ((1 - (epoch / num_epochs)) ** 0.9), (((epoch / num_epochs)) ** 0.9)
                        w = (8 / 15)
                        for p in range(4):
                            out = output[p].clone() + 1e-20
                            loss_dice = p2*soft_dice_loss_batch(out, target)
                            loss_ce = ce(torch.log(out), seg)
                            loss_vessel = p1*fnr(out,target)
                            if p == 0:
                                loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce + loss_vessel)
                            else:
                                loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce + loss_vessel)
                    elif supervision == 'ddt' or supervision == 'ddt-gar':
                        with torch.no_grad():
                            gt_dis = compute_ddt(target[:, 1:].cpu().numpy(), seg.shape, k)
                            gt_dis = torch.as_tensor(gt_dis, dtype=torch.long).to(device)
                        # CE
                        out = output[0].clone() + 1e-20
                        loss_ce = ce(torch.log(out), seg)
                        # compute CE Loss
                        out1 = output[1].clone() + 1e-20
                        loss_dist = ce(torch.log(out1), gt_dis)

                        loss_G_joint = loss_ce + loss_dist
                    elif supervision == 'radialloss':
                        with torch.no_grad():
                            gt_dis = compute_dtm(target.cpu().numpy(), target.shape)
                            gt_dis[:,0] = 0
                            gt_dis = torch.as_tensor((gt_dis), dtype=torch.long).to(device)
                        # CE
                        out = output[0].clone() + 1e-20
                        loss_dice = soft_dice_loss_old_new(out, target, gt_dis)
                        loss_ce = ce(torch.log(out), seg)
                        loss_G_joint = loss_ce + loss_dice
                    else: #example supervision == 'dense'
                        out = output[0].clone() + 1e-20
                        loss_dice = -torch.log(soft_dice_loss(out, target))
                        loss_ce = ce(torch.log(out), seg)
                        loss_G_joint = loss_ce + loss_dice

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

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 12)

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
                seg = torch.argmax(target, dim=1)

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

                if supervision == 'deep':
                    w = (8 / 15)
                    for p in range(4):
                        if log:
                            loss_dice = -torch.log(soft_dice_loss(output[p], target))
                        else:
                            loss_dice = soft_dice_loss_old(output[p], target)
                        loss_ce = ce(torch.log(output[p] + 1e-20), seg)
                        if p == 0:
                            loss_G_joint = w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                        else:
                            loss_G_joint += w * (1 / (2 ** p)) * (loss_dice + loss_ce)
                else:
                    out = output[0].clone() + 1e-20
                    loss_dice = -torch.log(soft_dice_loss(out, target))
                    loss_bce = ce(torch.log(out), seg)
                    loss_G_joint = loss_bce + loss_dice

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
                pred_t = torch.argmax(output[0], dim=1, keepdim=True)
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
            if supervision == 'dist' or supervision == 'dist2' or supervision == 'ddt' or supervision == 'ddt-gar':
                c = loss_ce.item()
                d = loss_dist.item()
                del loss_G_joint, loss_ce, loss_dist
            elif supervision[0:10]=='deepvessel':
                c = loss_ce.item()
                d = loss_dice.item()
                v = wv * (1/(channel_dim-1)) * loss_vessel.item()
                del loss_G_joint, loss_ce, loss_dice, loss_vessel
            elif supervision[0:2]=='ce':
                c = loss_ce.item()
                del loss_G_joint, loss_ce
            else:
                c = loss_ce.item()
                d = loss_dice.item()
                del loss_G_joint, loss_ce, loss_dice
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

        if supervision[0:10]=='deepvessel' or supervision=='fnr':
            print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f (ce: \t%.4f + d: \t%.4f  v: \t%.4f)'
                  % ((epoch + 1), num_epochs, G_losses[0], G_losses[-1], c, d, v))
        elif supervision[0:2]=='ce':
            print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f (ce: \t%.4f)'
                  % ((epoch + 1), num_epochs, G_losses[0], G_losses[-1], c))      
        else:
            print('[%d/%d]\tLoss_G --> first batch:\t%.4f\t and last batch: \t%.4f (ce: \t%.4f + d: \t%.4f)'
              % ((epoch + 1), num_epochs, G_losses[0], G_losses[-1], c, d))
        print('TRAINING (BEST AT 0) --> Average of batches of\t Loss_G: %.4f (of which Loss_V: %.4f)'
              % (temp_meanG, np.mean(V_losses)))

        del G_losses,V_losses

        print('starting validation...')
        torch.cuda.synchronize()
        valstep = time.time()
        if valsw:
            dices, hds = val(input_folder, patch_size, batch_size, workers, net, channel_dim, in_c, val_label_path, val_data_path,
         data_results, massimo, minimo, tta, res, preprocessing, device, epoch, do_seg=True)
            mdice = np.mean(dices)
            dices_s = np.mean(dices, axis=0)
        else:
            net.eval()
            dices = []
            dices_s = np.zeros((channel_dim - 1))
            if supervision[0:10]=='deepvessel' or supervision=='fnr':
                vessels = []
                vessels_s = np.zeros((channel_dim - 1))

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
                torch.cuda.synchronize()    
                valstep1 = time.time()
                #print("Loading time (s): ", valstep1 - valstep0)
                if learning == 'autocast':
                    with amp2.autocast():
                        with torch.no_grad():
                            predict = net(val_image)
                else:
                    with torch.no_grad():
                        predict = net(val_image)
                pred = predict[0]
                torch.cuda.synchronize()    
                valstep2 = time.time()
                #print("Inference time (s): ", valstep2 - valstep1)
                val_dice = 1 - soft_dice_loss_batch(pred, val_target)
                dices.append(val_dice.item())
                
                if supervision=='fnr':
                    val_vessel = fnr(pred,val_target)
                    vessels.append(val_vessel.item())
                    for k in range(1, channel_dim):
                        val_v = fnr(pred[batch // 2:batch, k, :, :, :],
                                                  val_target[batch // 2:batch, k, :, :, :],channel=True)
                        vessels_s[k - 1] += val_v.item()
                        val_s = 1 - soft_dice_loss_batch(pred[batch // 2:batch, k, :, :, :],
                                                  val_target[batch // 2:batch, k, :, :, :],channel=True)
                        dices_s[k - 1] += val_s.item()
                    
                elif supervision == 'deepvesselmsloss':
                    for v in range(1, channel_dim):
                        ww = torch.sum(val_target[:,v],dim=(1,2,3)) > 0
                        wt = ww.type(torch.uint8).detach()
                        val_eigenv_tr = vesselness_true_nogt((val_target[:,v]).type(torch.float16), wt)
                        val_eigenv_true = val_eigenv_tr.detach()
                        if v == 1:
                            val_vessel = vesselness_nogt(pred[:, v], wt, val_eigenv_true)
                        else:
                            val_vessel += vesselness_nogt(pred[:, v], wt, val_eigenv_true)
                    vessels.append(wv*(1/(channel_dim-1))*val_vessel.item())
                elif supervision == 'deepvesseltsloss':
                    for v in range(1, channel_dim):
                        ww = torch.sum(val_target[:,v],dim=(1,2,3)) > 0
                        wt = ww.type(torch.uint8).detach()
                        val_eigenv_tr = vesselness_true_nogt((val_target[:,v]).type(torch.float16), wt)
                        val_eigenv_true = val_eigenv_tr.detach()
                        if v == 1:
                            val_vessel = vesselness_nogt(pred[:, v], wt, val_eigenv_true, sigma=1)
                            val_vessel_self = vesselness_self_frangi(pred[:, v], val_target[:,v], sigma=1)
                        else:
                            val_vessel += vesselness_nogt(pred[:, v], wt, val_eigenv_true, sigma=1)
                            val_vessel_self += vesselness_self_frangi(pred[:, v], val_target[:,v], sigma=1)
                    vessels.append(wv*(1/(channel_dim-1))*val_vessel.item()+ (1/(channel_dim-1))*val_vessel_self.item())
                elif supervision == 'deepvesselfvloss':
                    for v in range(1, channel_dim):
                        if v == 1:
                            val_vessel_self = (1/(channel_dim-1))*vesselness_self_frangi(pred[:, v], val_target[:,v], sigma=1)
                        else:
                            val_vessel_self += (1/(channel_dim-1))*vesselness_self_frangi(pred[:, v], val_target[:,v], sigma=1)
                    vessels.append(val_vessel_self.item())
                elif supervision == 'deepvesselgtfrangissvmd':
                    kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
                    gt_dil = torch.clamp(
                                    torch.nn.functional.conv3d(val_target.type(torch.float16), kernel, padding=1, groups=channel_dim), 0, 1)
                    for v in range(1, channel_dim):
                        gt = gt_dil[:, v, :, :, :]
                        gt_dilate = gt.detach()
                        val_eigenv_tr = vesselness_true((val_target[:,v]).type(torch.float16), gt_dilate, sigma = 3)
                        val_eigenv_true = val_eigenv_tr.detach()
                        if v == 1:
                            val_vessel = vesselness_frangi_ssvmd(pred[:, v], gt_dilate, val_eigenv_true, sigma=3)
                        else:
                            val_vessel += vesselness_frangi_ssvmd(pred[:, v], gt_dilate, val_eigenv_true, sigma=3)
                    vessels.append(wv*val_vessel.item())
                elif supervision == 'deepvesselgtjermanssvmd':
                    kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
                    gt_dil = torch.clamp(
                                    torch.nn.functional.conv3d(val_target.type(torch.float16), kernel, padding=1, groups=channel_dim), 0, 1)
                    for v in range(1, channel_dim):
                        gt = gt_dil[:, v, :, :, :]
                        gt_dilate = gt.detach()
                        val_eigenv_tr = vesselness_true((val_target[:,v]).type(torch.float16), gt_dilate, sigma = 3)
                        val_eigenv_true = val_eigenv_tr.detach()
                        if v == 1:
                            val_vessel = vesselness_jerman_ssvmd(pred[:, v], gt_dilate, val_eigenv_true, sigma=3)
                        else:
                            val_vessel += vesselness_jerman_ssvmd(pred[:, v], gt_dilate, val_eigenv_true, sigma=3)
                    vessels.append(wv*val_vessel.item())
                elif supervision[0:12] == 'deepvesselgt':
                    kernel = torch.ones((channel_dim, 1, 3, 3, 3), dtype=torch.float16).to("cuda")
                    gt_dil = torch.clamp(
                                    torch.nn.functional.conv3d(val_target.type(torch.float16), kernel, padding=1, groups=channel_dim), 0, 1)
                    for v in range(1, channel_dim):
                        gt = gt_dil[:, v, :, :, :]
                        gt_dilate = gt.detach()
                        val_eigenv_tr = vesselness_true((val_target[:,v]).type(torch.float16), gt_dilate,sigma=25)
                        val_eigenv_true = val_eigenv_tr.detach()
                        if v == 1:
                            val_vessel = vesselness_gt(pred[:, v], gt_dilate, val_eigenv_true,sigma=25)
                        else:
                            val_vessel += vesselness_gt(pred[:, v], gt_dilate, val_eigenv_true,sigma=25)
                    vessels.append(wv*(1/(channel_dim-1))*val_vessel.item())
                else:
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

        net.train()
        valdices.append(1 - mdice.item())
        print('Dice Loss: ',mdice)
        for k in range(1, channel_dim):
            vals[epoch, k - 1] = 1 - dices_s[k - 1].item()
            print('Dice score (BEST AT 1) Structure' + str(k) + ': %.4f'
                  % (dices_s[k - 1]))
            np.savetxt('vals.csv', vals, fmt='%d', delimiter=',')
            
        if supervision[0:10]=='deepvessel'  or supervision=='fnr':
            vessels_s /= batch_val
            vsloss = np.mean(vessels, axis=0)           
            print('Vessel Loss: ',vsloss)
            valvs.append(vsloss)
            valtotal.append(1 - mdice.item() + vsloss)
            mdice = mdice + (3-vsloss)          
            for k in range(1, channel_dim):
                valv[epoch, k - 1] = vessels_s[k - 1].item()
                print('Vessel score (BEST AT 0) Structure' + str(k) + ': %.4f'
                      % (vessels_s[k - 1]))
                np.savetxt('valv.csv', valv, fmt='%d', delimiter=',')
        else:
            pass

        y_val.append(epoch)
        print('VALIDATION (BEST AT 1) -->Average of batches of validation loss: %.4f'
              % (mdice))




        if mdice > best_dice:
            best_dice = mdice
            torch.save(net.state_dict(), data_results + '/net_3d.pth')
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
    if supervision[0:10]=='deepvessel' or supervision=='fnr':
        plt.plot(valtotal, label="Validation Total Loss (best at 0)")
        plt.plot(valvs, label="Validation Vesselness Loss (best at 0)")    
    #for p in range(channel_dim - 1):
    #    plt.plot((vals[:, p]), label="Val Dice Structure " + str(p + 1))
    #    if supervision[0:10]=='deepvessel' or supervision=='fnr':
    #        plt.plot((valv[:, p]), label="Val Vessel Structure " + str(p + 1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(data_results + '/train_and_val_net_3d_plot.png')

    if supervision[0:10]=='deepvessel' or supervision=='fnr':
        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation VsLoss")
        plt.plot(vloss, label="Training Vesselness Loss (best at 0)")
        plt.plot(valvs, label="Validation Vesselness Loss (best at 0)")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(data_results + '/train_and_val_vsloss_net_3d_plot.png')

    if supervision[0:10]=='deepvessel':
            np.savez(data_results + '/training_alpha_' + str(wv) + '.npz', name1=mloss, name2=valdices, name3=valvs, name4=vloss)
    else:
            np.savez(data_results + '/training.npz', name1=mloss, name2=valdices)
