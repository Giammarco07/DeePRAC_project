import Train_Networks, Test_Networks

def Training(path, network,nets,optimizers,learning,lr,channel_dim, train_loader, val_loader,patch_size, start_epoch,num_epochs, data_results,  massimo, minimo, supervision,val_step, best_dice, best_dicestn,best_dicestn1,best_dicestn2, early_stopping, device, asnnunet, mode = None):
    #if mode='debug' you can print time and memory for each step during training


    if network == 'nnunet3D':
        Train_Networks.nnunet3d.train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, lr,
                       channel_dim, train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnnunet, mode=None, log=False)
    elif network == 'nnunet2.5D':
        Train_Networks.nnunet25d.train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, lr, channel_dim,
                       train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnnunet, mode=None, log=False)
    elif network == 'nnunet2D':
        Train_Networks.nnunet2d.train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, lr, channel_dim,
                       train_loader, val_loader, data_results, massimo, minimo, supervision, device, asnnunet, mode=None, log=False)

    elif network == 'pnet3D':
        Train_Networks.pnet3d.train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers,
                                    learning, lr,
                                    train_loader, val_loader, data_results, massimo, minimo, supervision, device,
                                    asnnunet, mode=None, log=False)

    elif network == 'stnpose':
        Train_Networks.stn.train_stnpose(learning, path, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                     early_stopping, nets, optimizers, lr, channel_dim, train_loader, val_loader, patch_size,
                                     data_results, massimo, minimo, supervision, device, asnnunet, mode=None)
    elif network == 'stnposennunet2D':
        Train_Networks.nnunet2d.train_stnpose(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets,
                                              optimizers, learning, lr, channel_dim,
                                              train_loader, val_loader, patch_size, data_results, massimo, minimo,
                                              supervision, device, asnnunet, mode=None, log=False)


    elif network == 'stncrop':
        Train_Networks.stn_crop.train_stncrop(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)
    elif network == 'stnposecrop':
        Train_Networks.stn_crop.train_stnpose_stncrop(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)
    elif network == 'faster-rcnn':
        Train_Networks.stn_crop.train_faster_rcnn(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                                  early_stopping, nets, lr, channel_dim,
                                                  train_loader, val_loader, data_results, massimo, minimo, supervision,
                                                  device, mode)
    elif network == 'stnposefaster-rcnn':
        Train_Networks.stn_crop.train_stnpose_faster_rcnn(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)
    elif network=='stnposecropnnunet2D':
        Train_Networks.nnunet2d.train_stnposecrop(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets, optimizers, learning, lr, channel_dim,
                       train_loader, val_loader, patch_size, data_results, massimo, minimo, supervision, device, asnnunet, mode=None, log=False)



    elif network == 'stnpose3D':
        Train_Networks.stn.train_stnpose3d(learning, path, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                     early_stopping, nets, optimizers, lr, channel_dim, train_loader, val_loader, patch_size,
                                     data_results, massimo, minimo, supervision, device, asnnunet, mode=None)
    elif network == 'stncrop3D':
        Train_Networks.stn_crop.train_stncrop3d(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                                early_stopping, nets, lr, channel_dim,
                                                train_loader, val_loader, data_results, massimo, minimo, supervision,
                                                device, mode)
    elif network == 'nnDet':
        Train_Networks.stn_crop.train_nnDet(learning, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                            early_stopping, nets, lr, channel_dim,
                                            train_loader, val_loader, data_results, massimo, minimo, supervision,
                                            device, mode)
    elif network == 'stnposecrop3D':
        Train_Networks.stn_crop.train_stnposecrop3d(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)


    elif network == 'stnpose-nnunet2D':
        Train_Networks.stn_nnunet2d.train_stnpose_nnunet2d(learning, path, start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, optimizers, lr, channel_dim,
                                   train_loader, val_loader, patch_size, data_results, massimo, minimo, supervision, device, asnnunet, mode=None, log=False)

    elif network == 'stncrop-nnunet2D':
        Train_Networks.stn_nnunet2d.train_stncrop_nnunet2d(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)

    elif network == 'stncrop3-nnunet2D':
        Train_Networks.stn_nnunet2d.train_stncrop_3_nnunet2d(learning,start_epoch, num_epochs, best_dice, best_dicestn, val_step, early_stopping, nets, lr, channel_dim,
                                   train_loader, val_loader, data_results, massimo, minimo, supervision, device, mode)

    elif network == 'stnpose-nnunet3D':
        Train_Networks.stn_nnunet3d.train_stnpose_nnunet3d(learning, path, start_epoch, num_epochs, best_dice, best_dicestn, val_step,
                                     early_stopping, nets, optimizers, lr, channel_dim, train_loader, val_loader, patch_size,
                                     data_results, massimo, minimo, supervision, device, mode=None)
    else: #network =='stnpose-stncrop-nnunet2D':
        Train_Networks.stn_nnunet2d.train_stnpose_stncrop_nnunet2d(learning, path, start_epoch, num_epochs, best_dice,best_dicestn1,best_dicestn2, val_step, early_stopping, nets, lr,
                                           channel_dim, train_loader, val_loader, patch_size, data_results, massimo, minimo, supervision,
                                           device, mode)

def Training_new(network,nets,optimizers,learning,lr,channel_dim, train_loader, val_loader,patch_size, start_epoch,num_epochs, data_results,  massimo, minimo, supervision,val_step, best_dice, early_stopping, device, asnnunet, input_folder, batch_size, workers, in_c, val_label_path, val_data_path,
                      tta, res, preprocessing):
    #if mode='debug' you can print time and memory for each step during training
    if network == 'nnunet3D'or network=='redcnn':
        Train_Networks.nnunet3d.train(start_epoch, num_epochs, best_dice, val_step, early_stopping, nets,
                                          optimizers, learning, lr,
                                          channel_dim, train_loader, val_loader, data_results, massimo, minimo,
                                          supervision, device, asnnunet, mode=None, log=False, valsw=True, others = [input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, val_label_path, val_data_path, data_results, massimo, minimo, tta, res, preprocessing, device])


def Testing(task,input_folder,  patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, rsz, norm, skel, supervision):

    if network == 'nnunet3D' or network == 'stnpose-nnunet3D':
      if task == 'Task208_NECKER' and skel:
            Test_Networks.nnunet3d.test_skel(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz, norm)
      else:
          Test_Networks.nnunet3d.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz, supervision, do_seg=False)
    elif network == 'pnet3D':
        Test_Networks.pnet3d.test(input_folder, patch_size, batch_size, workers, network, nets, label_path,
                                  test_data_path, data_results, massimo, minimo, tta, res, preprocessing, device, rsz)
    elif network == 'p-nnunet3D':
        Test_Networks.pnnunet3d.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=False)


    elif network == 'stnpose':
        Test_Networks.stnpose.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim,  in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=False)
    elif network=='stnposennunet2D':
        Test_Networks.nnunet2d.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=True, rsz= True)

    elif network == 'stncrop':
        Test_Networks.stncrop3d.test2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=True)
    elif network == 'stnposecrop':
        Test_Networks.stncrop3d.test_pose_2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=True)
    elif network == 'faster-rcnn':
        Test_Networks.stncrop3d.test_faster2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=True)
    elif network == 'stnposefaster-rcnn':
        Test_Networks.stncrop3d.test_pose_faster2D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=True)
    elif network=='stnposecropnnunet2D':
        Test_Networks.nnunet2d.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=True, rsz= True)

    elif network == 'stnpose3D':
        Test_Networks.stnpose.test3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim,  in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=False)
    elif network=='stncrop3D':
        Test_Networks.stncrop3d.test3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim,  in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=True)
    elif network=='stnposecrop3D':
        Test_Networks.stncrop3d.test_pose3D(input_folder, patch_size, batch_size, workers, network, nets, channel_dim,  in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, rsz, do_seg=True)
    elif network == 'nnDet':
        Test_Networks.stncrop3d.test_nnDet(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c,
                                     label_path, test_data_path, data_results, massimo, minimo, tta, res, preprocessing,
                                     device, rsz, do_seg=True)

    elif network == 'nnunet2.5D':
        Test_Networks.nnunet25d.test_25d(nets, channel_dim, label_path, test_data_path,
                data_results, massimo, minimo, tta, res, preprocessing, device)
    else: #all the others
        Test_Networks.nnunet2d.test(input_folder, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, do_seg=True, rsz= False)


