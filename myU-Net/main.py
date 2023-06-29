import argparse
import os
import pickle
import torch.utils.data
from pathlib import Path
import numpy as np
import torchvision
from Networks import UNet2D, UNet3D,STN_CROP, STN, VNet3D, STN3D, ResNet3D, Modified_VGG16, PNet3D, REDCNN, TopNet, DVAE
from utils.Loader import get_image_file, get_image_file_2, get_image_file_3, get_image_file_4
from Dataset import Prepare_Dataset, Prepare_Dataset_train, Prepare_Dataset_val
from Train_and_Test import Training, Testing, Training_new

#region Parser
parser = argparse.ArgumentParser()
parser.add_argument("network", help="Choose between: 'nnunet3D','nnunet2D', 'nnunet2.5D', 'stnpose-nnunet2D', 'stncrop-nnunet2D', 'stnpose-stncrop-nnunet2D', 'stnpose-nnunet3D', 'pnet3D' and others.")
parser.add_argument("mode", help="Choose between: 'train' or 'test'. In 'train' mode is included 'test' mode (inference).")
parser.add_argument("batchsize", help="Add batchsize as integer number.")
parser.add_argument("input_folder", help="Select your task folder.")
parser.add_argument("results_folder", help="Select your results folder. If it does not exist, it will be created.")

parser.add_argument("--multiorgans", required=False, default=2,
                    help="Choose but do not count Background. Default = 2")
parser.add_argument("--lr", required=False, default=0.01,
                    help="Define learning rate. Default lr = 0.01")
parser.add_argument("--num_epochs", required=False, default=1000,
                    help="Define number of epochs. Default = 1000")
parser.add_argument("--ngpu", required=False, default=1,
                    help="Number of GPUs available. Use 0 for CPU mode. Default=1")
parser.add_argument("--workers",  required=False, default=8,
                    help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default = 8")
parser.add_argument("--fold",  required=False, default=1,
                    help="LOVO method")
parser.add_argument("--min_pix",  required=False, default=0,
                    help="Number of minimum pixels/voxels in a patch with foreground")
parser.add_argument("--deepsup",  required=False, default='deepbatch',
                    help="deep supervision: fast convergence, better results, more memory allocated. It is only for 'nnunet' networks or '-nnunet' part of networks.")

parser.add_argument("--no_fp16",  action="store_true",
                    help="Use if you don't want to use mixed precision (fp16) in the net --> fp16: less memory, speed up training, slighty worst performance. It is set automatically to false for any 'stn' networks.")
parser.add_argument("--no_dda",  action="store_true",
                    help="Use if you don't want to use data augmentation in the net --> data augmentation: highly better results, more training time, no early stopping. It is set automatically to false for any 'stn' networks.")
parser.add_argument("--tta",  action="store_true",
                    help="Use if you want to use test time augmentation in the inference --> test time augmentation: possibles better results, not sure, more test time. Not suggested")

parser.add_argument("-c", "--continue_training", help="Use this if you want to continue a training. It is only for 'nnunet' networks or '-nnunet' part of networks.",
                    action="store_true")

parser.add_argument("-tl", "--transfer_learning", help="Use this if you want to do a trasnfer learning for train or test from nnunet.model.",
                    action="store_true")

parser.add_argument("-ts", "--transferskel", help="Use this if you want to do a self-transfer learning for train from nnunet skeleton.",
                    action="store_true")

parser.add_argument("-rsz", "--resize", help="Use this if you want to resize original image to 128x128x128 or 128x128.",
                    action="store_true")

parser.add_argument("-asnn", "--asnnunet", help="250 iterations and 1000 epochs",
                    action="store_true")

parser.add_argument("-all", "--all_together", help="children+adults for training",
                    action="store_true")
                    
parser.add_argument("-norm", "--normalization", help="my normalization",
                    action="store_true")
                    
parser.add_argument("-skel", "--skeleton", help="patch on skeleton",
                    action="store_true")

parser.add_argument("-tot", "--total", help="training without oversampling",
                    action="store_true")

parser.add_argument("-valsw", "--valsw", help="validation slidingwindow training",
                    action="store_true")
                    
parser.add_argument("-new", "--new", help="validation slidingwindow training",
                    action="store_true")             

print('WELCOME')

print("Use 'python main.py -h' to discover all the options.")
#endregion

#region Parameters initialization
args = parser.parse_args()
network = args.network
mode = args.mode
batch_size = int(args.batchsize)
lr = float(args.lr)
epochs = int(args.num_epochs)
min_pix = int(args.min_pix)
#path = str(os.getcwd())
path = '/home/infres/glabarbera/nnunet'

data_results = args.results_folder
if os.path.exists(data_results)==False:
    print('WARNING! This folder does not exist. It will be created.')
    os.mkdir(data_results)

channel_dim = int(args.multiorgans) + 1
skel = args.skeleton
if skel:
  print('WARNING! Skeleton mode.')

device = torch.device("cuda" if (torch.cuda.is_available() and int(args.ngpu) > 0) else "cpu")
print('Device:', device)
workers = int(args.workers)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.no_dda or network[0:3]=='stn':
    dda = False
else:
    dda = True
print('dda: ', dda)

if args.no_fp16:
    learning ='normal'
else:
    learning = 'autocast'
print('learning: ', learning)

supervision = args.deepsup
print('supervision: ', supervision)

rsz = args.resize
if rsz==True or network[:3]=='stn' :
    rsz = True
    print('WARNING! Original image will be resize')

tta = args.tta
asnnunet = args.asnnunet
norm = args.normalization
if norm:
  print('WARNING! Normalization using skeleton points will be applied.')
#endregion

#region Voxel res and preprocessing values manually extracted from nnU-Net preprocessing pkl files
if args.input_folder == 'Task201_KiTS19':
        res = [0.79394531, 0.79394531, 0.79394531]
        preprocessing = [99.9, 77.71, 303.0, -78.0]
        input = 'adults'
elif args.input_folder == 'Task204_CtMO20':
        res = [0.77636731, 0.77636731, 1.0]
        preprocessing = [-194.40, 553.29, 1200.0, -968.0]  # mean,sd,p95,p05
        input = 'adults'
elif args.input_folder == 'Task200_NECKER' or args.input_folder == 'Task300_NECKER' or args.input_folder == 'Task301_NECKER':
        preprocessing = [76.99, 67.73, 303.0, -36.0]  # mean,sd,p95,p05
        input = 'children'
elif args.input_folder =='Task210_NECKER':
        preprocessing = [84.12, 71.57, 303.0, -58.0]  # mean,sd,p95,p05
        input = 'children'
elif args.input_folder == 'Task202_NECKER' or args.input_folder == 'Task302_NECKER':
        preprocessing = [1252.05, 904.98, 4363.0, 147.0] #mean,sd,p95,p05
        input = 'children'
elif args.input_folder == 'Task202_NECKER_bis':
        res = [0.47265601, 0.47265601, 0.79992676]
        preprocessing = [1252.05, 904.98, 4363.0, 147.0] #mean,sd,p95,p05
        input = 'adults'
elif args.input_folder == 'Task203_NECKER':
        preprocessing = [78.44, 85.35, 315.0, -36.0] #mean,sd,p95,p05
        input = 'children'
elif args.input_folder == 'Task205_NECKER':
    preprocessing = [164.16, 72.35, 377.99, -23.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task206_NECKER':
    preprocessing = [77.02, 67.69, 303.0, -36.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task207_CtAb20':
    res = [0.77636731, 0.77636731, 1.0]
    preprocessing = [222.24, 314.44, 1351.0, -71.0]  # mean,sd,p95,p05
    input = 'adults'
elif args.input_folder == 'Task208_NECKER' or args.input_folder == 'Task308_NECKER' or args.input_folder == 'Task309_NECKER':
    preprocessing = [188.93, 175.44, 1034.5, -3.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task209_NECKER':
    preprocessing = [205.96, 209.74, 1847.66, 31.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task304_CtAb20' or args.input_folder == 'Task305_CtAb20' or args.input_folder == 'Task404_CtAb20':
    res = [0.77636731, 0.77636731, 1.0]
    preprocessing = [114.26, 63.71, 294.0, -94.0]  # mean,sd,p95,p05
    input = 'adults'
elif args.input_folder == 'Task408_NECKER':
    preprocessing = [175.91, 100.48, 631.98, -12.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task408_NECKER' and args.new:
    preprocessing = [-39.0, 70.0, 303.0, -70.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task508_NECKER':
    preprocessing = [106.34, 104.48, 391.0, -69.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task608_NECKER':
    #preprocessing = [[106.34, 104.48, 391.0, -69.0],[106.34, 104.48, 391.0, -69.0]]  # mean,sd,p95,p05
    #preprocessing = [[175.91, 100.48, 631.98, -12.0],[36.03, 43.69, 112.0, -98.40]]  # mean,sd,p95,p05
    #preprocessing = [[175.91, 100.48, 391.0, -69.0],[36.03, 43.69, 391.0, -69.0]]  # mean,sd,p95,p05
    preprocessing = [[-39.0, 70.0, 303.0, -70.0],[-39.0, 70.0, 303.0, -70.0]]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task708_NECKER':
    preprocessing = [36.03, 43.69, 112.0, -98.0]  # mean,sd,p95,p05
    #preprocessing = [106.34, 104.48, 391.0, -69.0]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task808_NECKER' or args.input_folder == 'Task818_NECKER' or args.input_folder == 'Task908_NECKER' or args.input_folder == 'Task918_NECKER':
    preprocessing = [[0, 1, 4.75, -0.46],[0, 1, 4.75, -0.46]]  # mean,sd,p95,p05
    input = 'children'
elif args.input_folder == 'Task828_NECKER':
    #preprocessing = [0, 1, 4.75, -0.46]  # mean,sd,p95,p05
    preprocessing = [188.93, 175.44, 1034.5, -3.0]
    input = 'children'
task = args.input_folder
print(task)
#endregion

#region Folders and patch_size definition
if input =='children' and (network=='nnunet3D' or network=='pnet3D' or network=='p-nnunet3D' or network=='TopNet' or network=='DVAE'):
    if mode=='train':
      if (task == 'Task208_NECKER' or task == 'Task308_NECKER' or task == 'Task202_NECKER' or task == 'Task302_NECKER') and skel:
        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches_perp_3D2_new')
      elif (task == 'Task208_NECKER' or task == 'Task308_NECKER' or task == 'Task202_NECKER' or task == 'Task302_NECKER') and not skel and not args.transferskel and network=='pnet3D':
      	data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches64_2_all_new')
      elif (task == 'Task208_NECKER' or task == 'Task209_NECKER' or task == 'Task202_NECKER' or task == 'Task302_NECKER') and not skel and not args.transferskel:
      	data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches160') #160') #64_2')
      elif task == 'Task308_NECKER' and not skel and not args.transferskel:
      	data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1_affine/Patches') #160') #64_2')
      elif (task == 'Task208_NECKER' or task == 'Task308_NECKER' or task == 'Task202_NECKER' or task == 'Task302_NECKER') and not skel and args.transferskel:
          print("ATTENTION! Training network for self-transfer learning")
          data_path_skel = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches_perp_3D2_new')
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches64_2_new')
      elif task == 'Task508_NECKER' and not skel and args.transferskel and not args.new:
          data_path_skel = Path(path + '/nnUNet_preprocessed/Task408_NECKER/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold)) 
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold))
      elif task == 'Task408_NECKER' and not skel and args.transferskel and not args.new:
          data_path_skel = Path(path + '/nnUNet_preprocessed/Task408_NECKER/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold)) 
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patchesnewrandom/fold'+str(args.fold))                    
      elif task == 'Task508_NECKER' and not skel and args.transferskel and args.new:
          data_path_skel = Path(path + '/nnUNet_preprocessed/Task408_NECKER/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold)) 
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches_cyclegan512/fold'+str(args.fold)) 
      elif task == 'Task408_NECKER' and not skel and args.transferskel and args.new:
          data_path_skel = Path(path + '/nnUNet_preprocessed/Task408_NECKER/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold)) 
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patchesnewrandomcgan/fold'+str(args.fold))        
      elif (task == 'Task408_NECKER' or task == 'Task508_NECKER' or task == 'Task608_NECKER' or task == 'Task708_NECKER' or task == 'Task808_NECKER' or task == 'Task818_NECKER'  or task == 'Task908_NECKER' or task == 'Task918_NECKER'):
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patchesnew/fold'+str(args.fold))   
      elif task == 'Task828_NECKER':
          data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches/fold'+str(args.fold))                  
      else:
        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches') #Patches2

    res = [0.45703101, 0.45703101, 0.8999939]
    patch_size = np.array((96,160,160))
    
    if task == 'Task408_NECKER' or task == 'Task508_NECKER' or task == 'Task608_NECKER' or task == 'Task708_NECKER' or task == 'Task808_NECKER' or task == 'Task818_NECKER' or task == 'Task908_NECKER' or task == 'Task918_NECKER' or task == 'Task828_NECKER' :
      label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs/fold'+str(args.fold)
      test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs/fold'+str(args.fold)
    else:
      label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
      test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'


elif input == 'children' and (network == 'stnpose-nnunet3D' or network == 'stnpose3D'or network == 'stncrop3D'or network == 'stnposecrop3D'):
    if mode=='train':
        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches2')
    res = [0.45703101, 0.45703101, 0.8999939]
    patch_size = np.array((128,128,128))
    label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
    test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'

#elif input == 'children' and network == 'stncrop3D':
#    if mode=='train':
#        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/')
#    res = [0.45703101, 0.45703101, 0.8999939]
#    patch_size = np.array((128,128,128))
#    label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
#    test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'

elif input =='children' and network!='nnunet3D':
    if network!='nnunet2.5D':
            data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_2D_stage0/Slices')
            patch_size = np.array((512, 512))
    else:
      if task == 'Task208_NECKER':
         data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches_perp_25')
      else:
         data_path = Path(path + '/nnUNet_preprocessed/' + task + '/nnUNetData_plans_v2.1_stage1/Slices')
      patch_size = np.array((64, 64))

    res = [0.45703101, 0.45703101, 0.8999939]
    if args.all_together:
        res = [2.5 , 0.79882812, 0.73828125] #Task210

    label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
    test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'

elif input =='adults' and network=='nnunet3D':
    if mode == 'train':
        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_stage1/Patches')
    patch_size = np.array((128,128,128))
    label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
    test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'

elif input =='adults' and network!='nnunet3D':
    res = [3.0, 0.79394531, 0.79394531]
    patch_size = np.array((512, 512))
    if mode == 'train':
        data_path = Path(path + '/nnUNet_preprocessed/'+task+'/nnUNetData_plans_v2.1_2D_stage0/Slices')
    label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/labelsTs'
    test_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/'+task+'/imagesTs'

else:
    print('WARNING! This code is not ready for this...')
    data_path = Path(path + task)

#endregion

#region Network definition
use_bias = True
k = 23 #after inspection with "Find_k_ddt.py"
if task == 'Task608_NECKER' or task == 'Task808_NECKER' or task == 'Task818_NECKER'  or task == 'Task908_NECKER' or task == 'Task918_NECKER':
    in_c = 2
    preprocessing_mean = preprocessing[0]
    massimo = (preprocessing_mean[2] - preprocessing_mean[0]) / preprocessing_mean[1]
    minimo = (preprocessing_mean[3] - preprocessing_mean[0]) / preprocessing_mean[1]
else:
    in_c = 1
    massimo = (preprocessing[2] - preprocessing[0]) / preprocessing[1]
    minimo = (preprocessing[3] - preprocessing[0]) / preprocessing[1]

if input =='children' and network=='nnunet3D':
  if task == 'Task209_NECKER':
    net = UNet3D.net_64(args.ngpu,channel_dim, use_bias=use_bias, in_c = in_c).to(device)
  else:
    if supervision=='ddt' or supervision=='ddt-gar':
        net = UNet3D.net_ddt(args.ngpu, channel_dim, k, use_bias=use_bias).to(device)
    elif supervision=='dist':
        net = UNet3D.net_dist(args.ngpu, channel_dim, use_bias=use_bias).to(device)
    elif supervision=='dense':
        net = UNet3D.net_Dense(args.ngpu, channel_dim, use_bias=use_bias).to(device)
        #net = UNet3D.net_Dense3(args.ngpu, channel_dim, use_bias=use_bias).to(device)
    else:
        net = UNet3D.net_new(args.ngpu,channel_dim, use_bias=use_bias, in_c = in_c).to(device)
elif input=='children' and network=='TopNet':
    net = TopNet.net(args.ngpu).to(device)
elif input=='children' and network=='DVAE':
    net1 = UNet3D.net_new(args.ngpu,channel_dim, use_bias=use_bias, in_c = in_c).to(device)
    net2 = DVAE.net(100).to(device)
elif input=='children' and network=='pnet3D':
    net = PNet3D.net_64(use_bias=use_bias, in_c = 1).to(device)
elif input=='children' and network=='p-nnunet3D':
    net1 = PNet3D.net_64(use_bias=use_bias, in_c = 1).to(device)
    net2 = UNet3D.net_64(args.ngpu, channel_dim, use_bias=use_bias, in_c=1).to(device)
elif input =='adults' and network=='nnunet3D':
    net = UNet3D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
    #net = ResNet3D.Deep_net(args.ngpu, channel_dim, ndf = 24).to(device)
    #net = VNet3D.net(args.ngpu, channel_dim, ndf=30, nbf = 240, use_bias=use_bias).to(device)
    #net = ResNet3D.net(args.ngpu, channel_dim, ndf=30).to(device)
elif network == 'nnunet2.5D':
    net = Modified_VGG16.net(args.ngpu,channel_dim,use_bias=use_bias).to(device)
elif network == 'nnunet2D':
    net = UNet2D.net(args.ngpu,channel_dim,use_bias=use_bias).to(device)
elif network == 'stnpose':
    net =  STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
elif network == 'stnpose-nnunet2D' or  network == 'stnposennunet2D':
    net1 = STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = UNet2D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
elif network == 'stncrop':
    net = STN_CROP.net_new(args.ngpu, 1, patch_size).to(device)
elif network == 'stnposecrop':
    net1 =  STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = STN_CROP.net(args.ngpu, 1, patch_size).to(device)
elif network == 'faster-rcnn':
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)
elif network == 'stnposefaster-rcnn':
    net1 =  STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)
elif network == 'stnposecropnnunet2D':
    net1 =  STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = torchvision.models.detection.fasterrcnn_resnet50_fpn().to(device)
    net3 = UNet2D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
elif network == 'stnpose3D':
    net = STN3D.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
elif network == 'stncrop3D':
    net = STN_CROP.net3D_new(args.ngpu, 1, patch_size).to(device)
elif network == 'stnposecrop3D':
    net1 = STN3D.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = STN_CROP.net3D_new(args.ngpu, 1, patch_size).to(device)


elif network == 'stncrop-nnunet2D':
    net1 = STN_CROP.net(args.ngpu, 1, patch_size).to(device)
    net2 = UNet2D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
elif network == 'stncrop3-nnunet2D':
    net1 = STN_CROP.net_3(args.ngpu, 1, patch_size).to(device)
    net2 = UNet2D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
elif network == 'stnpose-stncrop-nnunet2D':
    net1 = STN.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = STN_CROP.net(args.ngpu, 1, patch_size).to(device)
    net3 = UNet2D.net(args.ngpu,channel_dim,use_bias=use_bias).to(device)
elif network=='stnpose-nnunet3D':
    net1 = STN3D.net(args.ngpu, 2, patch_size, mode='rotation_translation_scale').to(device)
    net2 = UNet3D.net(args.ngpu,channel_dim, use_bias=use_bias).to(device)
else:
    print('WARNING! This network does not exist...')
    exit()
#endregion

#region Preparation or loading of paths for oversampling
if mode=='train':
    print("starting preparation of paths for oversampling...")
    if os.path.exists(data_results + '/paths_b.txt') and not args.transferskel:
        print('paths already exist and it will be loaded...')
        with open(data_results + "/paths_b.txt", "rb") as fp:  # Unpickling
            paths_b = pickle.load(fp)
        with open(data_results + "/paths_f.txt", "rb") as fp:  # Unpickling
            paths_f = pickle.load(fp)
        num = np.load(data_results + '/num.npy')
        print('DONE')
		
    elif args.transferskel:
        if os.path.exists(data_results + '/paths_bs.txt'):
            print('paths already exist and it will be loaded...')
            with open(data_results + "/paths_bs.txt", "rb") as fp:  # Unpickling
                paths_bs = pickle.load(fp)
            with open(data_results + "/paths_b.txt", "rb") as fp:  # Unpickling
                paths_b = pickle.load(fp)
            with open(data_results + "/paths_fs.txt", "rb") as fp:  # Unpickling
                paths_fs = pickle.load(fp)
            with open(data_results + "/paths_f.txt", "rb") as fp:  # Unpickling
                 paths_f = pickle.load(fp)
            num = np.load(data_results + '/num.npy')
            print('DONE')
        else:
            print('paths need to be created and it will take time...')
            paths_bs, paths_fs, num1 = get_image_file_2(data_path_skel, channel_dim, patch_size, min_pix)
            paths_b, paths_f, num2 = get_image_file_2(data_path, channel_dim, patch_size, min_pix)
            with open(data_results + "/paths_b.txt", "wb") as fp:  # Pickling
                pickle.dump(paths_b, fp)
            with open(data_results + "/paths_bs.txt", "wb") as fp:  # Pickling
                pickle.dump(paths_bs, fp)
            with open(data_results + "/paths_f.txt", "wb") as fp:  # Pickling
                pickle.dump(paths_f, fp)
            with open(data_results + "/paths_fs.txt", "wb") as fp:  # Pickling
                pickle.dump(paths_fs, fp)
            num = min(num1,num2)
            np.save(data_results + '/num.npy', np.asarray(num))
            print('DONE')
    else:
        print('paths need to be created and it will take time...')
        paths_b, paths_f, num = get_image_file_2(data_path, channel_dim, patch_size, min_pix)
        with open(data_results + "/paths_b.txt", "wb") as fp:  # Pickling
            pickle.dump(paths_b, fp)
        with open(data_results + "/paths_f.txt", "wb") as fp:  # Pickling
            pickle.dump(paths_f, fp)
        np.save(data_results + '/num.npy', np.asarray(num))
        print('DONE')
#endregion

#region Number of training epochs
if mode == 'train':
    if network == 'nnunet3D' or network== 'TopNet' or network== 'DVAE':
        n_images_seen_nnunet = 500 * epochs
    elif network == 'pnet3D':
        n_images_seen_nnunet = 1000 * epochs
    elif network == 'nnunet2.5D':
        n_images_seen_nnunet = 115 * epochs
    else:
        n_images_seen_nnunet = 3000 * epochs
    num_images_per_epoch = num * 2
    if num_images_per_epoch>=500 and asnnunet==True:
        if (network == 'nnunet3D' or network == 'pnet3D'  or network== 'TopNet'  or network== 'DVAE'):
            n_iter = 500//batch_size
            print('WARNING! Training as nnunet ', str(n_iter),' iterations with batch ', str(batch_size), 'and 1000 epochs: 500.000 iterations')
            num_epochs = epochs
            print('number of images per epoch:', n_iter * batch_size)
        elif network == 'nnunet2D' or network=='stnposennunet2D' or network == 'stnposecropnnunet2D':
            print('WARNING! Training as nnunet 250 iterations with batch 12 and 1000 epochs: 3.000.000 iterations')
            num_epochs = epochs
            print('number of images per epoch:', 250 * batch_size)
        elif network == 'nnunet2.5D':
            print('WARNING! Training as nnunet 115 iterations with batch 12 and 1000 epochs (?)')
            num_epochs = epochs
            print('number of images per epoch:', 115 * batch_size)
        else:
            print('WARNING! Training as nnunet 250 iterations with batch 12 and 2000 epochs for each network: 6.000.000 iterations')
            num_epochs = epochs*2
            print('number of images per epoch:', 250 * batch_size)
    elif network[:3] == 'stn' or network == 'faster-rcnn' or network == 'nnDet':
            num_epochs = 50
            print('number of images per epoch:', num_images_per_epoch)
    else:
        num_epochs = int(n_images_seen_nnunet / num_images_per_epoch)
        print('number of images per epoch:', num_images_per_epoch)
    print("number of epochs:", num_epochs)
else:
    num_epochs = 0
#endregion

#region Initialization or loading of training parameters
nets = []
if args.continue_training and mode=='train':
    print('WARNING! You are continuing a past training...')
    if os.path.exists(data_results + '/dice_val.npy'):
        dice_load = np.load(data_results + '/dice_val.npy')
        best_dice = dice_load[0]
    else:
        best_dice = 0
    if os.path.exists(data_results + '/last_epoch.npy'):
        start_epoch = int(np.load(data_results + '/last_epoch.npy') +1)
        if start_epoch == num_epochs:
            num_epochs += 15
    else:
        start_epoch = 0
    if os.path.exists(data_results + '/dice_valstn.npy'):
        dicestn_load = np.load(data_results + '/dice_valstn.npy')
        best_dicestn = dicestn_load[0]
    else:
        best_dicestn = 0
        if network=='stncrop3D':
                best_dicestn = -1
    if os.path.exists(data_results + '/dice_valstn1.npy'):
        dicestn_load = np.load(data_results + '/dice_valstn1.npy')
        best_dicestn1 = dicestn_load[0]
    else:
        best_dicestn1 = 0
    if os.path.exists(data_results + '/dice_valstn2.npy'):
        dicestn_load = np.load(data_results + '/dice_valstn2.npy')
        best_dicestn2 = dicestn_load[0]
    else:
        best_dicestn2 = 0

    if network=='nnunet3D' or network=='redcnn'  or network== 'TopNet':
        net.load_state_dict(torch.load(data_results + '/net_3d.pth'))
        nets.append(net)
        if dda:
            early_stopping = num_epochs
        else:
            early_stopping = int(num_epochs / 2)
        val_step = int(start_epoch / 15) + 1

    elif network=='pnet3D':
        net.load_state_dict(torch.load(data_results + '/pnet_3d.pth'))
        nets.append(net)
        if dda:
            early_stopping = num_epochs
        else:
            early_stopping = int(num_epochs / 2)
        val_step = int(start_epoch / 15) + 1

    elif network=='nnunet2D' or network=='nnunet2.5D':
        net.load_state_dict(torch.load(data_results + '/net.pth'))
        if dda:
            early_stopping = num_epochs
        else:
            early_stopping = num_epochs
        nets.append(net)
        val_step = int(start_epoch / 25) + 1

    elif network=='faster-rcnn' or network == 'stnpose3D' or network == 'stncrop3D' or network=='stncrop' or network=='nnDet':
        net.load_state_dict(torch.load(data_results + '/net_1.pth'))
        early_stopping = num_epochs + 1
        nets.append(net)
        val_step = int(start_epoch / 25) + 1

    elif network=='stnposecrop' or network=='stnposecrop3D' or network == 'stnposefaster-rcnn' or network== 'DVAE':
        net2.load_state_dict(torch.load(data_results + '/net_1.pth'))
        early_stopping = num_epochs
        nets.append(net1)
        nets.append(net2)
        val_step = int(start_epoch / 25) + 1

    elif network=='stnpose-nnunet2D' or network=='stncrop-nnunet2D'  or network=='stncrop3-nnunet2D':
        early_stopping = 2000
        num_epochs += 50
        if start_epoch<1000:
            val_step = int(start_epoch / 25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        elif start_epoch==1000:
            val_step = int(start_epoch / 25)
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        else:
            val_step = int(start_epoch/25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
            net2.load_state_dict(torch.load(data_results + '/net_2.pth'))

        nets.append(net1)
        nets.append(net2)
    elif network=='stnpose-nnunet3D':
        early_stopping = 1000
        num_epochs += 50
        if start_epoch<50:
            val_step = int(start_epoch / 25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        elif start_epoch==50:
            val_step = int(start_epoch / 25)
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        else:
            val_step = int(start_epoch/25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
            net2.load_state_dict(torch.load(data_results + '/net_2.pth'))

        nets.append(net1)
        nets.append(net2)
    elif network=='stncrop3D':
        early_stopping = 1000
        val_step = int(start_epoch / 25) + 1
        net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
        nets.append(net1)
    elif network == 'stnposennunet2D':
        early_stopping = int(num_epochs / 2)
        val_step = int(start_epoch / 25) + 1
        nets.append(net1)
        net2.load_state_dict(torch.load(data_results + '/net.pth'))
        nets.append(net2)
    elif network == 'stnposecropnnunet2D':
        early_stopping = int(num_epochs / 2)
        val_step = int(start_epoch / 25) + 1
        nets.append(net1)
        nets.append(net2)
        net3.load_state_dict(torch.load(data_results + '/net.pth'))
        nets.append(net3)
    else:
        early_stopping = 150
        num_epochs += 100
        if start_epoch<50:
            val_step = int(start_epoch / 25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        elif start_epoch==50:
            val_step = int(start_epoch / 25)
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))

        elif start_epoch<100:
            start_epoch=100
            val_step=int(start_epoch / 25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
            net2.load_state_dict(torch.load(data_results + '/net_2.pth'))

        elif start_epoch==100:
            start_epoch=100
            val_step=int(start_epoch / 25)
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
            net2.load_state_dict(torch.load(data_results + '/net_2.pth'))

        else:
            val_step = int(start_epoch / 25) + 1
            net1.load_state_dict(torch.load(data_results + '/net_1.pth'))
            net2.load_state_dict(torch.load(data_results + '/net_2.pth'))
            net3.load_state_dict(torch.load(data_results + '/net_3.pth'))

        nets.append(net1)
        nets.append(net2)
        nets.append(net3)


else:
    best_dice = 0
    best_dicestn = 0
    best_dicestn1 = 0
    best_dicestn2 = 0
    start_epoch = 0
    val_step=0
    if network=='nnunet3D' or network == 'pnet3D' or network=='redcnn'  or network== 'TopNet':
        if dda:
            early_stopping = num_epochs
        else:
            early_stopping = int(num_epochs / 2)
        nets.append(net)
    elif network=='nnunet2D'or network=='nnunet2.5D':
        if dda:
            early_stopping = num_epochs
        else:
            early_stopping = int(num_epochs / 2)
        nets.append(net)
    elif network=='faster-rcnn' or network == 'stnpose3D' or network=='stncrop' or network=='stncrop3D':
        early_stopping = num_epochs
        nets.append(net)
    elif network == 'stnposecrop' or network == 'stnposefaster-rcnn'  or network=='stnposecrop3D':
        early_stopping = num_epochs
        nets.append(net1)
        nets.append(net2)
    elif network=='stnpose-nnunet2D' or network=='stncrop-nnunet2D'  or network=='stncrop3-nnunet2D' or network=='stnpose-nnunet3D' or network=='p-nnunet3D':
        early_stopping = 100
        nets.append(net1)
        nets.append(net2)
        num_epochs += 50
    elif network == 'stnposennunet2D' or network == 'stnposecrop3D'  or network== 'DVAE':
        early_stopping = int(num_epochs / 2)
        nets.append(net1)
        nets.append(net2)
    elif network == 'stnposecropnnunet2D':
        early_stopping = int(num_epochs / 2)
        nets.append(net1)
        nets.append(net2)
        nets.append(net3)
    elif network=='stncrop3D':
        early_stopping = 500
        nets.append(net1)
        best_dicestn = -1
    else:
        early_stopping = 150
        nets.append(net1)
        nets.append(net2)
        nets.append(net3)
        num_epochs += 100
#endregion

#region Optimizer
import torch.optim as optim
if mode=='train':
    optimizers = []
    if network == 'nnunet3D' or network == 'nnunet2D' or network=='nnunet2.5D' or network=='redcnn'  or network== 'TopNet':
        lr = lr * ((1 - (start_epoch / num_epochs)) ** 0.9)
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training:
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)
        
    elif network=='pnet3D':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training:
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)

    elif network=='faster-rcnn' or network == 'stnpose3D' or network=='stncrop3D' or network=='stncrop' or network=='nnDet':
        optimizer = optim.SGD(net.parameters(), lr=lr)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training==True and os.path.exists(data_results + '/optimizer.pth'):
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)

    elif network=='stnposecrop' or network=='stnposefaster-rcnn'  or network=='stnposecrop3D':
        optimizer = optim.SGD(net2.parameters(), lr=lr)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training==True and os.path.exists(data_results + '/optimizer.pth'):
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)

    elif network=='stnposennunet2D' or network== 'DVAE':
        lr = lr * ((1 - (start_epoch / num_epochs)) ** 0.9)
        optimizer = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training:
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)
    elif network == 'stnposecropnnunet2D':
        lr = lr * ((1 - (start_epoch / num_epochs)) ** 0.9)
        optimizer = optim.SGD(net3.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training:
            optimizer.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer)
    elif network=='stnpose-nnunet2D' or network=='stncrop-nnunet2D'  or network=='stncrop3-nnunet2D' or network=='stnpose-nnunet3D':
        optimizer1 = optim.SGD(net1.parameters(), lr=lr)
        for state in optimizer1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if start_epoch >= 1000:
            lr = lr * ((1 - ((start_epoch - 1000) / (num_epochs-1000))) ** 0.9)
            optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            optimizer2 = optim.SGD(net2.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        for state in optimizer2.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training==True and start_epoch>1000:
            optimizer2.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer1)
        optimizers.append(optimizer2)

    elif network=='stncrop3D':
        optimizer1 = optim.SGD(net1.parameters(), lr=lr)
        for state in optimizer1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizers.append(optimizer1)

    else:
        optimizer1 = optim.SGD(net1.parameters(), lr=lr)
        for state in optimizer1.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        optimizer2 = optim.SGD(net2.parameters(), lr=lr)
        for state in optimizer2.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if start_epoch >= 100:
            lr = lr * ((1 - ((start_epoch - 100) / num_epochs)) ** 0.9)
            optimizer3 = optim.SGD(net3.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        else:
            optimizer3 = optim.SGD(net3.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
        for state in optimizer3.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if args.continue_training:
            optimizer3.load_state_dict(torch.load(data_results + '/optimizer.pth'))
        optimizers.append(optimizer1)
        optimizers.append(optimizer2)
        optimizers.append(optimizer3)
#endregion

#region Transfer learning
if args.transfer_learning:
    print('WARNING! TRANSFER LEARNING -- Loading weights of nnunet...')
    checkpoint = torch.load(data_results + '/model_final_checkpoint.model')
    pretrained_dict = checkpoint['state_dict']
    model_dict = nets[0].state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    nets[0].load_state_dict(model_dict)
    del checkpoint
#endregion

#region Training and/or Testing
if mode=='train':
    print('starting Prepare Dataset')
    print('Warning: minimum batch_size of 2 to have the best performances')
    if args.transferskel:
        train_loader0 = Prepare_Dataset_train(paths_b, patch_size, input, channel_dim, in_c, int(batch_size / 2),
                                                 workers, dda, massimo, minimo, network, rsz, norm)
        val_loader0 = Prepare_Dataset_val(paths_bs, patch_size, input, channel_dim, in_c, int(batch_size / 2),
                                                 workers, dda, massimo, minimo, network, rsz, norm)
        #train_loader0, val_loader0 = Prepare_Dataset(paths_b, patch_size, input, channel_dim, in_c, int(batch_size / 2),
        #                                         workers, dda, massimo, minimo, network, rsz, norm)
        train_loader1 = Prepare_Dataset_train(paths_f, patch_size, input, channel_dim,in_c,
                                                 int(batch_size / 2),workers, dda, massimo, minimo, network, rsz, norm)
        val_loader1 = Prepare_Dataset_val(paths_fs, patch_size, input, channel_dim,in_c,
                                                 int(batch_size / 2),
                                                 workers, dda, massimo, minimo, network, rsz, norm)
        train_loader = [train_loader0, train_loader1]
        val_loader = [val_loader0, val_loader1]
        print('Dataset DONE')
    elif args.valsw:
        val_label_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/' + task + '/labelsTv'
        val_data_path = path + '/nnUNet_raw_data_base/nnUNet_raw_data/' + task + '/imagesTv'
        train_loader0 = Prepare_Dataset_train(paths_b, patch_size, input, channel_dim, in_c, int(batch_size / 2),
                                                workers, dda, massimo, minimo, network, rsz, norm)
        train_loader1 = Prepare_Dataset_train(paths_f, patch_size, input, channel_dim, in_c,
                                              int(batch_size / 2), workers, dda, massimo, minimo, network, rsz, norm)
        train_loader = [train_loader0, train_loader1]
        val_loader = []
        print('Dataset DONE')
    else:
        train_loader0, val_loader0 = Prepare_Dataset(paths_b, patch_size, input, channel_dim, in_c,int(batch_size / 2),
                                                 workers, dda, massimo, minimo, network, rsz, norm)
        train_loader1, val_loader1 = Prepare_Dataset(paths_f, patch_size, input, channel_dim, in_c,int(batch_size / 2),
                                                 workers, dda, massimo, minimo, network, rsz, norm)
        if args.total:
            train_loader = [train_loader0, train_loader0]
            val_loader = [val_loader0, val_loader1]
        else:
            train_loader = [train_loader0, train_loader1]
            val_loader = [val_loader0, val_loader1]
        print('Dataset DONE')

    print('Training for ',num_epochs,' epochs.')
    if args.valsw:
        Training_new(network, nets, optimizers, learning, lr, channel_dim, train_loader, val_loader, patch_size,
                         start_epoch, num_epochs, data_results, massimo, minimo, supervision, val_step, best_dice,
                         early_stopping, device, asnnunet, input, batch_size, workers, in_c, val_label_path,
                         val_data_path,
                         tta, res, preprocessing)
    else:
        Training(path, network, nets, optimizers, learning, lr, channel_dim, train_loader, val_loader, patch_size, start_epoch,
                 num_epochs, data_results, massimo, minimo, supervision, val_step, best_dice, best_dicestn, best_dicestn1, best_dicestn2,
                 early_stopping, device, asnnunet, mode=None)
        print("Training DONE")

print('starting Test-Set Evaluation...')
Testing(task,input, patch_size, batch_size, workers, network, nets, channel_dim, in_c, label_path, test_data_path,
                      data_results, massimo, minimo, tta, res, preprocessing, device, rsz, norm, skel, supervision)
print("Test-Set DONE")
#endregion
