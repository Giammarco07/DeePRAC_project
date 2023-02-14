from argparse import ArgumentParser
import model_wsgan3 as md
import test_new as tst
import time
import neptune
import torch
from torch import nn
# To get arguments from commandlineS
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--lr_g', type=float, default=.0002)
    parser.add_argument('--wsgan',type=bool,default=False)
    parser.add_argument('--load_height', type=int, default=286)
    parser.add_argument('--load_width', type=int, default=286)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=512)
    parser.add_argument('--crop_width', type=int, default=512)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results/test-final-paired-new')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/test-final-paired-new')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--dropout', type=bool, default=True, help=' dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    args = parser.parse_args()
    return args
    
args = get_args()
converted_dict = vars(args)
neptune.init('francesco.maso/giammarco', api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNDk3MWJlMDUtNWE5ZC00MTlhLWEwNzAtZDY3M2UzNWMyZTRjIn0=')
neptune.create_experiment(name='training-test-final-paired-new', params = converted_dict)

def main():
  args = get_args()
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Device:', device)
  torch.cuda.empty_cache()
  #create_link(args.dataset_dir)

  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0: #imposte --gpu_ids=-1,force to use only the CPU(you have to change also the model.py)
      args.gpu_ids.append(id)
  print('dropout for the generators is: ' + str(args.dropout))
  if args.training:
      print("Training")
      model = md.cycleGANv1_abonly(args)
      model.train(args)
  if args.testing:
      print("Testing")
      tst.test_old(args)


if __name__ == '__main__':
    class SequenceWise(nn.Module):
        def __init__(self, module):
            """
            Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
            Allows handling of variable sequence lengths and minibatch sizes.
            :param module: Module to apply input to.
            """
            super(SequenceWise, self).__init__()
            self.module = torch.nn.Sequential(*(list(module.children())[:-4]))
            self.lin = nn.Conv2d(128, 1, 1, stride=1)

        def forward(self, x):
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, 1, x.size(2), x.size(3)).repeat(1, 3, 1, 1)
            x = self.module(x)
            x1 = self.lin(x)
            x = x1.mean([2, 3])
            x = x.view(t, n, -1)
            return x

        def __repr__(self):
            tmpstr = self.__class__.__name__ + ' (\n'
            tmpstr += self.module.__repr__()
            tmpstr += ')'
            return tmpstr
    start=time.time()
    main()
    end=time.time()
    hours,rem=divmod(end-start, 3600)
    minutes,seconds=divmod(rem,60)
    print('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
    neptune.log_metric('time',hours,minutes,seconds)
