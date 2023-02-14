"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, get_data_loader_folder_test
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
from data import ImageFolder
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")

opts = parser.parse_args()


torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

print('Load experiment setting')
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

print('Setup model and data loader')
image_names = []
for root, _, fnames in sorted(os.walk(opts.input_folder)):
        for fname in fnames:
                path = os.path.join(root, fname)
                image_names.append(path)
print(image_names)
if opts.a2b :
  data_loader = get_data_loader_folder_test(opts.input_folder, 1, False, new_size=config['new_size_a'], crop=False, data = 'A')
else:
  data_loader = get_data_loader_folder_test(opts.input_folder, 1, False, new_size=config['new_size_a'], crop=False, data = 'B') 

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
    
print('Load model')
try:
    print('Load checkpoint')
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
    print('DONE')
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint))
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.to("cuda")
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if opts.trainer == 'MUNIT':
    # Start testing
    style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).to("cuda"), volatile=True)
    for i, (images, names) in enumerate(zip(data_loader)):
        print(names[1])
        images = Variable(images.to("cuda"), volatile=True)
        content, _ = encode(images)
        style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).to("cuda"), volatile=True)
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
            basename = os.path.basename(names[1])
            path = os.path.join(opts.output_folder+"_%02d"%j,basename)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
elif opts.trainer == 'UNIT':
    print('Start testing with UNIT')
    for i, images in enumerate(data_loader):
        a_realp = torch.transpose(images[0], 0, -1).unsqueeze(1)
        j = 0
        size = a_realp.size()[0]
        while (j<size):
            a_real_test = Variable(a_realp[j:j+80].to("cuda"))
            with torch.no_grad():
                        content, _ = encode(a_real_test)
                        outputs = decode(content)
                        outputs = (outputs + 1) / 2.
            vutils.save_image(outputs[0,0].data, os.path.join(opts.output_folder, str(i) + str(j) + 'output{:03d}.jpg'.format(i)), padding=0, normalize=True)
            if not opts.output_only:
              # also save input images
              vutils.save_image(a_real_test[0,0].data, os.path.join(opts.output_folder, str(i) + str(j) + 'input{:03d}.jpg'.format(i)), padding=0, normalize=True, range =  tuple([-1,+1]))
            j += 80
else:
    pass
