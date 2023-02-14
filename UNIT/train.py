"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import random
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
batch_size = config['batch_size']
# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.to("cuda")
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_display_images_a = torch.stack([train_loader_a.dataset[i][:,:,50] for i in range(0,display_size*15,15)]).to("cuda")
train_display_images_b = torch.stack([train_loader_b.dataset[i][:,:,50] for i in range(0,display_size*15,15)]).to("cuda")
test_display_images_a = torch.stack([test_loader_a.dataset[i][:,:,50] for i in range(0,display_size*2,2)]).to("cuda")
test_display_images_b = torch.stack([test_loader_b.dataset[i][:,:,50] for i in range(0,display_size*2,2)]).to("cuda")

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (a_real_p, b_real_p) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        #images_a, images_b = images_a.to("cuda").detach(), images_b.to("cuda").detach()
        
        #######PBS#######
        r = random.sample(range(0, min(a_real_p.shape[-1], b_real_p.shape[-1])), k=int(batch_size))
        if np.argmin([a_real_p.shape[-1], b_real_p.shape[-1]]) == 0:
                r1 = r
                r2 = [int(rr * (b_real_p.shape[-1] - 1) / (a_real_p.shape[-1] - 1)) for rr in r]
        else:
                r2 = r
                r1 = [int(rr * (a_real_p.shape[-1] - 1) // (b_real_p.shape[-1] - 1)) for rr in r]
        a_real, b_real = a_real_p[..., r1], b_real_p[..., r2]
        a_real, b_real = torch.transpose(a_real[0], 0, -1).unsqueeze(1), torch.transpose(b_real[0], 0,-1).unsqueeze(1)
        images_a, images_b = a_real.to("cuda").detach(), b_real.to("cuda").detach()
        
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

