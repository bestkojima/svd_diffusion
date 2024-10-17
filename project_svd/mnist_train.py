from model import Unet
from schedule import GaussianDiffusion
from train import Trainer
import torchvision
import os
import errno
import shutil
import argparse
from config import denoise_mnist_train

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

create = 0

if create:
    trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True)
    root = './root_mnist/'
    del_folder(root)
    create_folder(root)

    for i in range(10):
        lable_root = root + str(i) + '/'
        create_folder(lable_root)

    for idx in range(len(trainset)):
        img, label = trainset[idx]
        print(idx)
        img.save(root + str(label) + '/' + str(idx) + '.png')

# 使用 TrainingConfig 类来设置参数
config = denoise_mnist_train()
print(config)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=1
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    device_of_kernel = 'cuda',
    channels = 1,
    timesteps = config.time_steps,   # number of steps
    loss_type = config.loss_type,    # L1 or L2
    train_routine = config.train_routine,
    sampling_routine = config.sampling_routine,
).cuda()

import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    config.data_path,
    image_size = 32,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = config.train_steps,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = config.save_folder,
    load_path = config.load_path,
    dataset = 'mnist'
)

trainer.train()