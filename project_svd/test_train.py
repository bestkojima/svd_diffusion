from model import Unet
from schedule import GaussianDiffusion,SVDDiffusion
from train import Trainer
import torchvision
import os
import errno
import shutil
import argparse
from config import denoise_mnist_train,model_config,mnist_config

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 使用 TrainingConfig 类来设置参数
config = denoise_mnist_train()
print(config)

model = Unet(
    dim = 8,
    dim_mults = (1, 2),
    channels=3
).to(device)

config.time_steps=32
diffusion = SVDDiffusion(
    model,
    image_size = 32,
    
    channels = 3,
    timesteps = config.time_steps,   # number of steps
    loss_type = config.loss_type,    # L1 or L2
    train_routine = config.train_routine,
    sampling_routine = config.sampling_routine,
).to(device)


# diffusion = GaussianDiffusion(
#     model,
#     image_size = 32,
    
#     channels = 3,
#     timesteps = config.time_steps,   # number of steps
#     loss_type = config.loss_type,    # L1 or L2
#     train_routine = config.train_routine,
#     sampling_routine = config.sampling_routine,
# ).to(device)


import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))

trainer = Trainer(
    diffusion,
    config.data_path,
    image_size = 32,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 10,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = config.save_folder,
    load_path = config.load_path,
    save_and_sample_every=2,
    dataset = 'mnist'
)

trainer.train()