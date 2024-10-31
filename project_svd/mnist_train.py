from main_structure.model import Unet
from main_structure.schedule import GaussianDiffusion,SVDDiffusion
from main_structure.train import Trainer
import torchvision
import os
import errno
import shutil
import argparse
from config import denoise_mnist_train,model_config,mnist_config

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from pprint import pprint

# 使用 TrainingConfig 类来设置参数
config = denoise_mnist_train()
pprint(config)
print("##########################")
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=3
).cuda()


image_size=32
timesteps=image_size+1
diffusion = SVDDiffusion(
    model,
    image_size = 32,
    
    channels = 3,
    timesteps = 32,   # number of steps mnist
    loss_type = config.loss_type,    # L1 or L2
    train_routine = config.train_routine,
    sampling_routine = config.sampling_routine,
).to(device)

# diffusion = GaussianDiffusion(
#     model,
#     image_size = 32,
    
#     channels = 3,
#     timesteps = 32,   # number of steps
#     loss_type = config.loss_type,    # L1 or L2
#     train_routine = config.train_routine,
#     sampling_routine = config.sampling_routine,
# ).to(device)





import torch
diffusion = torch.nn.DataParallel(diffusion, device_ids=range(torch.cuda.device_count()))
config.train_steps=10000
config.data_path= "./root_mnist"
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
    results_folder = "test_local",
    load_path = config.load_path,
    save_and_sample_every=200,
    dataset = 'mnist'
)

trainer.train()