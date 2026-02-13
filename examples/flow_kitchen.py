#!/usr/bin/env python
import sys,argparse
sys.dont_write_bytecode = True
# sys.path.append('../models')
# sys.path.append('../kitchen')
sys.path.append('./external/models')
sys.path.append('./external')
import os,random
# os.environ["MUJOCO_GL"] = "egl"
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from unet import ConditionalUnet1D
from resnet import get_resnet
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from termcolor import colored
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
import pathlib
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
import kitchen_lowdim_dataset
from diffusion_policy.env.kitchen.v0 import KitchenAllV0

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = 'cuda'
##################################
###### add Franka kitchen data to some folder
dataset_path = "./dataset/kitchen"
PATH_OFFICIAL_CP = './checkpoints/kitchen/flow_kitchen.pth'

parser = argparse.ArgumentParser()
parser.add_argument(
        "--eval_official",
        action='store_true',
    )
args = parser.parse_args() 

print('args:', args)
    
obs_horizon = 1
pred_horizon = 16
action_dim = 9
action_horizon = 8
num_epochs = 10000
vision_feature_dim = 60
batch_size = 128
eval_interval = 50

# create dataset from file
dataset = kitchen_lowdim_dataset.KitchenLowdimDataset(
    dataset_dir=dataset_path,
    horizon=16,
)
# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=16,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)
# batch = next(iter(dataloader))

print("Num samples:", len(dataset))
print("Num batches:", len(dataloader))



##################################################################
# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
).to(device)

##################################################################
sigma = 0.0
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)
avg_loss_val_list = []

########################################################################
#### Train the model

for epoch in tqdm(range(1 if args.eval_official else num_epochs)):
    noise_pred_net.train()
    total_loss_train = 0.0
    batct_cnt = 0
    for ii, batch in enumerate(dataloader):
        x_img = batch['obs'][:, :obs_horizon].to(device)
        x_traj = batch['action'].to(device)
        if ii == 0:
            print(batch.keys())
            print('obs:', batch['obs'].shape) # torch.Size([64, 16, 60])
            print('action:', batch['action'].shape) # torch.Size([64, 16, 9])
        x_traj = x_traj.float()
        x0 = torch.randn(x_traj.shape, device=device)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

        # encoder state features
        obs_cond = x_img.flatten(start_dim=1)
        vt = noise_pred_net(xt, timestep, global_cond=obs_cond)

        loss = torch.mean((vt - ut) ** 2)
        total_loss_train += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(noise_pred_net.parameters())
        batct_cnt += 1
        # ema.store(noise_pred_net.parameters()) 
        # ema.copy_to(noise_pred_net.parameters())
        # PATH = f'./checkpoints/kitchen/flow_kitchen_cp_{epoch}.pth'
        # torch.save({ 'noise_pred_net': noise_pred_net.state_dict(),}, PATH)
        # ema.restore(noise_pred_net.parameters())        
        # if batct_cnt >= 5:
        #     break
        if args.eval_official:
            break
        
    avg_loss_train = total_loss_train / len(dataloader)
    print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

    if (epoch > 0 and epoch % eval_interval == 0) or args.eval_official:
        if args.eval_official:
            state_dict = torch.load(PATH_OFFICIAL_CP, map_location='cuda')
            noise_pred_net.load_state_dict(state_dict['noise_pred_net'])

        noise_pred_net.eval()
        
        max_steps = 280
        env = KitchenAllV0(use_abs_action=False)

        n_test = 100

        n_success = 0
        final_rewards = []
            
        for trail_ix in range(n_test):
            seed = random.randint(1, 10000)
            env.seed(seed)
            obs = env.reset()

            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
            done = False
            step_idx = 0

            # with tqdm(total=max_steps, desc="Eval KitchenAllV0") as pbar:
            while not done:
                x_img = np.stack([x for x in obs_deque])
                x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # get image features
                    obs_cond = x_img.flatten(start_dim=1)

                    timehorion = 16
                    for i in range(timehorion):
                        noise = torch.rand(1, pred_horizon, action_dim).to(device) # torch.randn
                        x0 = noise.expand(x_img.shape[0], -1, -1)
                        timestep = torch.tensor([i / timehorion]).to(device)

                        if i == 0:
                            vt = noise_pred_net(x0, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / timehorion + x0)

                        else:
                            vt = noise_pred_net(traj, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / timehorion + traj)

                    naction = traj.detach().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = naction

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    for j in range(len(action)):
                        # stepping env
                        out_step = env.step(action[j])
                        assert len(out_step) == 4
                        obs, reward, done, info = out_step
                        # print(f'action step {step_idx} / {max_steps} --->', info['completed_tasks'], reward)
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        img = env.render(mode='rgb_array')
                        imgs.append(img)
                        # Image.fromarray(img).save(f"saved_images/franka_kitchen_{step_idx:06d}.png")
                        # update progress bar
                        step_idx += 1

                        # pbar.update(1)
                        # pbar.set_postfix(reward=reward)
                        assert reward in [0, 1]
                        tasks_n_completed = len(info['completed_tasks'])
                        # if tasks_n_completed == 4:
                        #     assert reward == 1 
                        # else:
                        #     assert reward == 0, f'reward error: {reward} {tasks_n_completed}'
                            

                        if len(info['completed_tasks'])>=4:
                            print(f'trial {trail_ix} succeed')
                            n_success += 1
                            done = True
                            break 

                        if step_idx > max_steps: 
                            done = True
                            print(f'trial {trail_ix} fail')
                            break                

            # tasks_complted = len(info['completed_tasks'])
            # print(f'trial reward: {reward} completed tasks: {tasks_complted}')
            final_rewards.append(reward) 
            assert len(final_rewards) == trail_ix + 1     
            print(f'trial eval summary at epoch {epoch} SR:{n_success / (trail_ix+1)}\n')
        print(f'final eval summary at epoch {epoch} SR:{n_success / (n_test)}\n')
        
        if args.eval_official:
            os._exit(0)

         