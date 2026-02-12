#!/usr/bin/env python
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys
import time

sys.dont_write_bytecode = True
sys.path.append('../models')
sys.path.append('../mimic')
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
# from unet import ConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from diffusers.optimization import get_scheduler
from termcolor import colored
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import pygame,h5py,argparse

def count_episodes(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        return len(demos.keys())


print('torch ver:', torch.__version__)
pygame.display.init()
assert torch.cuda.is_available()
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument(
        "--inference",
        action='store_true',
    )
args = parser.parse_args() 


##################################
# https://robomimic.github.io/docs/datasets/robomimic_v0.1.html
# dataset_path = os.path.expanduser("../mimic/low_dim_abs.hdf5")
dataset_path_ph = os.path.expanduser("/mnt/disk1t/robomimic_dataset/robomimic/v1.5/can/ph/low_dim_v15.hdf5") # 200 successful trajectories
dataset_path_mh = os.path.expanduser("/mnt/disk1t/robomimic_dataset/robomimic/v1.5/can/mh/low_dim_v15.hdf5") # 300 successful trajectories

print("PH episodes:", count_episodes(dataset_path_ph))
print("MH episodes:", count_episodes(dataset_path_mh))

obs_horizon = 1
pred_horizon = 16
action_horizon = 8
# action_dim = 10
num_epochs = 50


dataset_ph = RobomimicReplayLowdimDataset(
    dataset_path=dataset_path_ph,
    horizon=pred_horizon,
    abs_action=True,
)


dataset_mh = RobomimicReplayLowdimDataset(
    dataset_path=dataset_path_mh,
    horizon=pred_horizon,
    abs_action=True,
)


# with h5py.File(dataset_path_ph, "r") as f:
#     demo = f["data"]["demo_0"]
#     actions = demo["actions"][:10]
#     print('actions examples:', actions)
    

    
# 合并
dataset = ConcatDataset([dataset_ph,  dataset_mh])


normalizer = dataset_ph.get_normalizer()
normalizers = LinearNormalizer()
normalizers.load_state_dict(normalizer.state_dict())

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=16,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)


batch = next(iter(dataloader))
obs_cond_dim = batch["obs"][:, :obs_horizon].flatten(start_dim=1).shape[-1]
action_dim = batch["action"].shape[-1]

print('action_dim:', action_dim)
print('obs_cond_dim:', obs_cond_dim)
print("Num samples:", len(dataset))
print("Num batches:", len(dataloader))


for k, v in dataset[0].items():
    print(k, '===>',  v.shape)


##################################################################
# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_cond_dim
).to(device)

if not args.inference:
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
    avg_loss_train_list = []
    avg_loss_val_list = []

    ########################################################################
    #### Train the model
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        total_loss_train = 0.0
        for data in tqdm(dataloader, desc="Iter", leave=False):
            x_all = normalizers.normalize(data)
            x_img = x_all['obs'][:, :obs_horizon].to(device)
            x_traj = x_all['action'].to(device)
            # print('x_img:', x_img.shape) # torch.Size([64, 1, 23])
            # print('x_traj:',  x_traj.shape) # torch.Size([64, 16, action_dim]) 

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)
            # ut: velocity vec,  xt: a middle action
            # t ~ Uniform(0, 1) 
            # xt = (1-t)*x0 + t*x_traj
            # ut = x_traj - x0
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

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

        # save checkpoint for every epoch
        os.makedirs("./checkpoints/robomimic", exist_ok=True)
        ema.store(noise_pred_net.parameters()) 
        ema.copy_to(noise_pred_net.parameters())
        torch.save({'noise_pred_net': noise_pred_net.state_dict(),
                    'epoch': epoch,
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                    }, f'./checkpoints/robomimic/cp-{epoch}.pth')
        ema.restore(noise_pred_net.parameters())


    os._exit(0)

else:
    ###### test the model
    def undo_transform_action(action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


    epoch = 7
    state_dict = torch.load(f"./checkpoints/robomimic/cp-{epoch}.pth", map_location='cuda')
    noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
    dim = 500
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path_ph)
    
    
    
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path_ph,
        action_keys=["actions"]
    )

    ObsUtils.initialize_obs_utils_with_obs_specs(shape_meta["obs"])

    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object',
            'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
        ],
        render_hw=(dim, dim),
        render_camera_name='frontview'
    )

    n_test = 100
    max_steps = 700
    n_success = 0
    
    for trail_ix in range(n_test):
        seed = random.randint(1, 10000)
        wrapper.seed(seed)

        obs = wrapper.reset()
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        imgs = [wrapper.render(mode='rgb_array')]
        rewards = list()

        step_idx = 0
        done = False

        # with tqdm(total=max_steps, desc="Eval Env") as pbar:
        while not done:
            x_img = np.stack([x for x in obs_deque])
            x_img = torch.from_numpy(x_img)
            x_img = normalizers['obs'].normalize(x_img)
            x_img = x_img.to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # get image features
                obs_cond = x_img.flatten(start_dim=1)

                timehorion = 1
                for i in range(timehorion):
                    noise = torch.rand(1, pred_horizon, action_dim).to(device)
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
                action_pred = normalizers['action'].unnormalize(naction)

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]

                for j in range(len(action)):
                    # stepping env
                    env_action = undo_transform_action(action[j])
                    obs, reward, done, info = wrapper.step(env_action)
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)

                    imgs.append(wrapper.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1

                    # pbar.update(1)
                    # pbar.set_postfix(reward=reward)

                    if step_idx > max_steps :
                        print(f'trial {trail_ix} fail')
                        done = True
                        break
                    
                    if done or reward == 1:
                        n_success += 1
                        print(f'trial {trail_ix} succeed')
                        done = True
                        break

                   
                            
    print('summary:', n_success / n_test)