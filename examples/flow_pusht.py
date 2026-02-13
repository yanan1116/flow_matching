#!/usr/bin/env python
import sys,random,time
sys.dont_write_bytecode = True
sys.path.append('./external/models')
sys.path.append('./external')
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pusht
import torch.nn as nn
from tqdm import tqdm
from resnet import get_resnet
from TransformerForDiffusion import TransformerForDiffusion
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from termcolor import colored
import cv2
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
import pygame,h5py,argparse
from unet import ConditionalUnet1D
assert torch.cuda.is_available()
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument( "--eval_official", action='store_true')
parser.add_argument("--net", type=str, default="ConditionalUnet1D", choices=["TransformerForDiffusion", "ConditionalUnet1D"])
parser.add_argument("--frozen_vision", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--normalize_images_01", action="store_true")
parser.add_argument("--max_steps", type=int, default=300)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=5000)
parser.add_argument("--batchsize", type=int, default=64)
parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--obs_horizon", type=int, default=1)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--pred_horizon", type=int, default=16)
args = parser.parse_args() 
    
if args.eval_official:
    args.net = 'ConditionalUnet1D'
print('args:', args)
 
    
##################################

os.makedirs('./checkpoints/pusht', exist_ok=True)
# PATH_TMP = f'./checkpoints/pusht/cp-tmp-{args.net}-{args.frozen_vision}.pth'
PATH_OFFICIAL_CP = './checkpoints/pusht/flow_pusht.pth'

action_dim = 2
vision_feature_dim = 514

# create dataset from file
dataset = pusht.PushTImageDataset(
    dataset_path="./dataset/pusht/pusht_cchi_v7_replay.zarr",
    pred_horizon=args.pred_horizon,
    obs_horizon=args.obs_horizon,
    action_horizon=args.action_horizon
)

# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=args.batchsize,
    num_workers=16,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)


print("Num samples:", len(dataset))
print("Num batches:", len(dataloader))

# batch = next(iter(dataloader))
# print(batch['image'].shape)
# from PIL import Image
# imgs = batch['image']
# print(imgs.min(), imgs.max())

# B = imgs.shape[0]
# idxs = random.sample(range(B), 10)
# os.makedirs("saved_images", exist_ok=True)
# for i, idx in enumerate(idxs):
#     img = imgs[idx, 0]          # shape: [3, 96, 96]
#     img = img.permute(1, 2, 0)  # -> [96, 96, 3]
#     img = img.cpu().numpy().astype(np.uint8)
#     # img = np.clip(img, 0, 255).astype(np.uint8)
#     im = Image.fromarray(img)
#     im.save(f"saved_images/sample_{i}_idx_{idx}.png")

# os._exit(0)

##################################################################
# create network object
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
if args.net == 'TransformerForDiffusion':
    noise_pred_net = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=args.pred_horizon,
        cond_dim=vision_feature_dim
    )
elif args.net == 'ConditionalUnet1D':
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=vision_feature_dim
    )
else:
    raise ValueError("net not found")

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'noise_pred_net': noise_pred_net
}).to(device)
    
if args.frozen_vision:
    nets['vision_encoder'].eval()  # important: disable GN/Dropout behavior changes
    for p in nets['vision_encoder'].parameters():
        p.requires_grad = False
        
##################################################################
sigma = 0.0
# ema_params = list(nets.parameters())
ema_params = [p for p in nets.parameters() if p.requires_grad]
ema = EMAModel(parameters=ema_params, power=0.75)
optimizer = torch.optim.AdamW(params=ema_params, lr=1e-4,weight_decay=1e-6)

# optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * args.num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)

            
            
########################################################################
#### Train the model
for epoch in tqdm(range( 1 if args.eval_official else args.num_epochs ), desc="Training Epochs"):
    total_loss_train = 0.0
    
    nets.train()
    
    if args.frozen_vision:
        nets['vision_encoder'].eval()

    
    for ii, batch in enumerate(dataloader):
        if ii == 0 and epoch == 0:
            print('batch keys:',  batch.keys())
            
        x_img = batch['image'][:, :args.obs_horizon].to(device).float()
        x_pos = batch['agent_pos'][:, :args.obs_horizon].to(device).float()
        x_traj = batch['action'].to(device).float()

        assert batch['image'].min() >= 0 and batch['image'].max() <= 255 and batch['image'].max() > 1
        assert batch['action'].min() >= -1 and batch['action'].max() <= 1
        assert batch['agent_pos'].min() >= -1 and batch['agent_pos'].max() <= 1
    
        if args.normalize_images_01:
            x_img = x_img / 255.0
            
        x0 = torch.randn(x_traj.shape, device=device)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

        # encoder vision features
        if args.frozen_vision:
            with torch.no_grad():
                image_features = nets['vision_encoder'](x_img.flatten(end_dim=1))
        else:
            image_features = nets['vision_encoder'](x_img.flatten(end_dim=1))


        image_features = image_features.reshape(*x_img.shape[:2], -1)
        obs_features = torch.cat([image_features, x_pos], dim=-1)
        if args.net == 'ConditionalUnet1D':
            obs_cond = obs_features.flatten(start_dim=1)
            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond)
        elif args.net == 'TransformerForDiffusion':
            obs_cond = obs_features
            vt = nets['noise_pred_net'](xt, timestep, obs_cond)

        loss = torch.mean((vt - ut) ** 2)
        total_loss_train += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        # ema.step(nets.parameters())
        ema.step(ema_params)
        
        if (args.debug and ii >= 10) or args.eval_official:
            break
            

    if (epoch % args.eval_interval == 0 and epoch > 0) or args.eval_official :
        nets.eval()
        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))

        # os.makedirs("./checkpoints/pusht", exist_ok=True)
        # ema.store(ema_params) 
        # ema.copy_to(ema_params)
        # PATH = f'./checkpoints/pusht/cp-{epoch}.pth'
        
        # torch.save({'vision_encoder': nets['vision_encoder'].state_dict(),
        #             'noise_pred_net': nets['noise_pred_net'].state_dict(),
        #             'epoch': epoch,
        #             'ema': ema.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict()}, 
        #             PATH_TMP)
        # ema.restore(ema_params)

        if args.eval_official:
            state_dict = torch.load(PATH_OFFICIAL_CP, map_location='cuda')
            nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
            nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
            print('load official checkpoint success')
        
        env = pusht.PushTImageEnv()
        n_success = 0
        final_rewards = []
        
        for trail_ix in range(args.n_test):
            print(f'do eval at trail:{trail_ix}')
            seed = random.randint(1, 10000)
            env.seed(seed)

            obs, info = env.reset()
            obs_deque = collections.deque(
                [obs] * args.obs_horizon, maxlen = args.obs_horizon)
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
            done = False
            step_idx = 0

            # with tqdm(total=args.max_steps, disable=True, desc="Eval PushTImageEnv") as pbar:
            # print('step_idx:', step_idx)
            while not done:
                B = 1
                x_img = np.stack([x['image'] for x in obs_deque])
                x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                assert x_img.min() >= 0 and x_img.max() <= 1 , 'assertion error at obs_deque within loop'
                
                # if args.normalize_images_01:
                #     x_img = x_img / 255.0
            
                x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                # infer action
                with torch.no_grad():
                    # get image features
                    image_features = nets['vision_encoder'](x_img)
                    obs_features = torch.cat([image_features, x_pos], dim=-1)
                    

                    if args.net == 'ConditionalUnet1D':
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)   
                    elif args.net == 'TransformerForDiffusion':
                        obs_cond = obs_features.unsqueeze(0)
                        
                    timehorion = 16 
                    
                    for i in range(timehorion):
                        noise = torch.rand(1, args.pred_horizon, action_dim).to(device)
                        # noise = torch.randn(1, args.pred_horizon, action_dim).to(device)
                        x0 = noise.expand(x_img.shape[0], -1, -1)
                        timestep = torch.tensor([i / timehorion]).to(device)

                        if i == 0:
                            if args.net == 'TransformerForDiffusion':
                                vt = nets['noise_pred_net'](x0, timestep, obs_cond)
                            elif args.net == 'ConditionalUnet1D':
                                vt = nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                                
                            traj = (vt * 1 / timehorion + x0)

                        else:
                            if args.net == 'TransformerForDiffusion':
                                vt = nets['noise_pred_net'](traj, timestep, obs_cond)
                            elif args.net == 'ConditionalUnet1D':
                                vt = nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / timehorion + traj)

                naction = traj.detach().to('cpu').numpy()
                naction = naction[0]
                action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = args.obs_horizon - 1
                end = start + args.action_horizon

                # execute action_horizon number of steps
                for action in action_pred[start:end, :]:
                    # stepping env
                    obs, reward, done, _, info = env.step(action)
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)

                    img = env.render(mode='rgb_array')
                    imgs.append(img)

                    # update progress bar
                    step_idx += 1

                    # pbar.update(1)
                    # pbar.set_postfix(reward=reward)

                    if done:
                        print(f'trial {trail_ix} succeed')
                        n_success += 1
                        break
                    
                    if step_idx > args.max_steps:
                        done = True
                        print(f'trial {trail_ix} fail')
                        break 

            final_rewards.append(reward)  
            assert len(final_rewards) == trail_ix + 1     
            print(f'trial eval summary: reward:{sum(final_rewards)/(trail_ix + 1)} SR:{n_success / (trail_ix+1)}')
            print() 
                    
        print(colored(f'final eval summary epoch #{epoch}:  reward:{np.array(final_rewards).mean()}  SR:{n_success / args.n_test}', 'green'))

    
    if args.eval_official:
        os._exit(0)