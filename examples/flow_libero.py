#!/usr/bin/env python
import sys,random,time
sys.dont_write_bytecode = True
sys.path.append('./external/models')
sys.path.append('./external')
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from resnet import get_resnet
from TransformerForDiffusion import TransformerForDiffusion
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
# from sklearn.model_selection import train_test_split
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
from utils import *
from datasets import load_dataset

assert torch.cuda.is_available()
device = 'cuda'
parser = argparse.ArgumentParser()
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
print('args:', args)
 
##################################

action_dim = 7
# vision_feature_dim = 514

per_timestep_cond_dim = 512*2 + 8  # 1032
if args.net == "ConditionalUnet1D":
    global_cond_dim = per_timestep_cond_dim * args.obs_horizon
else:  # Transformer cond 是按 timestep 给的
    global_cond_dim = per_timestep_cond_dim



ds_name = "physical-intelligence/libero"
base_ds = load_dataset(ds_name, split="train")  # :contentReference[oaicite:3]{index=3}
print('base_ds info:', base_ds, '\n', base_ds.features)
print(type(base_ds[0]["image"]))
base_ds = base_ds.with_transform(hf_transform)
ds = LiberoWindowedDataset(base_ds, horizon=args.pred_horizon, obs_horizon=args.obs_horizon, normalize_images_01=args.normalize_images_01)

dataloader = DataLoader(ds, batch_size=args.batchsize, shuffle=True, 
                            num_workers=16, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

batch = next(iter(dataloader))
print(batch.keys())
print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
print('image:', batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])
print()

# create network object
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
if args.net == 'TransformerForDiffusion':
    noise_pred_net = TransformerForDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=args.pred_horizon,
        cond_dim=global_cond_dim
    )
elif args.net == 'ConditionalUnet1D':
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim
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
print('model initialized')

        
########################################################################
#### Train the model
for epoch in tqdm(range( args.num_epochs ), desc="Training Epochs"):
    total_loss_train = 0.0
    
    nets.train()
    
    if args.frozen_vision:
        nets['vision_encoder'].eval()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for ii, batch in enumerate(pbar):
            
        batch_wrist_image_min, batch_wrist_image_max = batch['wrist_image'].min().item(), batch['wrist_image'].max().item()
        batch_main_image_min, batch_main_image_max = batch['image'].min().item(), batch['image'].max().item()

        
        if args.normalize_images_01:
            assert batch_wrist_image_min >= 0 and batch_wrist_image_max <= 1, 'wrist_image range error'
            assert batch_main_image_min >= 0 and batch_main_image_max <= 1, 'image range error'
        else:
            assert batch_wrist_image_min >= 0 and ( 1 <= batch_wrist_image_max <= 255), 'wrist_image range error'
            assert batch_main_image_min >= 0 and ( 1 <= batch_main_image_max <= 255), 'image range error'
        
        assert batch['actions'].min() >= -1 and batch['actions'].max() <= 1, 'actions range error'
        
        batch_state_min = batch['state'].min()
        batch_state_max = batch['state'].max()
        assert batch['state'].min() >= -3.14*2 and batch['state'].max() <= 3.14*2, f'state range error: {batch_state_min} {batch_state_max}'

        x_main_img = batch['image'].to(device, non_blocking=True).float()
        x_wrist_image = batch['wrist_image'].to(device, non_blocking=True).float()
        x_pos = batch['state'].to(device, non_blocking=True).float()
        x_traj = batch['actions'].to(device, non_blocking=True).float()

        x0 = torch.randn(x_traj.shape, device=device)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

        # encoder vision features
        if args.frozen_vision:
            with torch.no_grad():
                image_main_features = nets['vision_encoder'](x_main_img.flatten(end_dim=1))
                image_wrist_features = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1))
        else:
            image_main_features = nets['vision_encoder'](x_main_img.flatten(end_dim=1))
            image_wrist_features = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1))

        # print(x_main_img.shape, x_main_img.flatten(end_dim=1).shape, image_main_features.shape)
        
        main_feat  = image_main_features.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
        wrist_feat = image_wrist_features.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]
        
        if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
            x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
        else:
            x_pos_rep = x_pos  # 已经是 [B,O,8] 或 O=1
            
        obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep], dim=-1)
        # print('obs_features:', obs_features.shape)
        
        # 期望的 per-timestep cond 维度
        expected_feat_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1]

        B, O = x_main_img.shape[:2]

        # 1) 检查 obs_features 本身
        assert obs_features.shape == (B, O, expected_feat_dim), \
            f"obs_features shape wrong: got {obs_features.shape}, expect {(B, O, expected_feat_dim)}"

        # 2) 检查送入网络前的形状
        if args.net == 'ConditionalUnet1D':
            flat = obs_features.flatten(start_dim=1)
            assert flat.shape == (B, O * expected_feat_dim), \
                f"global_cond shape wrong: got {flat.shape}, expect {(B, O * expected_feat_dim)}"
        elif args.net == 'TransformerForDiffusion':
            assert obs_features.shape == (B, O, expected_feat_dim), \
                f"cond shape wrong for Transformer: got {obs_features.shape}"

        
        if args.net == 'ConditionalUnet1D':
            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_features.flatten(start_dim=1))
        elif args.net == 'TransformerForDiffusion':
            vt = nets['noise_pred_net'](xt, timestep, obs_features)

        loss = torch.mean((vt - ut) ** 2)
        pbar.set_postfix(loss=float(loss.detach()))
        total_loss_train += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        # ema.step(nets.parameters())
        ema.step(ema_params)
        
        if args.debug and ii >= 100:
            break
    
    
    
    
    
    if epoch % args.eval_interval == 0 and epoch > 0 :

        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))

        os.makedirs("./checkpoints/libero", exist_ok=True)
        ema.store(ema_params) 
        ema.copy_to(ema_params)
        
        torch.save({'vision_encoder': nets['vision_encoder'].state_dict(),
                    'noise_pred_net': nets['noise_pred_net'].state_dict(),
                    'epoch': epoch,
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    "args": vars(args)}, 
                    f'./checkpoints/libero/cp-{args.net}-{epoch}.pth')
        ema.restore(ema_params)

        #  nets.eval()
        #     state_dict = torch.load(PATH_OFFICIAL_CP, map_location='cuda')
        #     nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
        #     nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
        #     print('load official checkpoint success')
        
        # env = pusht.PushTImageEnv()
        # n_success = 0
        # final_rewards = []
        
        # for trail_ix in range(args.n_test):
        #     print(f'do eval at trail:{trail_ix}')
        #     seed = random.randint(1, 10000)
        #     env.seed(seed)

        #     obs, info = env.reset()
        #     obs_deque = collections.deque(
        #         [obs] * args.obs_horizon, maxlen = args.obs_horizon)
        #     imgs = [env.render(mode='rgb_array')]
        #     rewards = list()
        #     done = False
        #     step_idx = 0

        #     # with tqdm(total=args.max_steps, disable=True, desc="Eval PushTImageEnv") as pbar:
        #     # print('step_idx:', step_idx)
        #     while not done:
        #         B = 1
        #         x_img = np.stack([x['image'] for x in obs_deque])
        #         x_pos = np.stack([x['agent_pos'] for x in obs_deque])
        #         x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

        #         x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
        #         x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
        #         # infer action
        #         with torch.no_grad():
        #             # get image features
        #             image_features = nets['vision_encoder'](x_img)
        #             obs_features = torch.cat([image_features, x_pos], dim=-1)
                    

        #             if args.net == 'ConditionalUnet1D':
        #                 obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)   
        #             elif args.net == 'TransformerForDiffusion':
        #                 obs_cond = obs_features.unsqueeze(0)
                        
        #             timehorion = 16 
                    
        #             for i in range(timehorion):
        #                 noise = torch.rand(1, args.pred_horizon, action_dim).to(device)
        #                 # noise = torch.randn(1, args.pred_horizon, action_dim).to(device)
        #                 x0 = noise.expand(x_img.shape[0], -1, -1)
        #                 timestep = torch.tensor([i / timehorion]).to(device)

        #                 if i == 0:
        #                     if args.net == 'TransformerForDiffusion':
        #                         vt = nets['noise_pred_net'](x0, timestep, obs_cond)
        #                     elif args.net == 'ConditionalUnet1D':
        #                         vt = nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                                
        #                     traj = (vt * 1 / timehorion + x0)

        #                 else:
        #                     if args.net == 'TransformerForDiffusion':
        #                         vt = nets['noise_pred_net'](traj, timestep, obs_cond)
        #                     elif args.net == 'ConditionalUnet1D':
        #                         vt = nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
        #                     traj = (vt * 1 / timehorion + traj)

        #         naction = traj.detach().to('cpu').numpy()
        #         naction = naction[0]
        #         action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

        #         # only take action_horizon number of actions
        #         start = args.obs_horizon - 1
        #         end = start + args.action_horizon

        #         # execute action_horizon number of steps
        #         for action in action_pred[start:end, :]:
        #             # stepping env
        #             obs, reward, done, _, info = env.step(action)
        #             # save observations
        #             obs_deque.append(obs)
        #             # and reward/vis
        #             rewards.append(reward)

        #             img = env.render(mode='rgb_array')
        #             imgs.append(img)

        #             # update progress bar
        #             step_idx += 1

        #             # pbar.update(1)
        #             # pbar.set_postfix(reward=reward)

        #             if done:
        #                 print(f'trial {trail_ix} succeed')
        #                 n_success += 1
        #                 break
                    
        #             if step_idx > args.max_steps:
        #                 done = True
        #                 print(f'trial {trail_ix} fail')
        #                 break 

        #     final_rewards.append(reward)  
        #     assert len(final_rewards) == trail_ix + 1     
        #     print(f'trial eval summary: reward:{sum(final_rewards)/(trail_ix + 1)} SR:{n_success / (trail_ix+1)}')
        #     print() 
                    
        # print(colored(f'final eval summary epoch #{epoch}:  reward:{np.array(final_rewards).mean()}  SR:{n_success / args.n_test}', 'green'))

    
