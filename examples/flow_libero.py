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

from libero.libero import benchmark
import pathlib,math,random,imageio,collections,os,sys
from libero.libero import get_libero_path
print('bddl files path:', get_libero_path("bddl_files"))

from libero.libero.envs import OffScreenRenderEnv
import numpy as np

benchmark_dict = benchmark.get_benchmark_dict()
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
num_steps_wait = 10
video_out_path: str = "./saved_videos"


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    print('task_bddl_file:', task_bddl_file)
    # change for libero-plus
    env_args = {
        "bddl_file_name": str(task_bddl_file),  # 或 task_bddl_file.as_posix()
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def get_state(obs):
    state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ))
    assert state.shape == (8,) 
    return state

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
        assert batch['state'].min() >= -3.14*2 and batch['state'].max() <= 3.14*2 and batch['state'].min() < -1 and batch['state'].max() > 1, \
                f'state range error: {batch_state_min} {batch_state_max}'

        x_main_img = batch['image'].to(device, non_blocking=True).float()
        x_wrist_image = batch['wrist_image'].to(device, non_blocking=True).float()
        x_pos = batch['state'].to(device, non_blocking=True).float()
        x_traj = batch['actions'].to(device, non_blocking=True).float()
        
        print('train x_main_img:', x_main_img.shape)
        print('train x_wrist_image:', x_wrist_image.shape)
        print('train x_pos:', x_pos.shape)
        
        # (batch_size, obs_horizon, channel, height, width)
        # train x_main_img: torch.Size([64, 1, 3, 256, 256])                                                                                                 | 16/3850 [00:04<09:47,  6.53it/s, loss=1.2]
        # train x_wrist_image: torch.Size([64, 1, 3, 256, 256])
        # train x_pos: torch.Size([64, 1, 8])
        # train image_main_features: torch.Size([64, 512])
        # train image_wrist_features: torch.Size([64, 512])
        # train main_feat: torch.Size([64, 1, 512])
        # train wrist_feat: torch.Size([64, 1, 512])
        # train x_pos_rep: torch.Size([64, 1, 8])
        # train obs_features: torch.Size([64, 1, 1032])
                
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
        print('train image_main_features:', image_main_features.shape)
        print('train image_wrist_features:', image_wrist_features.shape)
        
        
        main_feat  = image_main_features.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
        wrist_feat = image_wrist_features.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]
        
        print('train main_feat:', main_feat.shape)
        print('train wrist_feat:', wrist_feat.shape)        
        
        if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
            x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
        else:
            x_pos_rep = x_pos  # 已经是 [B,O,8] 或 O=1
        
        print('train x_pos_rep:', x_pos_rep.shape) 
         
        obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep], dim=-1)
        # print('obs_features:', obs_features.shape)
        print('train obs_features:', obs_features.shape) 
        
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
        
        if args.debug and ii >= 16:
            break
    
    
    
    
    # do evaluation below
    if (epoch % args.eval_interval == 0 and epoch > 0) or args.debug :

        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))

        # os.makedirs("./checkpoints/libero", exist_ok=True)
        # ema.store(ema_params) 
        # ema.copy_to(ema_params)
        
        # torch.save({'vision_encoder': nets['vision_encoder'].state_dict(),
        #             'noise_pred_net': nets['noise_pred_net'].state_dict(),
        #             'epoch': epoch,
        #             'ema': ema.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'lr_scheduler': lr_scheduler.state_dict(),
        #             "args": vars(args)}, 
        #             f'./checkpoints/libero/cp-{args.net}-{epoch}.pth')
        # ema.restore(ema_params)
        PATH_OFFICIAL_CP = './checkpoints/libero/cp-ConditionalUnet1D-300.pth'
        nets.eval()
        state_dict = torch.load(PATH_OFFICIAL_CP, map_location='cuda')
        nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
        nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
        print('load official checkpoint success')
        

        for task_suite_name in benchmark_dict.keys():

            if task_suite_name in ['libero_90', 'libero_100']:
                continue   
            
            seed = random.randint(1, 10000)
            np.random.seed(seed)
            task_suite = benchmark_dict[task_suite_name]()

            print(task_suite_name, task_suite.n_tasks)

            if task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            else:
                raise ValueError(f"Unknown task suite: {task_suite_name}")
            
            # for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            #     # Get task in suite
            #     task = task_suite.get_task(task_id)
                
                
            #     task_description = task.language
                
            #     print(task_id, '===>', task)
            #     print('task_description===>', task_description)
            task_ll = list(range(task_suite.n_tasks))
            random.shuffle(task_ll)
            for task_id in tqdm(task_ll):

                task = task_suite.get_task(task_id)

                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) != 50:
                    print(task_id , 'initial_states length:', len(initial_states))
                    continue
                # Initialize LIBERO environment and task description
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

                # Start episodes
                task_episodes, task_successes = 0, 0
                print('task_id:{} task_description:{}'.format(task_id, task_description))
                # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                for episode_idx in range(10):

                    # Reset environment
                    env.reset()
                    action_plan = collections.deque()

                    # Set initial states
                    obs = env.set_init_state(initial_states[episode_idx])
                    
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    for _ in range( num_steps_wait):
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)

                    print( 'obs:', obs.keys()) # odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 'cookies_1_pos', 'cookies_1_quat', 'cookies_1_to_robot0_eef_pos', 'cookies_1_to_robot0_eef_quat', 'glazed_rim_porcelain_ramekin_1_pos', 'glazed_rim_porcelain_ramekin_1_quat', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state'])
                     
                    assert obs["agentview_image"].shape == (256, 256, 3) and obs["robot0_eye_in_hand_image"].shape == (256, 256, 3) , 'inference images shape check'
                    for img_ in [obs["agentview_image"], obs["robot0_eye_in_hand_image"]]:
                        assert img_.min() >= 0 and img_.max() > 1 and img_.max() < 256

                    
                    obs_deque = collections.deque([obs] * args.obs_horizon, maxlen = args.obs_horizon)
                    
                    
                    done = False
                    # Setup
                    t = 0
                    replay_images = []

                    # logging.info(f"Starting episode {task_episodes+1}...")
                    print('trial:', task_episodes)
                    while not done:

                        for x in obs_deque:
                            assert  x["agentview_image"].shape == (256, 256, 3)
                        
                        x_main_img = np.stack([
                            np.ascontiguousarray(
                                x["agentview_image"][::-1, ::-1, :].transpose(2, 0, 1)[None, ...]
                            )
                            for x in obs_deque
                        ])

                        x_wrist_image = np.stack([
                                np.ascontiguousarray(
                                    x["robot0_eye_in_hand_image"][::-1, ::-1, :].transpose(2, 0, 1)[None, ...]
                                )
                                for x in obs_deque
                            ])                       

                      
                        x_pos = np.stack([get_state(x) for x in obs_deque])[None, ...]
                      
                        assert isinstance(x_main_img, np.ndarray) and isinstance(x_wrist_image, np.ndarray) and isinstance(x_pos, np.ndarray)
                        print('infer x_main_img:', x_main_img.shape)
                        print('infer x_wrist_image:', x_wrist_image.shape)
                        print('infer x_pos:', x_pos.shape)
                                               
                        x_main_img = torch.from_numpy(x_main_img).to(device, dtype=torch.float32)
                        x_wrist_image = torch.from_numpy(x_wrist_image).to(device, dtype=torch.float32)
                        x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)        

                        with torch.no_grad():
                            
                            image_main_features = nets['vision_encoder'](x_main_img.flatten(end_dim=1))
                            image_wrist_features = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1))

                            print('infer image_main_features:', image_main_features.shape)
                            print('infer image_wrist_features:', image_wrist_features.shape)                             

                            main_feat  = image_main_features.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
                            wrist_feat = image_wrist_features.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]                        
                                            
                            print('infer main_feat:', main_feat.shape)
                            print('infer wrist_feat:', wrist_feat.shape)  
        
                            if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
                                x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
                            else:
                                x_pos_rep = x_pos  # 已经是 [B,O,8] 或 O=1                            

                            print('infer x_pos_rep:', x_pos_rep.shape) 
                            obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep], dim=-1)
                            print('infer obs_features:', obs_features.shape) 
                        
                        os._exit(0)
                        # infer x_main_img: (1, 1, 3, 256, 256)
                        # infer x_wrist_image: (1, 1, 3, 256, 256)
                        # infer x_pos: (1, 1, 8)
                        # infer image_main_features: torch.Size([1, 512])
                        # infer image_wrist_features: torch.Size([1, 512])
                        # infer main_feat: torch.Size([1, 1, 512])
                        # infer wrist_feat: torch.Size([1, 1, 512])
                        # infer x_pos_rep: torch.Size([1, 1, 8])
                        # infer obs_features: torch.Size([1, 1, 1032])
                        # Get preprocessed image
                        # IMPORTANT: rotate 180 degrees to match train preprocessing

                    

                        # Save preprocessed image for replay video
                        replay_images.append(img)

                        if not action_plan:
                            # Finished executing previous action chunk -- compute new chunk
                            # Prepare observations dict
                            

                            # Query model to get action
                            t0 = time.time()
                            action_chunk = client.infer(element)["actions"]
                            t1 = time.time()
                            # print('inference time:', t1-t0)
                            assert action_chunk.shape == (10,7)
                            # assert (
                            #     len(action_chunk) >= args.replan_steps
                            # ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        # Execute action in environment
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1



                    task_episodes += 1
                    total_episodes += 1

                    # Save a replay video of the episode
                    suffix = "success" if done else "failure"
                    task_segment = task_description.replace(" ", "_")
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_trial_{episode_idx}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

                    # Log current results
                    # logging.info(f"Success: {done}")
                    # logging.info(f"# episodes completed so far: {total_episodes}")
                    # logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                    if done:
                        print('success!')
                    else:
                        print('failure!')
                print(f'task_summary --> task:{task_description} Total episodes: {task_episodes} SR:{float(task_successes) / float(task_episodes)}')

            # print(f'final_summary --> Total episodes: {total_episodes} SR:{float(total_successes) / float(total_episodes)}')
            
            
            print('-'*20)    
    
    

