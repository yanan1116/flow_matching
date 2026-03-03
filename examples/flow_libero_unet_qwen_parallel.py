#!/usr/bin/env python
import sys,random,time
sys.dont_write_bytecode = True
sys.path.append('./external/models')
sys.path.append('./external')

LIBERO_ROOT = "/home/yanan/robotics/LIBERO"
if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)
    
    
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
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModel
# from utils import *
from datasets import load_dataset

from libero.libero import benchmark
import pathlib,math,random,imageio,collections,os,sys
from libero.libero import get_libero_path
print('bddl files path:', get_libero_path("bddl_files"))

from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from datasets import load_dataset

benchmark_dict = benchmark.get_benchmark_dict()
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
num_steps_wait = 10
video_out_path: str = "./saved_videos"

import functools
print = functools.partial(print, flush=True)

# Avoid tokenizer parallelism warnings when DataLoader forks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _to_chw_float(img, normalize_01=True):
    # img 可能已经是 torch.uint8 CHW（来自 with_transform）
    if torch.is_tensor(img):
        t = img
    else:
        t = pil_to_tensor(img.convert("RGB"))
    t = t.float()
    if normalize_01:
        t = t / 255.0
    return t

def hf_transform(ex):
    # HF 可能传单条：ex["image"] 是 PIL
    # 也可能传 batch：ex["image"] 是 list[PIL]
    if "image" in ex:
        if isinstance(ex["image"], list):
            ex["image"] = [pil_to_tensor(im.convert("RGB")) for im in ex["image"]]
        else:
            ex["image"] = pil_to_tensor(ex["image"].convert("RGB"))

    if "wrist_image" in ex:
        if isinstance(ex["wrist_image"], list):
            ex["wrist_image"] = [pil_to_tensor(im.convert("RGB")) for im in ex["wrist_image"]]
        else:
            ex["wrist_image"] = pil_to_tensor(ex["wrist_image"].convert("RGB"))

    return ex

class LiberoWindowedDataset(Dataset):
    """
    sample:
      images:       (O, 3, 256, 256)
      wrist_images: (O, 3, 256, 256)
      state:        (1, 8)=
      actions:      (H, 7)  
    """

    def __init__(self, base_ds, horizon=16, obs_horizon=1, normalize_images_01=True, task_map=None):
        self.base = base_ds
        self.H = int(horizon)
        self.O = int(obs_horizon)
        assert self.H > 0 and self.O > 0
        self.normalize_images_01 = normalize_images_01
        
        # task_map: {task_index(int) -> natural language instruction(str)}
        self.task_map = task_map
        assert self.task_map is not None and len(self.task_map) == 40
        self.task_index_arr = np.asarray(self.base["task_index"], dtype=np.int64)

        eps = self.base["episode_index"]
        fis = self.base["frame_index"]
        
        self.eps = np.asarray(eps, dtype=np.int64)
        self.fis = np.asarray(fis, dtype=np.int64)
        
        assert isinstance(eps[0], (int, np.integer)), f"episode_index type: {type(eps[0])}"

        self.actions = np.asarray(self.base["actions"], dtype=np.float32)  # (N,7)
        self.state   = np.asarray(self.base["state"], dtype=np.float32)    # (N,8)

      
        def ep_scalar(x):
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return int(x[0])
            try:
                return int(x)
            except Exception:
                return int(x[0])

        episodes = {}
        for i, e in enumerate(eps):
            ep = ep_scalar(e)
            episodes.setdefault(ep, []).append(i)

                
        # t 是 episode 内观测起点；需要 O 帧观测 + H 步未来动作
        need = self.O + self.H
        self.windows = []
        for ep, idxs in episodes.items():
            L = len(idxs)
            if L >= need:
                for t in range(0, L - need + 1):
                    self.windows.append((idxs, t))

        print(f"[LiberoWindowedDataset] episodes={len(episodes)}, windows={len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        idxs, t = self.windows[idx]

        obs_ids = np.asarray(idxs[t : t + self.O], dtype=np.int64)
        act_ids = np.asarray(idxs[t + self.O : t + self.O + self.H], dtype=np.int64)

        if idx % 1024 == 0:
            ids = np.concatenate([obs_ids, act_ids])
            f = self.fis[ids]
            assert np.all(f[1:] > f[:-1]), f"frame_index not increasing: {f.tolist()}"
            ep0 = self.eps[ids[0]]
            assert np.all(self.eps[ids] == ep0), "cross-episode contamination"

        actions = torch.from_numpy(self.actions[act_ids])              # (H,7)
        state0  = torch.from_numpy(self.state[obs_ids[0]]).view(1, -1) # (1,8)

        images = torch.stack([_to_chw_float(self.base[i]["image"], normalize_01=self.normalize_images_01)
                            for i in obs_ids], dim=0)
        wrist_images = torch.stack([_to_chw_float(self.base[i]["wrist_image"], normalize_01=self.normalize_images_01)
                                    for i in obs_ids], dim=0)
        sample = {"actions": actions, "state": state0, "image": images, "wrist_image": wrist_images}
        ti = int(self.task_index_arr[obs_ids[0]])
        sample["task_text"] = self.task_map[ti]
        
        return sample

from torch.utils.data._utils.collate import default_collate

def collate_with_task_text(batch):
    task_text = [b["task_text"] for b in batch]  # 必定存在
    batch2 = []
    for b in batch:
        b = dict(b)
        b.pop("task_text")
        batch2.append(b)
    out = default_collate(batch2)
    out["task_text"] = task_text
    return out


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


def _init_eval_trials(task, initial_states, n_test_actual, max_steps, save_video):
    trial_states = []
    task_description = None
    for trail_ix in range(n_test_actual):
        env, desc = _get_libero_env(task, LIBERO_ENV_RESOLUTION, random.randint(1, 10000))
        if task_description is None:
            task_description = desc
        env.reset()
        obs = env.set_init_state(initial_states[trail_ix])
        replay_images = []
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            if save_video:
                replay_images.append(obs["agentview_image"][::-1, ::-1, :])
        assert obs["agentview_image"].shape == (256, 256, 3)
        assert obs["robot0_eye_in_hand_image"].shape == (256, 256, 3)
        trial_states.append(
            {
                "trail_ix": trail_ix,
                "env": env,
                "obs_deque": collections.deque([obs] * args.obs_horizon, maxlen=args.obs_horizon),
                "step_idx": 0,
                "done": False,
                "success": False,
                "max_steps": max_steps,
                "replay_images": replay_images,
            }
        )
    return task_description, trial_states


def _prepare_eval_batch(trial_states, active_indices):
    x_main_img = np.stack([
        np.stack([
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1, :].transpose(2, 0, 1))
            for obs in trial_states[idx]["obs_deque"]
        ], axis=0)
        for idx in active_indices
    ], axis=0)
    x_wrist_image = np.stack([
        np.stack([
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1, :].transpose(2, 0, 1))
            for obs in trial_states[idx]["obs_deque"]
        ], axis=0)
        for idx in active_indices
    ], axis=0)
    x_pos = np.stack([
        np.stack([get_state(obs) for obs in trial_states[idx]["obs_deque"]], axis=0)
        for idx in active_indices
    ], axis=0)

    if args.normalize_images_01:
        x_main_img = x_main_img.astype(np.float32) / 255.0
        x_wrist_image = x_wrist_image.astype(np.float32) / 255.0

    x_main_img = torch.from_numpy(x_main_img).to(device, dtype=torch.bfloat16)
    x_wrist_image = torch.from_numpy(x_wrist_image).to(device, dtype=torch.bfloat16)
    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.bfloat16)
    return x_main_img, x_wrist_image, x_pos


def _predict_action_batch(x_main_img, x_wrist_image, x_pos, task_ids_base, task_mask_base):
    B = x_main_img.shape[0]
    x_task_ids = task_ids_base.expand(B, -1).contiguous()
    x_task_mask = task_mask_base.expand(B, -1).contiguous()

    with torch.no_grad():
        image_main_features_visencoder = nets['vision_encoder'](x_main_img.flatten(end_dim=1).to(dtype=torch.bfloat16))
        image_wrist_features_visencoder = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1).to(dtype=torch.bfloat16))
        assert image_main_features_visencoder.shape == image_wrist_features_visencoder.shape

        main_feat = image_main_features_visencoder.reshape(*x_main_img.shape[:2], -1)
        wrist_feat = image_wrist_features_visencoder.reshape(*x_wrist_image.shape[:2], -1)

        if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
            x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
        else:
            x_pos_rep = x_pos

        text_out = nets['text_encoder'](input_ids=x_task_ids, attention_mask=x_task_mask)
        last_hidden = text_out.last_hidden_state
        if args.text_pool == "last":
            idx = x_task_mask.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            text_feat = last_hidden[torch.arange(last_hidden.size(0), device=device), idx]
        else:
            mask = x_task_mask.unsqueeze(-1)
            text_feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        text_feat = text_feat.to(dtype=torch.bfloat16)
        text_feat = text_feat.unsqueeze(1).expand(B, main_feat.shape[1], -1)

        obs_features = torch.cat([main_feat, wrist_feat, x_pos_rep, text_feat], dim=-1)
        if args.net == 'ConditionalUnet1D':
            obs_cond = obs_features.reshape(B, -1).to(dtype=torch.bfloat16)
        else:
            obs_cond = obs_features.to(dtype=torch.bfloat16)

        timehorion = 16
        x0 = torch.rand(B, args.pred_horizon, action_dim, device=device, dtype=torch.bfloat16)
        for i in range(timehorion):
            timestep = torch.tensor([i / timehorion], device=device, dtype=torch.bfloat16)
            if i == 0:
                if args.net == 'TransformerForDiffusion':
                    vt = nets['noise_pred_net'](x0, timestep, obs_cond)
                else:
                    vt = nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                traj = (vt * 1 / timehorion + x0)
            else:
                if args.net == 'TransformerForDiffusion':
                    vt = nets['noise_pred_net'](traj, timestep, obs_cond)
                else:
                    vt = nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
                traj = (vt * 1 / timehorion + traj)

    action_pred = traj.detach().to(dtype=torch.float32, device='cpu').numpy()
    assert action_pred.shape == (B, args.pred_horizon, action_dim)
    return action_pred


def _run_batched_eval_for_task(task, task_description, trial_states, task_ids_base, task_mask_base):
    start = args.obs_horizon - 1
    end = start + args.action_horizon
    n_success = 0

    while True:
        active_indices = [idx for idx, state in enumerate(trial_states) if not state["done"]]
        if not active_indices:
            break
        print(f"batched eval active trials: {len(active_indices)}")

        if args.save_image:
            ref_state = trial_states[active_indices[0]]
            ref_obs = ref_state["obs_deque"][-1]
            step_idx = ref_state["step_idx"]
            if step_idx in [0, 50, 100, 150, 200]:
                Image.fromarray(ref_obs["agentview_image"]).save(f"saved_images/libero_agentview_image_{step_idx}.png")
                Image.fromarray(ref_obs["robot0_eye_in_hand_image"]).save(f"saved_images/libero_robot0_eye_in_hand_image_{step_idx}.png")
                Image.fromarray(np.ascontiguousarray(ref_obs["agentview_image"][::-1, ::-1, :])).save(
                    f"saved_images/libero_agentview_image_flip_{step_idx}.png"
                )
                Image.fromarray(np.ascontiguousarray(ref_obs["robot0_eye_in_hand_image"][::-1, ::-1, :])).save(
                    f"saved_images/libero_robot0_eye_in_hand_image_flip_{step_idx}.png"
                )

        x_main_img, x_wrist_image, x_pos = _prepare_eval_batch(trial_states, active_indices)
        action_pred = _predict_action_batch(x_main_img, x_wrist_image, x_pos, task_ids_base, task_mask_base)

        for batch_idx, state_idx in enumerate(active_indices):
            state = trial_states[state_idx]
            env = state["env"]
            for action in action_pred[batch_idx][start:end, :]:
                obs, reward, done, info = env.step(action.tolist())
                state["step_idx"] += 1
                if args.save_video:
                    state["replay_images"].append(obs["agentview_image"][::-1, ::-1, :])
                state["obs_deque"].append(obs)

                if done:
                    state["done"] = True
                    state["success"] = True
                    n_success += 1
                    print(f'trial {state["trail_ix"]} succeed at step: {state["step_idx"]}')
                    break

                if state["step_idx"] > state["max_steps"]:
                    state["done"] = True
                    print(f'trial {state["trail_ix"]} fail')
                    break

    return n_success

assert torch.cuda.is_available()
device = 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default="ConditionalUnet1D", choices=["TransformerForDiffusion", "ConditionalUnet1D"])
# parser.add_argument("--frozen_vision", action="store_true")
parser.add_argument("--normalize_images_01", action="store_true")
parser.add_argument("--n_test", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=128)
# parser.add_argument("--eval_interval", type=int, default=100)
parser.add_argument("--obs_horizon", type=int, default=1)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--pred_horizon", type=int, default=16)
parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--frozen_text_model", action='store_true')
parser.add_argument("--text_lora_r", type=int, default=16)
parser.add_argument("--text_lora_alpha", type=int, default=32)
parser.add_argument("--text_lora_dropout", type=float, default=0)
parser.add_argument("--text_max_len", type=int, default=64)
parser.add_argument("--text_pool", type=str, default="last", choices=["last", "mean"])
parser.add_argument("--debug", action="store_true")
parser.add_argument( "--save_image", action='store_true')
parser.add_argument( "--save_video", action='store_true')
parser.add_argument( "--save_cp", action='store_true')
parser.add_argument("--eval_cp", type=str, default=None)
parser.add_argument("--video_name", type=str, default="")
parser.add_argument("--cp_name", type=str, default='')
parser.add_argument("--ema", action="store_true")
args = parser.parse_args() 
print('args:', args)
eval_state_dict = None
if args.eval_cp:
    eval_state_dict = torch.load(args.eval_cp, map_location='cpu')
    print("eval_cp---> args:", eval_state_dict.get("args"))

##################################
hf_dataset_repo = "physical-intelligence/libero"
action_dim = 7
base_ds = load_dataset(hf_dataset_repo, split="train")  # :contentReference[oaicite:3]{index=3}
print('base_ds info:', base_ds, '\n', base_ds.features)
print(type(base_ds[0]["image"]))
task_indices = base_ds['task_index']
assert len(set(task_indices)) == 40 and min(task_indices) == 0 and max(task_indices) == 39

meta = load_dataset(
    hf_dataset_repo,
    data_files="meta/tasks.jsonl",
    split="train",
)
task_map = {row["task_index"]: row["task"] for row in meta}
task_texts = [task_map[i] for i in sorted(task_map.keys())]
tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
task_text_enc = tokenizer(
    task_texts,
    padding="max_length",
    truncation=True,
    max_length=args.text_max_len,
    return_tensors="pt",
)
task_text_inputs = {
    t: (task_text_enc["input_ids"][i], task_text_enc["attention_mask"][i])
    for i, t in enumerate(task_texts)
}

base_ds = base_ds.with_transform(hf_transform)
ds = LiberoWindowedDataset(base_ds, 
                           horizon=args.pred_horizon, 
                           obs_horizon=args.obs_horizon, 
                           normalize_images_01=args.normalize_images_01, 
                           task_map=task_map)

dataloader = DataLoader(ds, batch_size=args.batchsize, shuffle=True, 
                            num_workers=16, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4,
                            collate_fn=collate_with_task_text)

batch = next(iter(dataloader))
print(batch.keys())
print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
print('image:', batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])
print('task_text:', len(batch['task_text']))
assert isinstance(batch["task_text"], list)
# for tt in batch['task_text']:
#     print(tt)


# os._exit(0)

if args.save_image:
    imgs = batch["image"]   # shape: [B, O, 3, 256, 256]
    B, O, C, H, W = imgs.shape
    N = 5
    idxs = random.sample(range(B), min(N, B))
    for i, idx in enumerate(idxs):
        img = imgs[idx, 0]
        img = img.permute(1, 2, 0).cpu().numpy()
        if args.normalize_images_01:
            img = np.clip(img, 0.0, 1.0) * 255.0
        else:
            img = np.clip(img, 0.0, 255.0)
        img = img.astype(np.uint8)
        Image.fromarray(img).save(f"saved_images/libero_hf_image_{i}.png")
# images are in normal orientation


# create network object
vision_encoder = get_resnet('resnet18')
vision_encoder = replace_bn_with_gn(vision_encoder)
assert torch.cuda.is_available(), "CUDA is required for bf16 training"
assert torch.cuda.is_bf16_supported(), "GPU does not support bf16"
text_encoder = AutoModel.from_pretrained(
    args.text_model,
    torch_dtype=torch.bfloat16,
)
use_text_lora = (not args.eval_cp and not args.frozen_text_model) or (
    args.eval_cp and eval_state_dict is not None and 'text_encoder_lora' in eval_state_dict
)
if use_text_lora:
    text_lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.text_lora_r,
        lora_alpha=args.text_lora_alpha,
        lora_dropout=args.text_lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    text_encoder = get_peft_model(text_encoder, text_lora_config)
text_embed_dim = int(text_encoder.config.hidden_size)
per_timestep_cond_dim = 512*2 + 8 + text_embed_dim  # 1032 + text
if args.net == "ConditionalUnet1D":
    global_cond_dim = per_timestep_cond_dim * args.obs_horizon
elif args.net == "TransformerForDiffusion":  # Transformer cond 是按 timestep 给的
    global_cond_dim = per_timestep_cond_dim

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
    'text_encoder': text_encoder,
    'noise_pred_net': noise_pred_net
}).to(device, dtype=torch.bfloat16)
    

if args.frozen_text_model:
    nets['text_encoder'].eval()
    for p in nets['text_encoder'].parameters():
        p.requires_grad = False
elif use_text_lora and hasattr(nets['text_encoder'], "print_trainable_parameters"):
    nets['text_encoder'].print_trainable_parameters()
        
##################################################################
sigma = 0.0
# ema_params = list(nets.parameters())
ema_params = [p for p in nets.parameters() if p.requires_grad]
ema = EMAModel(parameters=ema_params, power=0.75) if args.ema else None
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

    if (args.debug or args.eval_cp) and epoch > 0 :
        print('debug or test mode')
        os._exit(0)


    total_loss_train = 0.0
    
    nets.train()
    
    if args.frozen_text_model:
        nets['text_encoder'].eval()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for ii, batch in enumerate(pbar):

        if args.debug:
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
            assert batch_state_min >= -3.14*2 and batch_state_max <= 3.14*2, f'state range error: {batch_state_min} {batch_state_max}'

        x_main_img = batch['image'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        x_wrist_image = batch['wrist_image'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        x_pos = batch['state'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        x_traj = batch['actions'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        x_task_ids = torch.stack([task_text_inputs[t][0] for t in batch["task_text"]], dim=0).to(device, non_blocking=True)
        x_task_mask = torch.stack([task_text_inputs[t][1] for t in batch["task_text"]], dim=0).to(device, non_blocking=True)

        if args.debug:
            assert x_main_img.dtype == torch.bfloat16 and x_wrist_image.dtype == torch.bfloat16
            assert x_pos.dtype == torch.bfloat16 and x_traj.dtype == torch.bfloat16
            assert x_task_ids.dtype == torch.long and x_task_mask.dtype in (torch.long, torch.int64, torch.bool)
            
        # print('train x_main_img:', x_main_img.shape)
        # print('train x_wrist_image:', x_wrist_image.shape)
        # print('train x_pos:', x_pos.shape)
        
        # (batch_size, obs_horizon, channel, height, width)
        # train x_main_img: torch.Size([64, 1, 3, 256, 256])                                                                                                 | 16/3850 [00:04<09:47,  6.53it/s, loss=1.2]
        # train x_wrist_image: torch.Size([64, 1, 3, 256, 256])
        # train x_pos: torch.Size([64, 1, 8])
        # train image_main_features_visencoder: torch.Size([64, 512])
        # train image_wrist_features_visencoder: torch.Size([64, 512])
        # train main_feat: torch.Size([64, 1, 512])
        # train wrist_feat: torch.Size([64, 1, 512])
        # train x_pos_rep: torch.Size([64, 1, 8])
        # train obs_features: torch.Size([64, 1, 1032])
                
        x0 = torch.randn(x_traj.shape, device=device, dtype=torch.bfloat16)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)
        timestep = timestep.to(dtype=torch.bfloat16)
        xt = xt.to(dtype=torch.bfloat16)
        ut = ut.to(dtype=torch.bfloat16)

        # encoder vision features
        image_main_features_visencoder = nets['vision_encoder'](x_main_img.flatten(end_dim=1).to(dtype=torch.bfloat16))
        image_wrist_features_visencoder = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1).to(dtype=torch.bfloat16))

        # print(x_main_img.shape, x_main_img.flatten(end_dim=1).shape, image_main_features_visencoder.shape)
        # print('train image_main_features_visencoder:', image_main_features_visencoder.shape)
        # print('train image_wrist_features_visencoder:', image_wrist_features_visencoder.shape)
        
        
        main_feat  = image_main_features_visencoder.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
        wrist_feat = image_wrist_features_visencoder.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]
        
        # print('train main_feat:', main_feat.shape)
        # print('train wrist_feat:', wrist_feat.shape)        
        
        if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
            x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
        else:
            x_pos_rep = x_pos  # already is of shape [B,O,8] or O=1
        if args.debug:
            assert x_pos_rep.shape[:2] == main_feat.shape[:2]

        if args.frozen_text_model:
            with torch.inference_mode():
                text_out = nets['text_encoder'](input_ids=x_task_ids, attention_mask=x_task_mask)
        else:
            text_out = nets['text_encoder'](input_ids=x_task_ids, attention_mask=x_task_mask)
        last_hidden = text_out.last_hidden_state  # [B, L, D]
        if args.text_pool == "last":
            idx = x_task_mask.sum(dim=1) - 1
            idx = idx.clamp(min=0)
            text_feat = last_hidden[torch.arange(last_hidden.size(0), device=device), idx]
        else:
            mask = x_task_mask.unsqueeze(-1)
            text_feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        text_feat = text_feat.unsqueeze(1).expand(-1, main_feat.shape[1], -1)  # [B, O, D]

        if args.debug:
            assert text_feat.shape[:2] == main_feat.shape[:2]
            assert text_feat.dtype == torch.bfloat16
        
        # print('train x_pos_rep:', x_pos_rep.shape) 
         
        obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep, text_feat], dim=-1)
        # print('train obs_features:', obs_features.shape) 
        
        # expected dimension of per-timestep cond 
        expected_feat_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1] + text_feat.shape[-1]

        B, O = x_main_img.shape[:2]

        if args.debug:
            # 1) check obs_features 
            assert obs_features.shape == (B, O, expected_feat_dim), f"obs_features shape wrong: got {obs_features.shape}, expect {(B, O, expected_feat_dim)}"

        # 2) check the shape before feed to nets
        if args.net == 'ConditionalUnet1D':
            flat = obs_features.flatten(start_dim=1)
            if args.debug:
                assert flat.shape == (B, O * expected_feat_dim),  f"global_cond shape wrong: got {flat.shape}, expect {(B, O * expected_feat_dim)}"
        elif args.net == 'TransformerForDiffusion':
            if args.debug:
                assert obs_features.shape == (B, O, expected_feat_dim), f"cond shape wrong for Transformer: got {obs_features.shape}"

        
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

        if args.ema:
            # update Exponential Moving Average of the model weights
            # ema.step(nets.parameters())
            ema.step(ema_params)
        
        if (args.debug and ii >= 32)  or args.eval_cp:
            break
    
    if epoch % 10 == 0 :
        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))    
        
    
    # do evaluation below - inference
    if (epoch in [50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]) or args.debug or args.eval_cp:
        restore_raw_after_eval = False

        if args.save_cp:
            cp_save_path = "./checkpoints/libero/unet_qwen/"
            os.makedirs(cp_save_path, exist_ok=True)
            ckpt = {
                'vision_encoder': nets['vision_encoder'].state_dict(),
                'noise_pred_net': nets['noise_pred_net'].state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "args": vars(args),
                "model_weights": "raw",
            }
            if args.ema:
                ckpt['ema'] = ema.state_dict()
            if use_text_lora:
                ckpt['text_encoder_lora'] = get_peft_model_state_dict(nets['text_encoder'])
            else:
                ckpt['text_encoder'] = nets['text_encoder'].state_dict()
            torch.save(ckpt, f'{cp_save_path}/cp-{args.cp_name}-{epoch}.pth')
            
        if args.eval_cp:
            nets.eval()
            state_dict = eval_state_dict
            saved_args = state_dict.get("args") or {}
            model_weights = state_dict.get("model_weights")
            if model_weights is None:
                model_weights = "ema" if saved_args.get("ema") else "raw"
            nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
            if 'text_encoder_lora' in state_dict:
                if not use_text_lora:
                    raise RuntimeError("Checkpoint contains LoRA weights but text LoRA was not initialized.")
                set_peft_model_state_dict(nets['text_encoder'], state_dict['text_encoder_lora'])
            elif 'text_encoder' in state_dict:
                nets.text_encoder.load_state_dict(state_dict['text_encoder'])
            else:
                print('warning: text_encoder not found in checkpoint, using current initialized weights')
            nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
            if args.ema:
                if 'ema' in state_dict:
                    ema.load_state_dict(state_dict['ema'])
                    ema.copy_to(ema_params)
                elif model_weights == "ema":
                    print('warning: legacy checkpoint stores EMA weights directly; using loaded model weights as EMA.')
                else:
                    raise RuntimeError("EMA requested, but checkpoint does not contain EMA state.")
            elif model_weights == "ema":
                print('warning: legacy checkpoint stores EMA weights directly; raw weights are unavailable in this checkpoint.')
            print('load official checkpoint success')
        else:
            nets.eval()
            if args.ema:
                ema.store(ema_params)
                ema.copy_to(ema_params)
                restore_raw_after_eval = True

        

        for task_suite_name in benchmark_dict.keys():

            if task_suite_name in ['libero_90', 'libero_100']:
                continue   
            
            np.random.seed(random.randint(1, 10000))
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
            
            task_ll = list(range(task_suite.n_tasks))
            # random.shuffle(task_ll)
            
            for task_id in task_ll:
                task = task_suite.get_task(task_id)

                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < 50:
                    print(task_id, 'initial_states length < 50 :', len(initial_states))

                n_test_actual = min(len(initial_states), args.n_test)
                task_description, trial_states = _init_eval_trials(
                    task=task,
                    initial_states=initial_states,
                    n_test_actual=n_test_actual,
                    max_steps=max_steps,
                    save_video=args.save_video,
                )
                print(f'task_id:{task_id} task_description:{task_description}')

                text_enc = tokenizer(
                    [task_description],
                    padding="max_length",
                    truncation=True,
                    max_length=args.text_max_len,
                    return_tensors="pt",
                )
                task_ids_base = text_enc["input_ids"].to(device, non_blocking=True)
                task_mask_base = text_enc["attention_mask"].to(device, non_blocking=True)

                n_success = _run_batched_eval_for_task(
                    task=task,
                    task_description=task_description,
                    trial_states=trial_states,
                    task_ids_base=task_ids_base,
                    task_mask_base=task_mask_base,
                )

                if args.debug or args.eval_cp:
                    epoch_ = 'eval_cp'
                else:
                    epoch_ = epoch

                if args.save_video:
                    success_trial = next((state for state in trial_states if state["success"]), None)
                    video_state = success_trial if success_trial is not None else trial_states[0]
                    imageio.mimwrite(
                        f"./saved_videos/rollout_{args.video_name}_epoch_{epoch_}_{task_suite_name}_taskid_{task_id}_success_{n_success}_total_{n_test_actual}.mp4",
                        [np.asarray(x) for x in video_state["replay_images"]],
                        fps=10,
                    )

                print(f'task summary --> epoch: {epoch_} suite: {task_suite_name} task_id: {task_id} ({task_description}); success rate:{n_success / n_test_actual} n_test:{n_test_actual}')
                print()
                for state in trial_states:
                    state["env"].close()

 
            print('-'*20)    

        if restore_raw_after_eval:
            ema.restore(ema_params)
    
    
