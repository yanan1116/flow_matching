#!/usr/bin/env python
import copy
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
import io
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


def _ensure_pil_rgb(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if torch.is_tensor(img):
        return img
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path") is not None:
            return Image.open(img["path"]).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_chw_float(img, normalize_01=True):
    # img 可能已经是 torch.uint8 CHW（来自 with_transform）
    if torch.is_tensor(img):
        t = img
    else:
        t = pil_to_tensor(_ensure_pil_rgb(img))
    t = t.float()
    if normalize_01:
        t = t / 255.0
    return t

def hf_transform(ex):
    # HF 可能传单条 / batch；元素可能是 PIL，也可能是 {"bytes","path"} 字典
    if "image" in ex:
        if isinstance(ex["image"], list):
            ex["image"] = [pil_to_tensor(_ensure_pil_rgb(im)) for im in ex["image"]]
        else:
            ex["image"] = pil_to_tensor(_ensure_pil_rgb(ex["image"]))

    if "wrist_image" in ex:
        if isinstance(ex["wrist_image"], list):
            ex["wrist_image"] = [pil_to_tensor(_ensure_pil_rgb(im)) for im in ex["wrist_image"]]
        else:
            ex["wrist_image"] = pil_to_tensor(_ensure_pil_rgb(ex["wrist_image"]))

    return ex


def serialize_depth_to_tokens(depth_field):
    arr = np.asarray(depth_field)
    if arr.size == 0:
        return "<DEPTH_START> <DEPTH_END>"

    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)

    arr = arr.astype(np.float32, copy=False)
    finite_mask = np.isfinite(arr)
    if not np.all(finite_mask):
        arr = np.where(finite_mask, arr, 0.0)

    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_min >= 0.0 and arr_max <= 1.0 + 1e-6:
        arr = np.rint(arr * 256.0)
    else:
        arr = np.rint(arr)

    arr = np.clip(arr, 0, 256).astype(np.int64, copy=False)
    tokens = [f"<DEPTH_{int(v)}>" for v in arr.reshape(-1).tolist()]
    return " ".join(["<DEPTH_START>"] + tokens + ["<DEPTH_END>"])

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
        assert "depth" in self.base.column_names and "eef_traj" in self.base.column_names, (
            "Dataset must contain 'depth' and 'eef_traj' columns."
        )
        self.depth = list(self.base["depth"])
        self.eef_traj = list(self.base["eef_traj"])

      
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
        # Use current observation frame as CoT supervision target to avoid future leakage.
        sample["depth_text"] = serialize_depth_to_tokens(self.depth[int(obs_ids[0])])
        sample["eef_text"] = str(self.eef_traj[int(obs_ids[0])])
        
        return sample

from torch.utils.data._utils.collate import default_collate

def collate_with_task_text(batch):
    task_text = [b["task_text"] for b in batch]  # 必定存在
    depth_text = [b["depth_text"] for b in batch]
    eef_text = [b["eef_text"] for b in batch]
    batch2 = []
    for b in batch:
        b = dict(b)
        b.pop("task_text")
        b.pop("depth_text")
        b.pop("eef_text")
        batch2.append(b)
    out = default_collate(batch2)
    out["task_text"] = task_text
    out["depth_text"] = depth_text
    out["eef_text"] = eef_text
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


assert torch.cuda.is_available()
device = 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default="ConditionalUnet1D", choices=["TransformerForDiffusion", "ConditionalUnet1D"])
# parser.add_argument("--frozen_vision", action="store_true")
parser.add_argument("--normalize_images_01", action="store_true")
parser.add_argument("--n_test", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=3000)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--batch_earlystop", type=int, default=-1)
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
parser.add_argument("--depth_text_max_len", type=int, default=512)
parser.add_argument("--eef_text_max_len", type=int, default=64)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--save_image", action='store_true')
parser.add_argument("--save_video", action='store_true')
parser.add_argument("--save_cp", action='store_true')
parser.add_argument("--eval_cp", type=str, default=None)
parser.add_argument("--video_name", type=str, default="")
parser.add_argument("--cp_name", type=str, default='')
parser.add_argument("--eval_realtime", action="store_true")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--prefetch_factor", type=int, default=2)
parser.add_argument("--disable_text_input", action='store_true')
parser.add_argument("--dataset_repo", type=str, default="yananchen/libero_cot_contious")

parser.add_argument("--lambda_fm_start", type=float, default=0.1)
parser.add_argument("--lambda_fm_end", type=float, default=1.0)
parser.add_argument("--lambda_depth_start", type=float, default=1.0)
parser.add_argument("--lambda_depth_end", type=float, default=0.1)
parser.add_argument("--lambda_eef_start", type=float, default=1.0)
parser.add_argument("--lambda_eef_end", type=float, default=0.1)
parser.add_argument("--lambda_ramp_epochs", type=int, default=60)
parser.add_argument("--lambda_plateau_min_epochs", type=int, default=100)
parser.add_argument("--lambda_plateau_window", type=int, default=8)
parser.add_argument("--lambda_plateau_patience", type=int, default=3)
parser.add_argument("--lambda_plateau_eps_depth", type=float, default=0.01)
parser.add_argument("--lambda_plateau_eps_eef", type=float, default=0.01)
parser.add_argument("--lambda_ema_beta", type=float, default=0.9)
parser.add_argument("--lambda_min_cot_drop", type=float, default=0.2)
# Baseline prior: vanilla training often saturates around 500-600 epochs.
# Add guardrails so CoT phases do not consume most of the useful training budget.
parser.add_argument("--lambda_force_ramp_epoch", type=int, default=240)
parser.add_argument("--lambda_force_action_epoch", type=int, default=360)
args = parser.parse_args() 
if args.eval_cp and args.eval_realtime:
    raise ValueError("--eval_cp and --eval_realtime cannot be used together")

if args.disable_text_input:
    assert args.frozen_text_model , 'when disable_text_input, text model should be frozen'
print('args:', args)
if args.lambda_force_action_epoch < args.lambda_force_ramp_epoch:
    print(
        colored(
            f"warning: lambda_force_action_epoch ({args.lambda_force_action_epoch}) < "
            f"lambda_force_ramp_epoch ({args.lambda_force_ramp_epoch}); action_dominant may start directly.",
            "red",
        )
    )

eval_epoch_milestones = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2600, 2800]

eval_state_dict = None
if args.eval_cp:
    eval_state_dict = torch.load(args.eval_cp, map_location='cpu')
    print("eval_cp---> args:", eval_state_dict.get("args"))

##################################
hf_dataset_repo = args.dataset_repo
action_dim = 7
base_ds = load_dataset(hf_dataset_repo, split="train")  # :contentReference[oaicite:3]{index=3}
print('base_ds info:', base_ds, '\n', base_ds.features)
print(type(base_ds[0]["image"]))
task_indices = base_ds['task_index']
assert len(set(task_indices)) == 40 and min(task_indices) == 0 and max(task_indices) == 39


meta = load_dataset(
    "physical-intelligence/libero",
    data_files="meta/tasks.jsonl",
    split="train",
)
task_map = {row["task_index"]: row["task"] for row in meta}
task_texts = [task_map[i] for i in sorted(task_map.keys())]
tokenizer = AutoTokenizer.from_pretrained(args.text_model, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


special_tokens = ["<DEPTH_START>", "<DEPTH_END>"] + [
    f"<DEPTH_{i}>" for i in range(256 + 1)
]
n_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
print(f"depth special tokens added: {n_added}")

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
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=args.prefetch_factor,
                            collate_fn=collate_with_task_text)

batch = next(iter(dataloader))
print(batch.keys())
print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
print('image:', batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])
print('task_text:', len(batch['task_text']))
print('depth_text:', len(batch['depth_text']))
print('eef_text:', len(batch['eef_text']))
assert isinstance(batch["task_text"], list)
assert isinstance(batch["depth_text"], list) and isinstance(batch["eef_text"], list)
# for tt in batch['task_text']:
#     print(tt)


# os._exit(0)

if args.save_image:
    os.makedirs('./saved_images', exist_ok=True)
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
        Image.fromarray(img).save(f"./saved_images/libero_hf_image_{i}.png")
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
text_encoder.resize_token_embeddings(len(tokenizer))
target_text_encoder = copy.deepcopy(text_encoder)

if not args.frozen_text_model:
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



per_timestep_obs_dim = 512*2 + 8 + (0 if args.disable_text_input else text_embed_dim  )
per_timestep_cond_dim = per_timestep_obs_dim + 2 * text_embed_dim  # add predicted depth/eef CoT latents
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

cot_depth_head = nn.Sequential(
    nn.Linear(per_timestep_obs_dim, per_timestep_obs_dim),
    nn.SiLU(),
    nn.Linear(per_timestep_obs_dim, text_embed_dim),
)
cot_eef_head = nn.Sequential(
    nn.Linear(per_timestep_obs_dim, per_timestep_obs_dim),
    nn.SiLU(),
    nn.Linear(per_timestep_obs_dim, text_embed_dim),
)

nets = nn.ModuleDict({
    'vision_encoder': vision_encoder,
    'text_encoder': text_encoder,
    'target_text_encoder': target_text_encoder,
    'noise_pred_net': noise_pred_net,
    'cot_depth_head': cot_depth_head,
    'cot_eef_head': cot_eef_head,
}).to(device, dtype=torch.bfloat16)
    
nets['target_text_encoder'].eval()
for p in nets['target_text_encoder'].parameters():
    p.requires_grad = False

if args.frozen_text_model:
    nets['text_encoder'].eval()
    for p in nets['text_encoder'].parameters():
        p.requires_grad = False
else:
    nets['text_encoder'].print_trainable_parameters()
        
##################################################################
sigma = 0.0
trainable_params = [p for p in nets.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params=trainable_params, lr=1e-4,weight_decay=1e-6)

# optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * args.num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)
print('model initialized')

lambda_fm = float(args.lambda_fm_start)
lambda_depth = float(args.lambda_depth_start)
lambda_eef = float(args.lambda_eef_start)
lambda_stage = "cot_pretrain"  # cot_pretrain -> ramp -> action_dominant
ramp_start_epoch = None
plateau_count = 0
depth_ema = None
eef_ema = None
depth_ema_hist = []
eef_ema_hist = []
init_depth_ema = None
init_eef_ema = None

########################################################################
#### Train the model
for epoch in tqdm(range( args.num_epochs ), desc="Training Epochs"):

    total_loss_train = 0.0
    total_fm_loss = 0.0
    total_depth_loss = 0.0
    total_eef_loss = 0.0

    if lambda_stage == "ramp":
        assert ramp_start_epoch is not None
        ramp_len = max(1, int(args.lambda_ramp_epochs))
        progress = min(1.0, max(0.0, float(epoch - ramp_start_epoch + 1) / float(ramp_len)))
        lambda_fm = float(args.lambda_fm_start + (args.lambda_fm_end - args.lambda_fm_start) * progress)
        lambda_depth = float(args.lambda_depth_start + (args.lambda_depth_end - args.lambda_depth_start) * progress)
        lambda_eef = float(args.lambda_eef_start + (args.lambda_eef_end - args.lambda_eef_start) * progress)
        if progress >= 1.0:
            lambda_stage = "action_dominant"
    elif lambda_stage == "action_dominant":
        lambda_fm = float(args.lambda_fm_end)
        lambda_depth = float(args.lambda_depth_end)
        lambda_eef = float(args.lambda_eef_end)
    else:
        lambda_fm = float(args.lambda_fm_start)
        lambda_depth = float(args.lambda_depth_start)
        lambda_eef = float(args.lambda_eef_start)

    # Hard guardrails from known baseline saturation window (500~600 epochs).
    # Ensure we do not stay too long in CoT-heavy stages.
    if lambda_stage == "cot_pretrain" and (epoch + 1) >= int(args.lambda_force_ramp_epoch):
        lambda_stage = "ramp"
        ramp_start_epoch = epoch + 1
        plateau_count = 0
        print(colored(
            f"[adaptive-lambda] force ramp at epoch={epoch + 1} (lambda_force_ramp_epoch reached)",
            "cyan",
        ))
    if (epoch + 1) >= int(args.lambda_force_action_epoch):
        if lambda_stage != "action_dominant":
            print(colored(
                f"[adaptive-lambda] force action_dominant at epoch={epoch + 1} (lambda_force_action_epoch reached)",
                "cyan",
            ))
        lambda_stage = "action_dominant"
        lambda_fm = float(args.lambda_fm_end)
        lambda_depth = float(args.lambda_depth_end)
        lambda_eef = float(args.lambda_eef_end)
    
    nets.train()
    
    if args.frozen_text_model:
        nets['text_encoder'].eval()
    nets['target_text_encoder'].eval()
    
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
        x_depth_text = batch["depth_text"]
        x_eef_text = batch["eef_text"]

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
        if args.disable_text_input:
            obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep], dim=-1)
            expected_obs_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1]

        else:
            obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep, text_feat], dim=-1)
            expected_obs_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1] + text_feat.shape[-1]

        B, O = x_main_img.shape[:2]

        if args.debug:
            assert obs_features.shape == (B, O, expected_obs_dim), f"obs_features shape wrong: got {obs_features.shape}, expect {(B, O, expected_obs_dim)}"

        depth_enc = tokenizer(
            x_depth_text,
            padding="max_length",
            truncation=True,
            max_length=args.depth_text_max_len,
            return_tensors="pt",
        )
        eef_enc = tokenizer(
            x_eef_text,
            padding="max_length",
            truncation=True,
            max_length=args.eef_text_max_len,
            return_tensors="pt",
        )
        depth_ids = depth_enc["input_ids"].to(device, non_blocking=True)
        depth_mask = depth_enc["attention_mask"].to(device, non_blocking=True)
        eef_ids = eef_enc["input_ids"].to(device, non_blocking=True)
        eef_mask = eef_enc["attention_mask"].to(device, non_blocking=True)

        with torch.no_grad():
            depth_out = nets["target_text_encoder"](input_ids=depth_ids, attention_mask=depth_mask)
            eef_out = nets["target_text_encoder"](input_ids=eef_ids, attention_mask=eef_mask)

        depth_last_hidden = depth_out.last_hidden_state
        eef_last_hidden = eef_out.last_hidden_state
        if args.text_pool == "last":
            depth_idx = depth_mask.sum(dim=1) - 1
            depth_idx = depth_idx.clamp(min=0)
            eef_idx = eef_mask.sum(dim=1) - 1
            eef_idx = eef_idx.clamp(min=0)
            depth_target = depth_last_hidden[torch.arange(depth_last_hidden.size(0), device=device), depth_idx]
            eef_target = eef_last_hidden[torch.arange(eef_last_hidden.size(0), device=device), eef_idx]
        else:
            depth_m = depth_mask.unsqueeze(-1)
            eef_m = eef_mask.unsqueeze(-1)
            depth_target = (depth_last_hidden * depth_m).sum(dim=1) / depth_m.sum(dim=1).clamp(min=1)
            eef_target = (eef_last_hidden * eef_m).sum(dim=1) / eef_m.sum(dim=1).clamp(min=1)
        depth_target = depth_target.detach().to(dtype=torch.bfloat16)
        eef_target = eef_target.detach().to(dtype=torch.bfloat16)

        # Predict CoT latents from current multimodal observation context.
        obs_context = obs_features.mean(dim=1)  # [B, D_obs]
        depth_pred = nets["cot_depth_head"](obs_context)
        eef_pred = nets["cot_eef_head"](obs_context)
        loss_depth = torch.mean((depth_pred - depth_target) ** 2)
        loss_eef = torch.mean((eef_pred - eef_target) ** 2)

        depth_cond = depth_pred.unsqueeze(1).expand(-1, O, -1)  # [B, O, D_text]
        eef_cond = eef_pred.unsqueeze(1).expand(-1, O, -1)      # [B, O, D_text]
        fm_cond = torch.cat([obs_features, depth_cond, eef_cond], dim=-1)
        expected_cond_dim = expected_obs_dim + depth_cond.shape[-1] + eef_cond.shape[-1]
        if args.debug:
            assert fm_cond.shape == (B, O, expected_cond_dim), f"fm_cond shape wrong: got {fm_cond.shape}, expect {(B, O, expected_cond_dim)}"

        if args.net == 'ConditionalUnet1D':
            vt = nets['noise_pred_net'](xt, timestep, global_cond=fm_cond.flatten(start_dim=1))
        elif args.net == 'TransformerForDiffusion':
            vt = nets['noise_pred_net'](xt, timestep, fm_cond)
        loss_fm = torch.mean((vt - ut) ** 2)

        loss = lambda_fm * loss_fm + lambda_depth * loss_depth + lambda_eef * loss_eef
        pbar.set_postfix(
            loss=float(loss.detach()),
            fm=float(loss_fm.detach()),
            depth=float(loss_depth.detach()),
            eef=float(loss_eef.detach()),
            l_fm=f"{lambda_fm:.3f}",
            l_d=f"{lambda_depth:.3f}",
            l_e=f"{lambda_eef:.3f}",
        )
        total_loss_train += loss.detach()
        total_fm_loss += loss_fm.detach()
        total_depth_loss += loss_depth.detach()
        total_eef_loss += loss_eef.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        
        if (args.debug and ii >= 32)  or args.eval_cp or (args.batch_earlystop == ii ):
            break

    avg_loss_train = float((total_loss_train / len(dataloader)).item())
    avg_fm_loss = float((total_fm_loss / len(dataloader)).item())
    avg_depth_loss = float((total_depth_loss / len(dataloader)).item())
    avg_eef_loss = float((total_eef_loss / len(dataloader)).item())

    beta = float(args.lambda_ema_beta)
    if depth_ema is None:
        depth_ema = avg_depth_loss
        eef_ema = avg_eef_loss
        init_depth_ema = depth_ema
        init_eef_ema = eef_ema
    else:
        depth_ema = beta * depth_ema + (1.0 - beta) * avg_depth_loss
        eef_ema = beta * eef_ema + (1.0 - beta) * avg_eef_loss
    depth_ema_hist.append(depth_ema)
    eef_ema_hist.append(eef_ema)

    if lambda_stage == "cot_pretrain" and (epoch + 1) >= int(args.lambda_plateau_min_epochs):
        w = int(args.lambda_plateau_window)
        if len(depth_ema_hist) > w:
            prev_d = depth_ema_hist[-1 - w]
            prev_e = eef_ema_hist[-1 - w]
            imp_d = (prev_d - depth_ema_hist[-1]) / max(abs(prev_d), 1e-8)
            imp_e = (prev_e - eef_ema_hist[-1]) / max(abs(prev_e), 1e-8)
            drop_d = (init_depth_ema - depth_ema_hist[-1]) / max(abs(init_depth_ema), 1e-8)
            drop_e = (init_eef_ema - eef_ema_hist[-1]) / max(abs(init_eef_ema), 1e-8)
            is_plateau = (
                imp_d < float(args.lambda_plateau_eps_depth)
                and imp_e < float(args.lambda_plateau_eps_eef)
                and drop_d >= float(args.lambda_min_cot_drop)
                and drop_e >= float(args.lambda_min_cot_drop)
            )
            plateau_count = plateau_count + 1 if is_plateau else 0
            if plateau_count >= int(args.lambda_plateau_patience):
                lambda_stage = "ramp"
                ramp_start_epoch = epoch + 1
                plateau_count = 0
                print(colored(
                    f"[adaptive-lambda] trigger ramp at epoch={epoch + 1} "
                    f"(imp_depth={imp_d:.4f}, imp_eef={imp_e:.4f}, drop_depth={drop_d:.4f}, drop_eef={drop_e:.4f})",
                    "cyan",
                ))

    if epoch % 10 == 0 :
        print(colored(
            f"epoch: {epoch}, total: {avg_loss_train:.6f}, fm: {avg_fm_loss:.6f}, depth: {avg_depth_loss:.6f}, eef: {avg_eef_loss:.6f}, "
            f"lambdas(fm/depth/eef)=({lambda_fm:.3f}/{lambda_depth:.3f}/{lambda_eef:.3f}), "
            f"stage={lambda_stage}, cot_ema(depth/eef)=({depth_ema:.6f}/{eef_ema:.6f})",
            'yellow'
        ))    
        

    if args.save_cp and epoch in eval_epoch_milestones: # save checkpint at some intervals
        cp_save_path = "./checkpoints/libero/unet_qwen/"
        os.makedirs(cp_save_path, exist_ok=True)
        ckpt = {
            'vision_encoder': nets['vision_encoder'].state_dict(),
            'noise_pred_net': nets['noise_pred_net'].state_dict(),
            'cot_depth_head': nets['cot_depth_head'].state_dict(),
            'cot_eef_head': nets['cot_eef_head'].state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            "args": vars(args),
        }


        if not args.frozen_text_model:
            ckpt['text_encoder_lora'] = get_peft_model_state_dict(nets['text_encoder'])
     
        torch.save(ckpt, f'{cp_save_path}/cp-{args.cp_name}-{epoch}.pth')

    # do evaluation below - inference
    if  args.debug or args.eval_cp or (args.eval_realtime and epoch in eval_epoch_milestones):
        if args.eval_cp: # if eval_cp is not none, then load the checkpoint
            nets.eval()
            state_dict = eval_state_dict
            saved_args = state_dict.get("args") or {}
            compatibility_keys = [
                "net",
                "obs_horizon",
                "action_horizon",
                "pred_horizon",
                "text_model",
                "text_max_len",
                "text_pool",
                "frozen_text_model",
            ]
            incompatible = [
                f"{key}: current={getattr(args, key)!r}, checkpoint={saved_args.get(key)!r}"
                for key in compatibility_keys
                if key in saved_args and saved_args.get(key) != getattr(args, key)
            ]
            if incompatible:
                raise RuntimeError(
                    "Checkpoint args are incompatible with current args: "
                    + "; ".join(incompatible)
                )

            nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])

            if not args.frozen_text_model:
                assert 'text_encoder_lora' in state_dict, 'eval on trained model, text_encoder_lora must exist'
                set_peft_model_state_dict(nets['text_encoder'], state_dict['text_encoder_lora'])
            else:
                assert 'text_encoder_lora' not in state_dict, 'eval on frozen model, text_encoder_lora should not be there'


            nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
            if 'cot_depth_head' in state_dict:
                nets.cot_depth_head.load_state_dict(state_dict['cot_depth_head'])
            if 'cot_eef_head' in state_dict:
                nets.cot_eef_head.load_state_dict(state_dict['cot_eef_head'])
            print('load official checkpoint success')
        else:
            nets.eval()

        

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
                if args.debug:
                    if task_suite_name != 'libero_spatial' or task_id not in [1, 2, 3, 4]:
                        continue

                replay_images = []
                task = task_suite.get_task(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, random.randint(1, 10000))

                if args.debug:
                    print(f'debug mode --> task_id:{task_id}-->{task_description}')

                # Start episodes
                print(f'task_id:{task_id} task_description:{task_description}')
                
                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < 50:
                    print(task_id , 'initial_states length < 50 :', len(initial_states))
                
                n_test_actual = min(len(initial_states), args.n_test)
                assert n_test_actual >= 10
                # print('initial_states cnt:', len(initial_states), '\n')
                # continue

                n_success = 0
                for trail_ix in range(n_test_actual):

                    # Reset environment
                    env.reset()

                    # Set initial states
                    obs = env.set_init_state(initial_states[trail_ix])
                    
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    for _ in range( num_steps_wait):
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        replay_images.append(obs["agentview_image"][::-1, ::-1, :])
                    # print( 'obs:', obs.keys()) # odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'agentview_image', 'robot0_eye_in_hand_image', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 'cookies_1_pos', 'cookies_1_quat', 'cookies_1_to_robot0_eef_pos', 'cookies_1_to_robot0_eef_quat', 'glazed_rim_porcelain_ramekin_1_pos', 'glazed_rim_porcelain_ramekin_1_quat', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_pos', 'glazed_rim_porcelain_ramekin_1_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state'])
                     
                    assert obs["agentview_image"].shape == (256, 256, 3) and obs["robot0_eye_in_hand_image"].shape == (256, 256, 3) , 'inference images shape check'
                    for img_ in [obs["agentview_image"], obs["robot0_eye_in_hand_image"]]:
                        assert img_.min() >= 0 and img_.max() > 1 and img_.max() < 256
                    
                    obs_deque = collections.deque([obs] * args.obs_horizon, maxlen = args.obs_horizon)
                    assert len(obs_deque) == args.obs_horizon
                    done = False
                    # Setup
                    step_idx = 0

                    while not done:
                        B = 1
                        for x in obs_deque:
                            assert x["agentview_image"].shape == (256, 256, 3) and x["agentview_image"].dtype == np.uint8
                            assert x["robot0_eye_in_hand_image"].shape == (256, 256, 3) and x["robot0_eye_in_hand_image"].dtype == np.uint8
                        
                        if args.save_image and step_idx in [0, 50, 100, 150, 200]:
                            x = obs_deque[-1]
                            # save original images (HWC, unit 8)
                            Image.fromarray(x["agentview_image"]).save(f"saved_images/libero_agentview_image_{step_idx}.png")
                            Image.fromarray(x["robot0_eye_in_hand_image"]).save(f"saved_images/libero_robot0_eye_in_hand_image_{step_idx}.png")

                            # 180 degree flip, still HWC unit 8
                            Image.fromarray(np.ascontiguousarray(x["agentview_image"][::-1, ::-1, :])).save(f"saved_images/libero_agentview_image_flip_{step_idx}.png")
                            Image.fromarray(np.ascontiguousarray(x["robot0_eye_in_hand_image"][::-1, ::-1, :])).save(f"saved_images/libero_robot0_eye_in_hand_image_flip_{step_idx}.png")                        
                            
                        
                        x_main_img = np.stack([
                            np.ascontiguousarray(
                                x["agentview_image"][::-1, ::-1, :].transpose(2,0,1) # IMPORTANT: rotate 180 degrees to match train preprocessing
                            )
                            for x in obs_deque
                        ], axis=0)# (O,3,256,256)
                        x_main_img = x_main_img[None]   # (1,O,3,256,256)
                        
                        x_wrist_image = np.stack([
                            np.ascontiguousarray(
                                x["robot0_eye_in_hand_image"][::-1, ::-1, :].transpose(2,0,1) # IMPORTANT: rotate 180 degrees to match train preprocessing
                            )
                            for x in obs_deque
                        ], axis=0)   # (O,3,256,256)                    
                        x_wrist_image = x_wrist_image[None] # (1,O,3,256,256)
  
                        x_pos = np.stack([get_state(x) for x in obs_deque])[None, ...]
                      
                        assert isinstance(x_main_img, np.ndarray) and isinstance(x_wrist_image, np.ndarray) and isinstance(x_pos, np.ndarray)
                        assert x_main_img.max() > 1 and x_wrist_image.max() > 1 and x_main_img.min() >= 0 and x_wrist_image.min() >= 0
                        # print('infer x_main_img:', x_main_img.shape)
                        # print('infer x_wrist_image:', x_wrist_image.shape)
                        # print('infer x_pos:', x_pos.shape)
                        
                        if args.normalize_images_01:
                            x_main_img = x_main_img.astype(np.float32) / 255.0
                            x_wrist_image = x_wrist_image.astype(np.float32) / 255.0
                            if args.debug:
                                assert x_main_img.min() >= 0.0 and x_main_img.max() <= 1.0, "eval x_main_img range error after normalize"
                                assert x_wrist_image.min() >= 0.0 and x_wrist_image.max() <= 1.0, "eval x_wrist_image range error after normalize"
                                             
                        x_main_img = torch.from_numpy(x_main_img).to(device, dtype=torch.bfloat16)
                        x_wrist_image = torch.from_numpy(x_wrist_image).to(device, dtype=torch.bfloat16)
                        x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.bfloat16)
                        if args.debug:
                            assert x_main_img.dtype == torch.bfloat16 and x_wrist_image.dtype == torch.bfloat16
                            assert x_pos.dtype == torch.bfloat16

                        assert x_main_img.shape == (B, args.obs_horizon, 3, 256, 256) == x_wrist_image.shape
                        assert x_pos.shape == (B, args.obs_horizon, 8)
                        with torch.no_grad():
                             
                            image_main_features_visencoder = nets['vision_encoder'](x_main_img.flatten(end_dim=1).to(dtype=torch.bfloat16))
                            image_wrist_features_visencoder = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1).to(dtype=torch.bfloat16))
                            assert image_main_features_visencoder.shape == image_wrist_features_visencoder.shape 
                            assert image_main_features_visencoder.shape == (B*args.obs_horizon, 512), f'assert shape error: {image_main_features_visencoder.shape}'
                            
                            main_feat  = image_main_features_visencoder.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
                            wrist_feat = image_wrist_features_visencoder.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]                        
                                            
                            assert main_feat.shape == wrist_feat.shape == (B, args.obs_horizon, 512)
                            
                            if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
                                x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
                            else:
                                x_pos_rep = x_pos  # 已经是 [B,O,8] 或 O=1                            
                            assert x_pos_rep.shape == (B, args.obs_horizon, 8)
                            
                            # print('infer x_pos_rep:', x_pos_rep.shape) 
                            text_enc = tokenizer(
                                [task_description],
                                padding="max_length",
                                truncation=True,
                                max_length=args.text_max_len,
                                return_tensors="pt",
                            )
                            x_task_ids = text_enc["input_ids"].to(device, non_blocking=True)
                            x_task_mask = text_enc["attention_mask"].to(device, non_blocking=True)
                            if args.debug:
                                assert x_task_ids.dtype == torch.long and x_task_mask.dtype in (torch.long, torch.int64, torch.bool)
                            text_out = nets['text_encoder'](input_ids=x_task_ids, attention_mask=x_task_mask)
                            last_hidden = text_out.last_hidden_state  # [1, L, D]
                            if args.text_pool == "last":
                                idx = x_task_mask.sum(dim=1) - 1
                                idx = idx.clamp(min=0)
                                text_feat = last_hidden[torch.arange(last_hidden.size(0), device=device), idx]
                            else:
                                mask = x_task_mask.unsqueeze(-1)
                                text_feat = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                            text_feat = text_feat.to(dtype=torch.bfloat16)
                            text_feat = text_feat.unsqueeze(1).expand(B, main_feat.shape[1], -1)  # [B, O, D]
                            if args.debug:
                                assert text_feat.shape[:2] == main_feat.shape[:2]
                                assert text_feat.dtype == torch.bfloat16
                            if args.disable_text_input:
                                obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep], dim=-1)
                            else:
                                obs_features = torch.cat([ main_feat,  wrist_feat,  x_pos_rep, text_feat], dim=-1)
                            # print('infer obs_features:', obs_features.shape) 

                        # expected dimension of per-timestep cond                       
                        if args.disable_text_input:
                            expected_obs_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1]
                        else: 
                            expected_obs_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1] + text_feat.shape[-1]

                        B, O, D = obs_features.shape
                        # 1) check obs_features 
                        assert obs_features.shape == (B, O, expected_obs_dim), f"obs_features shape wrong: got {obs_features.shape}, expect {(B, O, expected_obs_dim)}"

                        obs_context = obs_features.mean(dim=1)  # [B, D_obs]
                        depth_pred = nets["cot_depth_head"](obs_context)  # [B, D_text]
                        eef_pred = nets["cot_eef_head"](obs_context)      # [B, D_text]
                        depth_cond = depth_pred.unsqueeze(1).expand(-1, O, -1)
                        eef_cond = eef_pred.unsqueeze(1).expand(-1, O, -1)
                        fm_cond = torch.cat([obs_features, depth_cond, eef_cond], dim=-1)
                        expected_cond_dim = expected_obs_dim + depth_cond.shape[-1] + eef_cond.shape[-1]
                        assert fm_cond.shape == (B, O, expected_cond_dim), f"fm_cond shape wrong: got {fm_cond.shape}, expect {(B, O, expected_cond_dim)}"

                        # 2) check the shape before feed to nets                 

                        if args.net == 'ConditionalUnet1D':
                            obs_cond = fm_cond.view(B, O * expected_cond_dim)  
                            assert obs_cond.shape == (B, O * expected_cond_dim)
                        elif args.net == 'TransformerForDiffusion':
                            obs_cond = fm_cond
                            assert obs_cond.shape == (B, O, expected_cond_dim)
                            
                        timehorion = 16 
                        
                        fm_cond = fm_cond.to(dtype=torch.bfloat16)
                        if args.net == 'ConditionalUnet1D':
                            obs_cond = obs_cond.to(dtype=torch.bfloat16)
                        if args.debug:
                            assert fm_cond.dtype == torch.bfloat16
                            if args.net == 'ConditionalUnet1D':
                                assert obs_cond.dtype == torch.bfloat16
                        x0 = torch.rand(B, args.pred_horizon, action_dim, device=device, dtype=torch.bfloat16) # noise
                        for i in range(timehorion):
                            # x0 = torch.rand(B, args.pred_horizon, action_dim, device=device) # noise
                            timestep = torch.tensor([i / timehorion], device=device, dtype=torch.bfloat16)
                            if args.debug:
                                assert timestep.dtype == torch.bfloat16
                            
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

                        action_pred = traj.detach().to(dtype=torch.float32, device='cpu').numpy()
                        # print('action_pred:', action_pred.shape) # (1, 16, 7)
                        assert action_pred.shape == (1, 16, 7)
                        start = args.obs_horizon - 1
                        end = start + args.action_horizon

                        # execute action_horizon number of steps
                        for action in action_pred[0][start:end, :]:    
                            assert (7,) == action.shape
                            # Execute action in environment
                            obs, reward, done, info = env.step(action.tolist())
                            replay_images.append(obs["agentview_image"][::-1, ::-1, :])
                            step_idx += 1
                            assert 'agentview_image' in obs.keys() and 'robot0_eye_in_hand_image' in obs.keys()
                            
                            assert obs["agentview_image"].shape == (256, 256, 3) and obs["robot0_eye_in_hand_image"].shape == (256, 256, 3) , 'inference images shape check'
                            for img_ in [obs["agentview_image"], obs["robot0_eye_in_hand_image"]]:
                                assert img_.min() >= 0 and img_.max() > 1 and img_.max() < 256
                                
                            obs_deque.append(obs)
                            
                            if done:
                                print(f'trial {trail_ix} succeed at step: {step_idx}')
                                n_success += 1
                                break
                            
                            if step_idx > max_steps:
                                done = True
                                print(f'trial {trail_ix} fail')
                                break 

                epoch_ = 'eval_cp'
                if args.save_video:
                    imageio.mimwrite( f"./saved_videos/rollout_{args.video_name}_epoch_{epoch_}_{task_suite_name}_taskid_{task_id}_success_{n_success}_total_{n_test_actual}.mp4", [np.asarray(x) for x in replay_images], fps=10)
                    
                print(f'task summary --> epoch: {epoch_} suite: {task_suite_name} task_id: {task_id} ({task_description}); success rate:{n_success / n_test_actual} n_test:{n_test_actual}')
                print()
                env.close()

 
            print('-'*20)    

    if args.eval_cp or (args.debug and epoch > 3 )  :
        print('debug or test mode')
        os._exit(0)
