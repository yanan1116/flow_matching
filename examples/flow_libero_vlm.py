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
import transformers
from transformers import AutoTokenizer, AutoModel, AutoProcessor
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
from concurrent.futures import ThreadPoolExecutor
print = functools.partial(print, flush=True)

# Avoid tokenizer parallelism warnings when DataLoader forks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Throughput-oriented defaults for CUDA kernels
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def _to_chw_tensor(img, normalize_01=True):
    # img may already be torch.uint8 CHW (from HF with_transform)
    if torch.is_tensor(img):
        t = img
    else:
        t = pil_to_tensor(img.convert("RGB"))
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    if normalize_01:
        t = t.float() / 255.0
    return t

def _to_hwc_uint8_tensor(img):
    # Return CPU uint8 HWC tensor for VLM processor path.
    if torch.is_tensor(img):
        t = img
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
        if t.ndim == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0).contiguous()
        return t
    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(arr)

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

    def __init__(self, base_ds, horizon=16, obs_horizon=1, normalize_images_01=True, task_map=None, vlm_mode=False):
        self.base = base_ds
        self.H = int(horizon)
        self.O = int(obs_horizon)
        assert self.H > 0 and self.O > 0
        self.normalize_images_01 = normalize_images_01
        self.vlm_mode = bool(vlm_mode)
        
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

        if self.vlm_mode:
            images = torch.stack([_to_hwc_uint8_tensor(self.base[i]["image"]) for i in obs_ids], dim=0)
            wrist_images = torch.stack([_to_hwc_uint8_tensor(self.base[i]["wrist_image"]) for i in obs_ids], dim=0)
        else:
            images = torch.stack([_to_chw_tensor(self.base[i]["image"], normalize_01=self.normalize_images_01)
                                for i in obs_ids], dim=0)
            wrist_images = torch.stack([_to_chw_tensor(self.base[i]["wrist_image"], normalize_01=self.normalize_images_01)
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

def _batch_to_hwc_uint8_np(imgs, normalize_01):
    # Accept [B,O,3,H,W] or [B,O,H,W,3] tensors and convert once.
    t = imgs.detach().to("cpu")
    if t.ndim != 5:
        raise ValueError(f"Unexpected image batch shape: {tuple(t.shape)}")
    if (not normalize_01) and t.dtype == torch.uint8:
        if t.shape[-1] == 3:
            return t.contiguous().numpy()
        return t.permute(0, 1, 3, 4, 2).contiguous().numpy()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    if normalize_01:
        t = (t.clamp(0, 1) * 255.0).round()
    else:
        t = t.clamp(0, 255)
    t = t.to(torch.uint8)
    if t.shape[-1] == 3:
        return t.contiguous().numpy()
    return t.permute(0, 1, 3, 4, 2).contiguous().numpy()

def _extract_pooled_feature(model_out, attention_mask=None):
    if hasattr(model_out, "pooler_output") and model_out.pooler_output is not None:
        return model_out.pooler_output

    if hasattr(model_out, "last_hidden_state") and model_out.last_hidden_state is not None:
        h = model_out.last_hidden_state
        if h.ndim == 3 and attention_mask is not None and attention_mask.shape[0] == h.shape[0]:
            m = attention_mask.to(h.device).unsqueeze(-1).to(dtype=h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if h.ndim == 3:
            return h[:, -1]
        return h

    if hasattr(model_out, "hidden_states") and model_out.hidden_states is not None and len(model_out.hidden_states) > 0:
        h = model_out.hidden_states[-1]
        if h.ndim == 3 and attention_mask is not None and attention_mask.shape[0] == h.shape[0]:
            m = attention_mask.to(h.device).unsqueeze(-1).to(dtype=h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if h.ndim == 3:
            return h[:, -1]
        return h

    if isinstance(model_out, tuple) and len(model_out) > 0:
        h = model_out[0]
        if torch.is_tensor(h) and h.ndim == 3:
            return h[:, -1]
        if torch.is_tensor(h):
            return h

    raise RuntimeError("Cannot extract pooled feature from VLM output.")

def _build_vlm_texts_and_images(vlm_processor, prompts, image_pairs, prompt_cache=None):
    # image_pairs: List[List[np.ndarray(H, W, 3), np.ndarray(H, W, 3)]]
    use_chat_template = hasattr(vlm_processor, "apply_chat_template")
    if use_chat_template:
        texts = []
        for p in prompts:
            if prompt_cache is not None and p in prompt_cache:
                txt = prompt_cache[p]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
                txt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                if prompt_cache is not None:
                    prompt_cache[p] = txt
            texts.append(txt)
    else:
        texts = prompts
    return texts, image_pairs

def _processor_call_compat(vlm_processor, texts, image_pairs, text_max_len):
    proc_kwargs = dict(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    # For VLM multi-image prompts, truncation may cut image placeholder tokens
    # and cause processor-side image token mismatch.
    _ = text_max_len

    # Try several input formats because processor image APIs differ by model/version.
    attempts = [
        dict(proc_kwargs, images=image_pairs),
        dict(proc_kwargs, images=[img for pair in image_pairs for img in pair]),
    ]
    last_err = None
    for kw in attempts:
        try:
            return vlm_processor(**kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"VLM processor input formatting failed. Last error: {last_err}")

def _get_vlm_hidden_size(vlm_encoder):
    cfg = vlm_encoder.config
    if hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    for key in ["text_config", "language_config", "llm_config"]:
        sub_cfg = getattr(cfg, key, None)
        if sub_cfg is not None and hasattr(sub_cfg, "hidden_size"):
            return int(sub_cfg.hidden_size)
    raise RuntimeError("Cannot infer hidden size from VLM config.")

def _from_pretrained_with_loading_info(model_cls, model_name, dtype):
    # transformers has migrated from torch_dtype -> dtype on newer versions.
    common = dict(trust_remote_code=True, output_loading_info=True)
    try:
        return model_cls.from_pretrained(model_name, dtype=dtype, **common)
    except TypeError:
        return model_cls.from_pretrained(model_name, torch_dtype=dtype, **common)

def _load_vlm_encoder(model_name, dtype):
    # Prefer generation-capable VLM classes first for Qwen VL checkpoints.
    candidates = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
        AutoModel,
    ]
    last_err = None
    for cls in candidates:
        if cls is None:
            continue
        try:
            model, loading_info = _from_pretrained_with_loading_info(cls, model_name, dtype)
            missing = loading_info.get("missing_keys", []) if isinstance(loading_info, dict) else []
            if len(missing) > 128:
                raise RuntimeError(
                    f"VLM checkpoint appears partially initialized with {len(missing)} missing keys. "
                    "This usually means an incompatible model class/version was used."
                )
            return model
        except Exception as e:
            last_err = e
            continue

    ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if "Qwen3-VL" in model_name and ver < (4, 50):
        raise RuntimeError(
            f"{model_name} may require newer transformers than {transformers.__version__}. "
            f"Please upgrade transformers or use an older VLM checkpoint."
        ) from last_err
    raise RuntimeError(f"Failed to load VLM encoder for {model_name}: {last_err}")

def encode_vlm_obs_features(
    vlm_encoder,
    vlm_processor,
    x_main_img,
    x_wrist_image,
    task_texts,
    normalize_images_01,
    text_max_len,
    device,
    prompt_cache=None,
):
    B, O = x_main_img.shape[:2]
    main_imgs_hwc = _batch_to_hwc_uint8_np(x_main_img, normalize_images_01)
    wrist_imgs_hwc = _batch_to_hwc_uint8_np(x_wrist_image, normalize_images_01)
    prompts = []
    image_pairs = []
    for b in range(B):
        prompt = task_texts[b]
        for o in range(O):
            prompts.append(prompt)
            img_main = main_imgs_hwc[b, o]
            img_wrist = wrist_imgs_hwc[b, o]
            image_pairs.append([img_main, img_wrist])
    texts, image_pairs = _build_vlm_texts_and_images(vlm_processor, prompts, image_pairs, prompt_cache=prompt_cache)
    vlm_inputs = _processor_call_compat(vlm_processor, texts, image_pairs, text_max_len)
    for k, v in list(vlm_inputs.items()):
        if torch.is_tensor(v):
            vlm_inputs[k] = v.to(device=device, non_blocking=True)

    out = vlm_encoder(**vlm_inputs, output_hidden_states=True, return_dict=True)
    pooled = _extract_pooled_feature(out, attention_mask=vlm_inputs.get("attention_mask"))
    pooled = pooled.to(dtype=torch.bfloat16)
    return pooled.view(B, O, -1)


def _prepare_vlm_feat_async(
    batch,
    vlm_encoder,
    vlm_processor,
    normalize_images_01,
    text_max_len,
    vlm_device,
    train_device,
    prompt_cache,
):
    with torch.inference_mode():
        vlm_feat = encode_vlm_obs_features(
            vlm_encoder=vlm_encoder,
            vlm_processor=vlm_processor,
            x_main_img=batch["image"],
            x_wrist_image=batch["wrist_image"],
            task_texts=batch["task_text"],
            normalize_images_01=normalize_images_01,
            text_max_len=text_max_len,
            device=vlm_device,
            prompt_cache=prompt_cache,
        )
    return vlm_feat.to(device=train_device, non_blocking=True)


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
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default="ConditionalUnet1D", choices=["TransformerForDiffusion", "ConditionalUnet1D"])
# parser.add_argument("--frozen_vision", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--normalize_images_01", action="store_true")
parser.add_argument("--n_test", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--eval_interval", type=int, default=50)
parser.add_argument("--obs_horizon", type=int, default=1)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--pred_horizon", type=int, default=16)
parser.add_argument("--encoder_mode", type=str, default="separate", choices=["separate", "vlm"])
parser.add_argument("--text_model", type=str, default="Qwen/Qwen3-0.6B")
parser.add_argument("--text_max_len", type=int, default=64)
parser.add_argument("--text_pool", type=str, default="last", choices=["last", "mean"])
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen3-VL-4B-Thinking")
parser.add_argument("--frozen_vlm", action="store_true")
parser.add_argument("--train_device", type=str, default="cuda:0")
parser.add_argument("--vlm_device", type=str, default="cuda:0")
parser.add_argument("--vlm_async_pipeline", action="store_true")
parser.add_argument("--vlm_prefetch_depth", type=int, default=2)
parser.add_argument("--eval_official", action='store_true')
parser.add_argument("--save_image", action='store_true')
parser.add_argument("--save_video", action='store_true')
parser.add_argument("--save_cp", action='store_true')
parser.add_argument("--eval_cp", type=str, default="./checkpoints/libero/cp-ConditionalUnet1D-400.pth")
parser.add_argument("--video_name", type=str, default="")
args = parser.parse_args() 
print('args:', args)
device = args.train_device
vlm_device = args.vlm_device
 
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
if args.encoder_mode == "separate":
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
    vlm_processor = None
else:
    tokenizer = None
    task_text_inputs = None
    vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_prompt_cache = {}

if args.encoder_mode == "separate":
    base_ds = base_ds.with_transform(hf_transform)
ds = LiberoWindowedDataset(base_ds, 
                           horizon=args.pred_horizon, 
                           obs_horizon=args.obs_horizon, 
                           normalize_images_01=args.normalize_images_01, 
                           task_map=task_map,
                           vlm_mode=(args.encoder_mode == "vlm"))

dataloader_kwargs = dict(
    dataset=ds,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0),
    collate_fn=collate_with_task_text,
)
if args.num_workers > 0:
    dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
dataloader = DataLoader(**dataloader_kwargs)

batch = next(iter(dataloader))
print(batch.keys())
print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
print('image:', batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])
print('task_text:', len(batch['task_text']))
print('image dtype:', batch["image"].dtype, 'wrist_image dtype:', batch["wrist_image"].dtype)
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
        img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"saved_images/libero_hf_image_{i}.png")
# images are in normal orientation


assert torch.cuda.is_available(), "CUDA is required for bf16 training"
assert torch.cuda.is_bf16_supported(), "GPU does not support bf16"
# create network object
modules = {}
vlm_encoder_model = None
if args.encoder_mode == "separate":
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    text_encoder = AutoModel.from_pretrained(
        args.text_model,
        torch_dtype=torch.bfloat16,
    )
    text_embed_dim = int(text_encoder.config.hidden_size)
    per_timestep_cond_dim = 512 * 2 + 8 + text_embed_dim
    modules["vision_encoder"] = vision_encoder
    modules["text_encoder"] = text_encoder
else:
    vlm_encoder_model = _load_vlm_encoder(args.vlm_model, dtype=torch.bfloat16)
    vlm_encoder_model = vlm_encoder_model.to(vlm_device, dtype=torch.bfloat16)
    vlm_embed_dim = _get_vlm_hidden_size(vlm_encoder_model)
    per_timestep_cond_dim = vlm_embed_dim + 8

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

nets = nn.ModuleDict({**modules, 'noise_pred_net': noise_pred_net}).to(device, dtype=torch.bfloat16)

if args.encoder_mode == "vlm" and args.frozen_vlm:
    vlm_encoder_model.eval()
    for p in vlm_encoder_model.parameters():
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
pipeline_enabled = (
    args.encoder_mode == "vlm"
    and args.frozen_vlm
    and args.vlm_async_pipeline
    and (vlm_device != device)
)
if pipeline_enabled:
    print(f"[pipeline] enabled: vlm_device={vlm_device} -> train_device={device}")
    print(f"[pipeline] prefetch_depth={max(1, args.vlm_prefetch_depth)}")
vlm_run_device = vlm_device if args.encoder_mode == "vlm" else device

        
########################################################################
#### Train the model
for epoch in tqdm(range( args.num_epochs ), desc="Training Epochs"):

    if (args.debug or args.eval_official) and epoch > 0 :
        print('debug or test mode')
        os._exit(0)


    total_loss_train = 0.0
    
    nets.train()

    if args.encoder_mode == "vlm" and args.frozen_vlm:
        vlm_encoder_model.eval()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    async_executor = None
    if pipeline_enabled:
        data_iter = iter(dataloader)
        async_executor = ThreadPoolExecutor(max_workers=1)
        prefetch_q = collections.deque()
        prefetch_depth = max(1, int(args.vlm_prefetch_depth))

        def _submit_next_feat():
            try:
                b = next(data_iter)
            except StopIteration:
                return False
            fut = async_executor.submit(
                _prepare_vlm_feat_async,
                b,
                vlm_encoder_model,
                vlm_processor,
                args.normalize_images_01,
                args.text_max_len,
                vlm_device,
                device,
                vlm_prompt_cache,
            )
            prefetch_q.append((b, fut))
            return True

        for _ in range(prefetch_depth):
            if not _submit_next_feat():
                break
        ii = 0
    else:
        batch_iter = enumerate(pbar)

    while True:
        if pipeline_enabled:
            if not prefetch_q:
                break
            batch, feat_future = prefetch_q.popleft()
            vlm_feat_prefetched = feat_future.result()
            _submit_next_feat()
        else:
            try:
                ii, batch = next(batch_iter)
            except StopIteration:
                break
            vlm_feat_prefetched = None

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

        x_main_img = batch['image']
        x_wrist_image = batch['wrist_image']
        x_pos = batch['state'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        x_traj = batch['actions'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
        if args.encoder_mode == "separate":
            x_main_img = x_main_img.to(device, non_blocking=True).to(dtype=torch.bfloat16)
            x_wrist_image = x_wrist_image.to(device, non_blocking=True).to(dtype=torch.bfloat16)
            x_task_ids = torch.stack([task_text_inputs[t][0] for t in batch["task_text"]], dim=0).to(device, non_blocking=True)
            x_task_mask = torch.stack([task_text_inputs[t][1] for t in batch["task_text"]], dim=0).to(device, non_blocking=True)
        elif pipeline_enabled:
            vlm_feat = vlm_feat_prefetched.to(dtype=torch.bfloat16)

        if args.debug:
            if args.encoder_mode == "separate":
                assert x_main_img.dtype == torch.bfloat16 and x_wrist_image.dtype == torch.bfloat16
                assert x_task_ids.dtype == torch.long and x_task_mask.dtype in (torch.long, torch.int64, torch.bool)
            assert x_pos.dtype == torch.bfloat16 and x_traj.dtype == torch.bfloat16
            
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

        # encoder features
        if args.encoder_mode == "separate":

            image_main_features_visencoder = nets['vision_encoder'](x_main_img.flatten(end_dim=1).to(dtype=torch.bfloat16))
            image_wrist_features_visencoder = nets['vision_encoder'](x_wrist_image.flatten(end_dim=1).to(dtype=torch.bfloat16))

            main_feat  = image_main_features_visencoder.reshape(*x_main_img.shape[:2], -1)   # [B,O,D]
            wrist_feat = image_wrist_features_visencoder.reshape(*x_wrist_image.shape[:2], -1) # [B,O,D]

            if x_pos.shape[1] == 1 and main_feat.shape[1] > 1:
                x_pos_rep = x_pos.expand(-1, main_feat.shape[1], -1)
            else:
                x_pos_rep = x_pos  # already is of shape [B,O,8] or O=1
            if args.debug:
                assert x_pos_rep.shape[:2] == main_feat.shape[:2]

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

            obs_features = torch.cat([main_feat, wrist_feat, x_pos_rep, text_feat], dim=-1)
            expected_feat_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1] + text_feat.shape[-1]
        else:
            if pipeline_enabled:
                pass
            elif args.frozen_vlm:
                with torch.inference_mode():
                    vlm_feat = encode_vlm_obs_features(
                        vlm_encoder=vlm_encoder_model,
                        vlm_processor=vlm_processor,
                        x_main_img=x_main_img,
                        x_wrist_image=x_wrist_image,
                        task_texts=batch["task_text"],
                        normalize_images_01=args.normalize_images_01,
                        text_max_len=args.text_max_len,
                        device=vlm_run_device,
                        prompt_cache=vlm_prompt_cache,
                    )
            else:
                vlm_feat = encode_vlm_obs_features(
                    vlm_encoder=vlm_encoder_model,
                    vlm_processor=vlm_processor,
                    x_main_img=x_main_img,
                    x_wrist_image=x_wrist_image,
                    task_texts=batch["task_text"],
                    normalize_images_01=args.normalize_images_01,
                    text_max_len=args.text_max_len,
                    device=vlm_run_device,
                    prompt_cache=vlm_prompt_cache,
                )
            if vlm_feat.device != x_pos.device:
                vlm_feat = vlm_feat.to(device=x_pos.device, non_blocking=True)
            if x_pos.shape[1] == 1 and vlm_feat.shape[1] > 1:
                x_pos_rep = x_pos.expand(-1, vlm_feat.shape[1], -1)
            else:
                x_pos_rep = x_pos
            obs_features = torch.cat([vlm_feat, x_pos_rep], dim=-1)
            expected_feat_dim = vlm_feat.shape[-1] + x_pos_rep.shape[-1]

        B, O = obs_features.shape[:2]

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
        if ii % 20 == 0:
            pbar.set_postfix(loss=f"{loss.detach().item():.4f}")
        total_loss_train += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        # ema.step(nets.parameters())
        ema.step(ema_params)
        pbar.update(1 if pipeline_enabled else 0)
        
        if (args.debug and ii >= 128)  or (args.eval_official and ii >= 4 )  :
            break
        if pipeline_enabled:
            ii += 1
    if async_executor is not None:
        async_executor.shutdown(wait=False, cancel_futures=True)
    
    
    
    
    # do evaluation below - inference
    if (epoch % args.eval_interval == 0 and epoch > 0) or args.debug or args.eval_official:

        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))

        if args.save_cp:
            os.makedirs("./checkpoints/libero", exist_ok=True)
            ema.store(ema_params) 
            ema.copy_to(ema_params)

            ckpt = {
                'noise_pred_net': nets['noise_pred_net'].state_dict(),
                'epoch': epoch,
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "args": vars(args),
            }
            if args.encoder_mode == "separate":
                ckpt['vision_encoder'] = nets['vision_encoder'].state_dict()
                ckpt['text_encoder'] = nets['text_encoder'].state_dict()
            else:
                ckpt['vlm_encoder'] = vlm_encoder_model.state_dict()

            torch.save(ckpt, f'./checkpoints/libero/cp-{args.net}-{epoch}.pth')
            ema.restore(ema_params)
            
        if args.eval_official:
            nets.eval()
            state_dict = torch.load(args.eval_cp, map_location='cuda')
            if args.encoder_mode == "separate":
                nets['vision_encoder'].load_state_dict(state_dict['vision_encoder'])
                if 'text_encoder' in state_dict:
                    nets['text_encoder'].load_state_dict(state_dict['text_encoder'])
            else:
                if 'vlm_encoder' not in state_dict:
                    raise KeyError("Checkpoint missing key 'vlm_encoder' for --encoder_mode vlm")
                vlm_encoder_model.load_state_dict(state_dict['vlm_encoder'])
            nets['noise_pred_net'].load_state_dict(state_dict['noise_pred_net'])
            print('load official checkpoint success')
        

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
                replay_images = []
                task = task_suite.get_task(task_id)

                # Initialize LIBERO environment and task description
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, random.randint(1, 10000))
                # Start episodes
                print(f'task_id:{task_id} task_description:{task_description}')
                
                # Get default LIBERO initial states
                initial_states = task_suite.get_task_init_states(task_id)
                if len(initial_states) < 50:
                    print(task_id , 'initial_states length < 50 :', len(initial_states))
                
                n_test_actual = min(len(initial_states), args.n_test)

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
                            x_main_img /= 255.0  
                            x_wrist_image /= 255.0 
                                             
                        x_main_img = torch.from_numpy(x_main_img).to(device, dtype=torch.bfloat16)
                        x_wrist_image = torch.from_numpy(x_wrist_image).to(device, dtype=torch.bfloat16)
                        x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.bfloat16)
                        if args.debug:
                            assert x_main_img.dtype == torch.bfloat16 and x_wrist_image.dtype == torch.bfloat16
                            assert x_pos.dtype == torch.bfloat16

                        assert x_main_img.shape == (B, args.obs_horizon, 3, 256, 256) == x_wrist_image.shape
                        assert x_pos.shape == (B, args.obs_horizon, 8)
                        with torch.no_grad():
                            if args.encoder_mode == "separate":
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
                                    x_pos_rep = x_pos
                                assert x_pos_rep.shape == (B, args.obs_horizon, 8)

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

                                obs_features = torch.cat([main_feat, wrist_feat, x_pos_rep, text_feat], dim=-1)
                                expected_feat_dim = main_feat.shape[-1] + wrist_feat.shape[-1] + x_pos_rep.shape[-1] + text_feat.shape[-1]
                            else:
                                vlm_feat = encode_vlm_obs_features(
                                    vlm_encoder=vlm_encoder_model,
                                    vlm_processor=vlm_processor,
                                    x_main_img=x_main_img,
                                    x_wrist_image=x_wrist_image,
                                    task_texts=[task_description],
                                    normalize_images_01=args.normalize_images_01,
                                    text_max_len=args.text_max_len,
                                    device=vlm_run_device,
                                    prompt_cache=vlm_prompt_cache,
                                )
                                if vlm_feat.device != x_pos.device:
                                    vlm_feat = vlm_feat.to(device=x_pos.device, non_blocking=True)
                                if x_pos.shape[1] == 1 and vlm_feat.shape[1] > 1:
                                    x_pos_rep = x_pos.expand(-1, vlm_feat.shape[1], -1)
                                else:
                                    x_pos_rep = x_pos
                                obs_features = torch.cat([vlm_feat, x_pos_rep], dim=-1)
                                expected_feat_dim = vlm_feat.shape[-1] + x_pos_rep.shape[-1]
                        
                        B, O, D = obs_features.shape
                        # 1) check obs_features 
                        assert obs_features.shape == (B, O, expected_feat_dim), f"obs_features shape wrong: got {obs_features.shape}, expect {(B, O, expected_feat_dim)}"

                        # 2) check the shape before feed to nets                 

                        if args.net == 'ConditionalUnet1D':
                            obs_cond = obs_features.view(B, O * D)  
                            assert obs_cond.shape == (B, O * D)
                        elif args.net == 'TransformerForDiffusion':
                            obs_cond = obs_features
                            assert obs_cond.shape == (B, O, D)
                            
                        timehorion = 16 
                        
                        obs_features = obs_features.to(dtype=torch.bfloat16)
                        if args.net == 'ConditionalUnet1D':
                            obs_cond = obs_cond.to(dtype=torch.bfloat16)
                        if args.debug:
                            assert obs_features.dtype == torch.bfloat16
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
                            
                            
                
                if args.debug or args.eval_official:
                    epoch_ = 'eval_official'
                else:
                    epoch_ = epoch
                
                if args.save_video:
                    imageio.mimwrite( f"./saved_videos/rollout_{args.video_name}_epoch_{epoch_}_{task_suite_name}_taskid_{task_id}_success_{n_success}_total_{n_test_actual}.mp4", [np.asarray(x) for x in replay_images], fps=10)
                    
                print(f'task summary --> epoch: {epoch_} suite: {task_suite_name} task_id: {task_id} ({task_description}); success rate:{n_success / n_test_actual} n_test:{n_test_actual}')
                print()
                env.close()

 
            print('-'*20)    
    
    
