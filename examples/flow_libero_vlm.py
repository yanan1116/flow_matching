#!/usr/bin/env python
import argparse
import collections
import functools
import imageio
import os
import random
import sys
sys.dont_write_bytecode = True
sys.path.append('./external/models')
sys.path.append('./external')

LIBERO_ROOT = "/home/yanan/robotics/LIBERO"
if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)
       
import numpy as np
import torch
from tqdm import tqdm
from diffusers.training_utils import EMAModel
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from diffusers.optimization import get_scheduler
from termcolor import colored
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoProcessor
# from utils import *
from datasets import load_dataset
from libero.libero import benchmark
from libero.libero import get_libero_path
print('bddl files path:', get_libero_path("bddl_files"))

from PIL import Image
from utils import (_ensure_pil_rgb, 
            _get_libero_env, _quat2axisangle, get_state, 
            hf_transform,  LIBERO_ENV_RESOLUTION, LIBERO_DUMMY_ACTION 
)
from model_builders import build_flow_libero_vlm_model

benchmark_dict = benchmark.get_benchmark_dict()

num_steps_wait = 10
video_out_path: str = "./saved_videos"

print = functools.partial(print, flush=True)

# Avoid tokenizer parallelism warnings when DataLoader forks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Throughput-oriented defaults for CUDA kernels
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def _to_hwc_uint8_tensor(img):
    # Return CPU HWC uint8 tensor for the VLM processor path.
    if torch.is_tensor(img):
        t = img
        if t.ndim == 3 and t.shape[0] == 3:
            t = t.permute(1, 2, 0).contiguous()
        if t.dtype != torch.uint8:
            t = t.clamp(0, 255).to(torch.uint8)
        return t
    else:
        arr = np.array(_ensure_pil_rgb(img), dtype=np.uint8, copy=True)
        return torch.from_numpy(arr)

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
        _ = normalize_images_01
        
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

        images = torch.stack([_to_hwc_uint8_tensor(self.base[i]["image"])
                              for i in obs_ids], dim=0)
        wrist_images = torch.stack([_to_hwc_uint8_tensor(self.base[i]["wrist_image"])
                                    for i in obs_ids], dim=0)
        sample = {"actions": actions, "state": state0, "image": images, "wrist_image": wrist_images}
        ti = int(self.task_index_arr[obs_ids[0]])
        sample["task_text"] = self.task_map[ti]
        
        return sample

def _batch_to_hwc_uint8_np(imgs):
    # Accept CPU uint8 image tensors in [B,O,H,W,3] or [B,O,3,H,W] form.
    t = imgs
    if t.ndim != 5:
        raise ValueError(f"Unexpected image batch shape: {tuple(t.shape)}")
    if t.device.type != "cpu":
        t = t.cpu()
    if t.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 images for VLM path, got {t.dtype}")
    if t.shape[-1] == 3:
        return t.contiguous().numpy()
    return t.permute(0, 1, 3, 4, 2).contiguous().numpy()


def _build_task_prompt_map(vlm_processor, task_texts):
    use_chat_template = hasattr(vlm_processor, "apply_chat_template")
    if not use_chat_template:
        return {task_text: task_text for task_text in task_texts}

    prompt_map = {}
    for task_text in task_texts:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": task_text},
            ],
        }]
        prompt_map[task_text] = vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return prompt_map

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

def _processor_call_compat(vlm_processor, texts, image_pairs, text_max_len):
    proc_kwargs = dict(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    _ = text_max_len
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

def _build_vlm_inputs_from_batch(vlm_processor, task_prompt_map, text_max_len, x_main_img, x_wrist_image, task_texts):
    B, O = x_main_img.shape[:2]
    main_imgs_hwc = _batch_to_hwc_uint8_np(x_main_img)
    wrist_imgs_hwc = _batch_to_hwc_uint8_np(x_wrist_image)
    prompts = []
    image_pairs = []
    for b in range(B):
        prompt = task_prompt_map[task_texts[b]]
        for o in range(O):
            prompts.append(prompt)
            image_pairs.append([main_imgs_hwc[b, o], wrist_imgs_hwc[b, o]])
    vlm_inputs = _processor_call_compat(vlm_processor, prompts, image_pairs, text_max_len)
    return vlm_inputs, (B, O)


class VLMCollator:
    def __init__(self, vlm_processor, task_prompt_map, text_max_len, keep_raw_images=False):
        self.vlm_processor = vlm_processor
        self.task_prompt_map = task_prompt_map
        self.text_max_len = text_max_len
        self.keep_raw_images = keep_raw_images

    def __call__(self, batch):
        task_texts = [sample["task_text"] for sample in batch]
        batch_wo_text = []
        for sample in batch:
            item = dict(sample)
            item.pop("task_text")
            batch_wo_text.append(item)

        out = default_collate(batch_wo_text)
        vlm_inputs, obs_shape = _build_vlm_inputs_from_batch(
            vlm_processor=self.vlm_processor,
            task_prompt_map=self.task_prompt_map,
            text_max_len=self.text_max_len,
            x_main_img=out["image"],
            x_wrist_image=out["wrist_image"],
            task_texts=task_texts,
        )
        out["vlm_inputs"] = vlm_inputs
        out["vlm_obs_shape"] = obs_shape
        if self.keep_raw_images:
            out["task_text"] = task_texts
        else:
            out.pop("image")
            out.pop("wrist_image")
        return out


def encode_vlm_obs_features(vlm_encoder, vlm_inputs, obs_shape, device):
    for k, v in list(vlm_inputs.items()):
        if torch.is_tensor(v):
            vlm_inputs[k] = v.to(device=device, non_blocking=True)
    out = vlm_encoder(**vlm_inputs, output_hidden_states=True, return_dict=True)
    pooled = _extract_pooled_feature(out, attention_mask=vlm_inputs.get("attention_mask"))
    pooled = pooled.to(dtype=torch.bfloat16)
    B, O = obs_shape
    return pooled.view(B, O, -1)

assert torch.cuda.is_available()
device = 'cuda'
parser = argparse.ArgumentParser()
parser.add_argument("--net", type=str, default="ConditionalUnet1D", choices=["TransformerForDiffusion", "ConditionalUnet1D"])
parser.add_argument("--debug", action="store_true")
parser.add_argument("--normalize_images_01", action="store_true")
parser.add_argument("--n_test", type=int, default=50)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--save_interval", type=int, default=50) ###
parser.add_argument("--obs_horizon", type=int, default=1)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--pred_horizon", type=int, default=16)
parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
parser.add_argument("--vlm_lora_r", type=int, default=16)
parser.add_argument("--vlm_lora_alpha", type=int, default=32)
parser.add_argument("--vlm_lora_dropout", type=float, default=0.0)
parser.add_argument("--text_max_len", type=int, default=64)
parser.add_argument("--frozen_vlm", action="store_true") ###
parser.add_argument("--save_image", action='store_true')
parser.add_argument("--save_video", action='store_true')
parser.add_argument("--eval_cp", type=str, default=None)
parser.add_argument("--cp_name", type=str, default='')
parser.add_argument("--video_name", type=str, default="")
args = parser.parse_args() 
print('args:', args)
eval_state_dict = None
if args.eval_cp:
    eval_state_dict = torch.load(args.eval_cp, map_location='cpu')
 
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
vlm_processor = AutoProcessor.from_pretrained(args.vlm_model, trust_remote_code=True)
task_prompt_map = _build_task_prompt_map(vlm_processor, task_texts)
vlm_collator = VLMCollator(
    vlm_processor=vlm_processor,
    task_prompt_map=task_prompt_map,
    text_max_len=args.text_max_len,
    keep_raw_images=(args.debug or args.save_image),
)
base_ds = base_ds.with_transform(hf_transform)
ds = LiberoWindowedDataset(base_ds, 
                           horizon=args.pred_horizon, 
                           obs_horizon=args.obs_horizon, 
                           normalize_images_01=args.normalize_images_01, 
                           task_map=task_map)

dataloader_kwargs = dict(
    dataset=ds,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=(args.num_workers > 0),
    collate_fn=vlm_collator,
)
if args.num_workers > 0:
    dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
dataloader = DataLoader(**dataloader_kwargs)

batch = next(iter(dataloader))
print(batch.keys())
print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
if "image" in batch:
    print('image:', batch["image"].shape)         # torch.Size([64, 1, 256, 256, 3])
    print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 256, 256, 3])
if "task_text" in batch:
    print('task_text:', len(batch['task_text']))
    assert isinstance(batch["task_text"], list)
for k, v in batch["vlm_inputs"].items():
    if torch.is_tensor(v):
        print(f'vlm_inputs[{k}]:', v.shape, v.dtype)
# for tt in batch['task_text']:
#     print(tt)


# os._exit(0)

if args.save_image:
    os.makedirs('./saved_images', exist_ok=True)
    imgs = batch["image"]   # shape: [B, O, H, W, 3]
    B, O, H, W, C = imgs.shape
    N = 5
    idxs = random.sample(range(B), min(N, B))
    for i, idx in enumerate(idxs):
        img = imgs[idx, 0]
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(img).save(f"saved_images/libero_hf_image_{i}.png")
# images are in normal orientation


assert torch.cuda.is_available(), "CUDA is required for bf16 training"
assert torch.cuda.is_bf16_supported(), "GPU does not support bf16"
# create network object
nets = build_flow_libero_vlm_model(
    args=args,
    device=device,
    action_dim=action_dim,
    eval_state_dict=eval_state_dict,
)
        
##################################################################
sigma = 0.0
ema_params = [p for p in nets.parameters() if p.requires_grad]
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

        
#### Train the model
if not args.eval_cp:
    for epoch in tqdm(range( args.num_epochs ), desc="Training Epochs"):

        total_loss_train = 0.0
        
        nets.train()
        if args.frozen_vlm:
            nets['vlm_encoder'].eval()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="it", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]")
        optimizer.zero_grad(set_to_none=True)
        for ii, batch in enumerate(pbar):

            if args.debug:
                batch_wrist_image_min, batch_wrist_image_max = batch['wrist_image'].min().item(), batch['wrist_image'].max().item()
                batch_main_image_min, batch_main_image_max = batch['image'].min().item(), batch['image'].max().item()
                assert batch['wrist_image'].dtype == torch.uint8 and batch['image'].dtype == torch.uint8
                assert batch_wrist_image_min >= 0 and (1 <= batch_wrist_image_max <= 255), 'wrist_image range error'
                assert batch_main_image_min >= 0 and (1 <= batch_main_image_max <= 255), 'image range error'
                
                assert batch['actions'].min() >= -1 and batch['actions'].max() <= 1, 'actions range error'
            
                batch_state_min = batch['state'].min()
                batch_state_max = batch['state'].max()
                assert batch_state_min >= -3.14*2 and batch_state_max <= 3.14*2, f'state range error: {batch_state_min} {batch_state_max}'

            x_pos = batch['state'].to(device, non_blocking=True).to(dtype=torch.bfloat16)
            x_traj = batch['actions'].to(device, non_blocking=True).to(dtype=torch.bfloat16)

            if args.debug:
                x_main_img = batch['image']
                x_wrist_image = batch['wrist_image']
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

            if args.frozen_vlm:
                with torch.inference_mode():
                    vlm_feat = encode_vlm_obs_features(
                        vlm_encoder=nets['vlm_encoder'],
                        vlm_inputs=batch["vlm_inputs"],
                        obs_shape=batch["vlm_obs_shape"],
                        device=device,
                    )
            else:
                vlm_feat = encode_vlm_obs_features(
                    vlm_encoder=nets['vlm_encoder'],
                    vlm_inputs=batch["vlm_inputs"],
                    obs_shape=batch["vlm_obs_shape"],
                    device=device,
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

            
            if args.debug and ii >= 8 :
                break
        
        avg_loss_train = total_loss_train / len(dataloader)
        print(colored(f"epoch: {epoch},  loss_train: {avg_loss_train:.6f}", 'yellow'))
            
        
        # save checkpoint
        if (epoch > 0 and epoch % args.save_interval  == 0 and args.cp_name) or args.debug :
            cp_save_path = "./checkpoints/libero/vlm/"
            os.makedirs(cp_save_path, exist_ok=True)

            if  args.debug :
                epoch = 'debug'
                
            ckpt = {
                'noise_pred_net': nets['noise_pred_net'].state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                "args": vars(args),
            }
            if not args.frozen_vlm:
                ckpt['vlm_encoder_lora'] = get_peft_model_state_dict(nets['vlm_encoder'])
            torch.save(ckpt, f'{cp_save_path}/cp-{args.net}-{args.cp_name}-{epoch}.pth')

        if args.debug:
            break

                
                    

else:
    # do evaluation below - inference
    nets.eval()
    state_dict = eval_state_dict
    if 'vlm_encoder_lora' in state_dict:
        if not hasattr(nets['vlm_encoder'], "peft_config"):
            raise RuntimeError("Checkpoint contains LoRA weights but VLM LoRA was not initialized.")
        set_peft_model_state_dict(nets['vlm_encoder'], state_dict['vlm_encoder_lora'])
    elif 'vlm_encoder' in state_dict:
        nets['vlm_encoder'].load_state_dict(state_dict['vlm_encoder'])
    elif not args.frozen_vlm:
        raise KeyError("Checkpoint missing key 'vlm_encoder_lora' or 'vlm_encoder'")
    nets['noise_pred_net'].load_state_dict(state_dict['noise_pred_net'])
    print('load checkpoint success')    
    
    nets['vlm_encoder'].eval()

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
                            x["agentview_image"][::-1, ::-1, :] # IMPORTANT: rotate 180 degrees to match train preprocessing
                        )
                        for x in obs_deque
                    ], axis=0) # (O,256,256,3)
                    x_main_img = x_main_img[None]   # (1,O,256,256,3)
                    
                    x_wrist_image = np.stack([
                        np.ascontiguousarray(
                            x["robot0_eye_in_hand_image"][::-1, ::-1, :] # IMPORTANT: rotate 180 degrees to match train preprocessing
                        )
                        for x in obs_deque
                    ], axis=0)   # (O,256,256,3)
                    x_wrist_image = x_wrist_image[None] # (1,O,256,256,3)

                    x_pos = np.stack([get_state(x) for x in obs_deque])[None, ...]
                    
                    assert isinstance(x_main_img, np.ndarray) and isinstance(x_wrist_image, np.ndarray) and isinstance(x_pos, np.ndarray)
                    assert x_main_img.max() > 1 and x_wrist_image.max() > 1 and x_main_img.min() >= 0 and x_wrist_image.min() >= 0
                    # print('infer x_main_img:', x_main_img.shape)
                    # print('infer x_wrist_image:', x_wrist_image.shape)
                    # print('infer x_pos:', x_pos.shape)
                    
                    x_main_img = torch.from_numpy(x_main_img)
                    x_wrist_image = torch.from_numpy(x_wrist_image)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.bfloat16)
                    if args.debug:
                        assert x_main_img.dtype == torch.uint8 and x_wrist_image.dtype == torch.uint8
                        assert x_pos.dtype == torch.bfloat16

                    assert x_main_img.shape == (B, args.obs_horizon, 256, 256, 3) == x_wrist_image.shape
                    assert x_pos.shape == (B, args.obs_horizon, 8)
                    with torch.no_grad():
                        vlm_inputs, obs_shape = _build_vlm_inputs_from_batch(
                            vlm_processor=vlm_processor,
                            task_prompt_map=task_prompt_map,
                            text_max_len=args.text_max_len,
                            x_main_img=x_main_img,
                            x_wrist_image=x_wrist_image,
                            task_texts=[task_description],
                        )
                        vlm_feat = encode_vlm_obs_features(
                            vlm_encoder=nets['vlm_encoder'],
                            vlm_inputs=vlm_inputs,
                            obs_shape=obs_shape,
                            device=device,
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
                        

            
            if args.save_video:
                imageio.mimwrite( f"./saved_videos/rollout_{args.video_name}_{task_suite_name}_taskid_{task_id}_success_{n_success}_total_{n_test_actual}.mp4", [np.asarray(x) for x in replay_images], fps=10)
                
            print(f'task summary --> suite: {task_suite_name} task_id: {task_id} ({task_description}); success rate:{n_success / n_test_actual} n_test:{n_test_actual}')
            print()
            env.close()


        print('-'*20)    

    
