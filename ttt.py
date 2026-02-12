import torch
from torch.utils.data import Dataset

def _to_chw_float(img, normalize_01=True):
    """
    img: 通常是 numpy array (H, W, 3) 或 PIL.Image
    return: torch.FloatTensor (3, H, W)
    """
    t = torch.as_tensor(img)  # uint8, shape (H,W,3)
    if t.ndim != 3 or t.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={tuple(t.shape)}")
    t = t.permute(2, 0, 1).contiguous()  # CHW
    t = t.float()
    if normalize_01:
        t = t / 255.0
    return t

class LiberoWindowedDataset(Dataset):
    """
    base_ds: HF datasets 的 split，比如 load_dataset(..., split="train")
             每条是单步，包含 keys: image, wrist_image, state, actions, episode_index
             这些字段在 dataset card 的 meta/info.json 里声明。:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, base_ds, horizon=16, obs_horizon=1, normalize_images_01=True):
        self.base = base_ds
        self.H = horizon
        self.obs_h = obs_horizon  # 你这里希望 obs_horizon=1
        self.normalize_images_01 = normalize_images_01

        # 用列操作更快：episode_index 是 shape (1,) 的 int64（但读出来通常是标量或长度1数组）:contentReference[oaicite:2]{index=2}
        eps = self.base["episode_index"]  # list-like

        # 兼容 episode_index 可能是 [x] 或 x
        def ep_scalar(x):
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return int(x[0])
            # HF 有时会给 numpy scalar / torch scalar
            try:
                return int(x)
            except Exception:
                return int(x[0])

        episodes = {}
        for i, e in enumerate(eps):
            ep = ep_scalar(e)
            episodes.setdefault(ep, []).append(i)

        # 生成所有滑窗 (idx_list, start)
        self.windows = []
        for ep, idxs in episodes.items():
            L = len(idxs)
            if L >= self.H:
                for s in range(0, L - self.H + 1):
                    self.windows.append((idxs, s))

        print(f"[LiberoWindowedDataset] episodes={len(episodes)}, windows={len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        idxs, s = self.windows[idx]
        win = idxs[s : s + self.H]  # 连续 16 步

        # actions: [16,7]
        actions = torch.stack(
            [torch.as_tensor(self.base[i]["actions"], dtype=torch.float32) for i in win],
            dim=0,
        )

        # 只取窗口第0帧的 obs（你想要 [1,...]）
        i0 = win[0]
        state0 = torch.as_tensor(self.base[i0]["state"], dtype=torch.float32).view(1, -1)  # [1,8]

        img0 = _to_chw_float(self.base[i0]["image"], normalize_01=self.normalize_images_01).unsqueeze(0)        # [1,3,256,256]
        wrist0 = _to_chw_float(self.base[i0]["wrist_image"], normalize_01=self.normalize_images_01).unsqueeze(0) # [1,3,256,256]

        return {
            "actions": actions,        # [16,7]
            "state": state0,           # [1,8]
            "image": img0,             # [1,3,256,256]
            "wrist_image": wrist0,     # [1,3,256,256]
        }


from datasets import load_dataset
from torch.utils.data import DataLoader

# param: 
# obs_horizon 
# normalize_images_01  
# horizon
base_ds = load_dataset("physical-intelligence/libero", split="train")  # :contentReference[oaicite:3]{index=3}
ds = LiberoWindowedDataset(base_ds, horizon=16, obs_horizon=1, normalize_images_01=True)

dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

batch = next(iter(dl))
print(batch["actions"].shape)       # torch.Size([64, 16, 7])
print(batch["state"].shape)         # torch.Size([64, 1, 8])
print(batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
print(batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])

