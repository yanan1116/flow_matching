import torch,os,sys
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from datasets import load_dataset
from torch.utils.data import DataLoader

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
      state:        (1, 8)  只取 obs 的第 0 帧
      actions:      (H, 7)  取 obs 之后的未来 H 步
    """

    def __init__(self, base_ds, horizon=16, obs_horizon=1, normalize_images_01=True):
        self.base = base_ds
        self.H = int(horizon)
        self.O = int(obs_horizon)
        assert self.H > 0 and self.O > 0
        self.normalize_images_01 = normalize_images_01

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

        return {"actions": actions, "state": state0, "image": images, "wrist_image": wrist_images}




def main():
    # param: 
    # obs_horizon 
    # normalize_images_01  
    # horizon

    # ds_name = 'HuggingFaceVLA/libero' # 
    ds_name = "physical-intelligence/libero"
    base_ds = load_dataset(ds_name, split="train")  # :contentReference[oaicite:3]{index=3}
    print('base_ds info:', base_ds, '\n', base_ds.features)
    print(type(base_ds[0]["image"]))
    ds = LiberoWindowedDataset(base_ds, horizon=16, obs_horizon=1, normalize_images_01=False)

    dataloader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)
    
    for ii, batch in enumerate(dataloader):
        # batch = next(iter(dataloader))
        print('batch:', ii)
        print('actions:', batch["actions"].shape)       # torch.Size([64, 16, 7])
        print('state:', batch["state"].shape)         # torch.Size([64, 1, 8])
        print('image:', batch["image"].shape)         # torch.Size([64, 1, 3, 256, 256])
        print('wrist_image:', batch["wrist_image"].shape)   # torch.Size([64, 1, 3, 256, 256])
        print()
        
if __name__ == "__main__":
    main()