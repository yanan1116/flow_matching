import io
import math
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms.functional import pil_to_tensor

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


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
    if torch.is_tensor(img):
        t = img
    else:
        t = pil_to_tensor(_ensure_pil_rgb(img))
    t = t.float()
    if normalize_01:
        t = t / 255.0
    return t


def hf_transform(ex):
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


class LiberoWindowedDataset(Dataset):
    """
    sample:
      images:       (O, 3, 256, 256)
      wrist_images: (O, 3, 256, 256)
      state:        (1, 8)
      actions:      (H, 7)
    """

    def __init__(self, base_ds, horizon=16, obs_horizon=1, normalize_images_01=True, task_map=None):
        self.base = base_ds
        self.H = int(horizon)
        self.O = int(obs_horizon)
        assert self.H > 0 and self.O > 0
        self.normalize_images_01 = normalize_images_01

        self.task_map = task_map
        assert self.task_map is not None and len(self.task_map) == 40
        self.task_index_arr = np.asarray(self.base["task_index"], dtype=np.int64)

        eps = self.base["episode_index"]
        fis = self.base["frame_index"]

        self.eps = np.asarray(eps, dtype=np.int64)
        self.fis = np.asarray(fis, dtype=np.int64)

        assert isinstance(eps[0], (int, np.integer)), f"episode_index type: {type(eps[0])}"

        self.actions = np.asarray(self.base["actions"], dtype=np.float32)
        self.state = np.asarray(self.base["state"], dtype=np.float32)

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

        actions = torch.from_numpy(self.actions[act_ids])
        state0 = torch.from_numpy(self.state[obs_ids[0]]).view(1, -1)

        images = torch.stack(
            [_to_chw_float(self.base[i]["image"], normalize_01=self.normalize_images_01) for i in obs_ids],
            dim=0,
        )
        wrist_images = torch.stack(
            [_to_chw_float(self.base[i]["wrist_image"], normalize_01=self.normalize_images_01) for i in obs_ids],
            dim=0,
        )
        sample = {"actions": actions, "state": state0, "image": images, "wrist_image": wrist_images}
        ti = int(self.task_index_arr[obs_ids[0]])
        sample["task_text"] = self.task_map[ti]
        return sample


def collate_with_task_text(batch):
    task_text = [b["task_text"] for b in batch]
    batch2 = []
    for b in batch:
        b = dict(b)
        b.pop("task_text")
        batch2.append(b)
    out = default_collate(batch2)
    out["task_text"] = task_text
    return out


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print("task_bddl_file:", task_bddl_file)
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_state(obs):
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    assert state.shape == (8,)
    return state



eval_epoch_milestones = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2600, 2800]
LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]