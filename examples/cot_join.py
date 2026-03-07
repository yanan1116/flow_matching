#!/usr/bin/env python3
"""
Stage-1 exact join between ds_molmoact and ds_libero.

Join key:
  normalized_instruction + hash(image bytes) + hash(wrist bytes)

Default dataset repos:
  - libero: physical-intelligence/libero
  - molmoact: yananchen/molmoact_libero_cot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from datasets import DownloadMode, Image, config as ds_config, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm


WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class JoinedPayload:
    depth: str
    eef_traj: str


def normalize_instruction(text: str) -> str:
    return WS_RE.sub(" ", text.strip().lower())


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def image_like_to_bytes(v) -> bytes:
    # datasets Image(decode=False): {"bytes": ..., "path": ...}
    if isinstance(v, dict):
        b = v.get("bytes", None)
        if b is not None:
            return b
        p = v.get("path", None)
        if p:
            return _read_file_bytes(p)
        raise ValueError("Image dict has neither bytes nor path.")

    # PIL image or np array fallback
    arr = np.asarray(v)
    if arr.size == 0:
        raise ValueError("Empty image array.")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def make_key(instruction: str, image_v, wrist_v) -> str:
    # inst = normalize_instruction(instruction)
    img_h = hash_bytes(image_like_to_bytes(image_v))
    wrs_h = hash_bytes(image_like_to_bytes(wrist_v))
    return f"{instruction}|{img_h}|{wrs_h}"


def ensure_decode_false(ds, col_name: str):
    feat = ds.features[col_name]
    if isinstance(feat, Image) and getattr(feat, "decode", True):
        ds = ds.cast_column(col_name, Image(decode=False))
    return ds




def build_molmo_index(
    ds_molmoact,
    instruction_col: str,
    image_col: str,
    wrist_col: str,
    depth_col: str,
    eef_traj_col: str,
) -> Tuple[Dict[str, Optional[JoinedPayload]], int]:
    index: Dict[str, Optional[JoinedPayload]] = {}
    ambiguous = 0

    for ex in ds_molmoact:
        key = make_key(ex[instruction_col], ex[image_col], ex[wrist_col])
        payload = JoinedPayload(depth=ex[depth_col], eef_traj=ex[eef_traj_col])

        if key not in index:
            index[key] = payload
            continue

        prev = index[key]
        if prev is None:
            continue

        if prev != payload:
            index[key] = None
            ambiguous += 1

    return index, ambiguous

meta = load_dataset(
    'physical-intelligence/libero',
    data_files="meta/tasks.jsonl",
    split="train",
)
task_map = {row["task_index"]: row["task"] for row in meta}

def add_instruction(example):
    instruction = task_map[example['task_index']]
    return {'instruction': instruction}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_repo", default="physical-intelligence/libero")
    parser.add_argument("--molmo_repo", default="yananchen/molmoact_libero_cot")
    parser.add_argument("--instruction_col", default="instruction")
    parser.add_argument("--libero_image_col", default="image")
    parser.add_argument("--libero_wrist_col", default="wrist_image")
    parser.add_argument("--molmo_image_col", default="image")
    parser.add_argument("--molmo_wrist_col", default="wrist")
    parser.add_argument("--depth_col", default="depth")
    parser.add_argument("--eef_traj_col", default="eef_traj")
    args = parser.parse_args()

    print("Loading datasets...")
    ds_libero = load_dataset(
        path=args.libero_repo,
        split='train',
    )
    # ds_libero = ds_libero.shuffle().select(range(10000))
    ds_libero = ds_libero.map(add_instruction, num_proc=16, desc='add instruction')
    ds_libero = ensure_decode_false(ds_libero, args.libero_image_col)
    ds_libero = ensure_decode_false(ds_libero, args.libero_wrist_col)

    ds_libero_task_index = set(list(ds_libero['task_index']))

    print(f"ds_libero: {len(ds_libero)} rows  distinct task index: {len(ds_libero_task_index)} ")

    if not Path("./molmo_index.pkl").is_file():
        ds_molmo = load_dataset(
            path=args.molmo_repo,
            split='train',
        )
        ds_molmo = ensure_decode_false(ds_molmo, args.molmo_image_col)
        ds_molmo = ensure_decode_false(ds_molmo, args.molmo_wrist_col)
        print(f"ds_molmoact: {len(ds_molmo)} rows")
        print("Building exact index from ds_molmoact...")
        molmo_index, ambiguous_keys_cnt = build_molmo_index(
            ds_molmo,
            instruction_col=args.instruction_col,
            image_col=args.molmo_image_col,
            wrist_col=args.molmo_wrist_col,
            depth_col=args.depth_col,
            eef_traj_col=args.eef_traj_col,
        )
        unique_keys = sum(v is not None for v in molmo_index.values())
        print(f"molmo_index keys: {len(molmo_index)}")
        print(f"molmo_unique_keys: {unique_keys}")
        print(f"molmo_ambiguous_keys: {ambiguous_keys_cnt}")
        assert ambiguous_keys_cnt == 0
        with open('./molmo_index.pkl', "wb") as f:
            pickle.dump(molmo_index, f, protocol=pickle.HIGHEST_PROTOCOL, )
    else:
        with open("./molmo_index.pkl", "rb") as f:
            molmo_index = pickle.load(f)

    print("Joining onto ds_libero...")
    unmatched = 0
    joined_depth = []
    joined_eef = []

    missed_task_episode_index = set()
    for ex in tqdm(ds_libero, total=len(ds_libero), desc="Joining ds_libero"):

        key = make_key(ex[args.instruction_col], ex[args.libero_image_col], ex[args.libero_wrist_col])
        payload = molmo_index.get(key, None)
        if not payload:
            unmatched += 1
            missed_task_episode_index.add(f"{ex['task_index']}-{ex['episode_index']}" )
            joined_depth.append('')
            joined_eef.append('')            
        else:
            joined_depth.append(payload.depth)
            joined_eef.append(payload.eef_traj)

    print(f"unmatched: {unmatched}", unmatched / len(ds_libero))
    print( "missed_task_episode_index:", len(missed_task_episode_index) )

    assert len(joined_depth) == len(ds_libero)
    assert len(joined_eef) == len(ds_libero)

    ds_libero = ds_libero.add_column("depth", joined_depth)
    ds_libero = ds_libero.add_column("eef_traj", joined_eef)
    ds_libero_filter = ds_libero.filter(lambda ex: f"{ex['task_index']}-{ex['episode_index']}" not in missed_task_episode_index)
    # ds_libero_filter = ds_libero.filter(lambda ex: ex['depth'] != '' and ex['eef_traj'] != '')
    print('ds_libero_filter:', ds_libero_filter.num_rows / ds_libero.num_rows)
    assert ds_libero_filter.filter(lambda ex: ex['depth'] == '' or  ex['eef_traj'] == '').num_rows == 0, " ds_libero_filter should have no blank cot cols"
    # ds_libero_sample = ds_libero_filter.select(range(10000))

    ds_libero_filter.push_to_hub(
        f"yananchen/libero_cot_contious",
        private=False,
        embed_external_files=True,
    )

if __name__ == "__main__":
    main()
