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
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from datasets import Image, load_dataset


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
    inst = normalize_instruction(instruction)
    img_h = hash_bytes(image_like_to_bytes(image_v))
    wrs_h = hash_bytes(image_like_to_bytes(wrist_v))
    return f"{inst}|{img_h}|{wrs_h}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_repo", default="physical-intelligence/libero")
    parser.add_argument("--molmo_repo", default="yananchen/molmoact_libero_cot")
    parser.add_argument("--split", default="train")

    parser.add_argument("--instruction_col", default="instruction")
    parser.add_argument("--libero_image_col", default="image")
    parser.add_argument("--libero_wrist_col", default="wrist_image")
    parser.add_argument("--molmo_image_col", default="image")
    parser.add_argument("--molmo_wrist_col", default="wrist")
    parser.add_argument("--depth_col", default="depth")
    parser.add_argument("--eef_traj_col", default="eef_traj")

    parser.add_argument(
        "--output_parquet",
        default="",
        help="Optional path to write ds_libero with joined columns.",
    )
    args = parser.parse_args()

    print("Loading datasets...")
    ds_libero = load_dataset(args.libero_repo, split=args.split)
    ds_molmo = load_dataset(args.molmo_repo, split=args.split)
    print(f"ds_libero: {len(ds_libero)} rows")
    print(f"ds_molmoact: {len(ds_molmo)} rows")

    ds_libero = ensure_decode_false(ds_libero, args.libero_image_col)
    ds_libero = ensure_decode_false(ds_libero, args.libero_wrist_col)
    ds_molmo = ensure_decode_false(ds_molmo, args.molmo_image_col)
    ds_molmo = ensure_decode_false(ds_molmo, args.molmo_wrist_col)
    ds_molmo = ds_molmo.map(
                lambda ex: {"instruction": ex["instruction"].rstrip(".")},
                    num_proc=16,
                )


    print("Building exact index from ds_molmoact...")
    molmo_index, ambiguous_keys = build_molmo_index(
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
    print(f"molmo_ambiguous_keys: {ambiguous_keys}")

    print("Joining onto ds_libero...")
    matched = 0
    unmatched = 0
    ambiguous_hit = 0
    joined_depth = []
    joined_eef = []

    for ex in ds_libero:
        key = make_key(ex[args.instruction_col], ex[args.libero_image_col], ex[args.libero_wrist_col])
        payload = molmo_index.get(key, "MISS")
        if payload == "MISS":
            unmatched += 1
            joined_depth.append(None)
            joined_eef.append(None)
        elif payload is None:
            ambiguous_hit += 1
            joined_depth.append(None)
            joined_eef.append(None)
        else:
            matched += 1
            joined_depth.append(payload.depth)
            joined_eef.append(payload.eef_traj)

    total = len(ds_libero)
    hit_rate = (matched / total) if total else 0.0
    miss_rate = (unmatched / total) if total else 0.0
    ambiguous_rate = (ambiguous_hit / total) if total else 0.0

    print("\n=== Stage-1 Exact Join Stats ===")
    print(f"total_libero: {total}")
    print(f"matched: {matched}")
    print(f"unmatched: {unmatched}")
    print(f"ambiguous_hit: {ambiguous_hit}")
    print(f"hit_rate: {hit_rate:.6f}")
    print(f"miss_rate: {miss_rate:.6f}")
    print(f"ambiguous_rate: {ambiguous_rate:.6f}")

    if args.output_parquet:
        out_dir = os.path.dirname(args.output_parquet)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        ds_out = ds_libero.add_column("joined_depth", joined_depth)
        ds_out = ds_out.add_column("joined_eef_traj", joined_eef)
        ds_out.to_parquet(args.output_parquet)
        print(f"Wrote joined dataset parquet: {args.output_parquet}")


if __name__ == "__main__":
    main()
