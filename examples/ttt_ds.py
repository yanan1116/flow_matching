import argparse
import re,os
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, DownloadConfig

ds = load_dataset(
            "yananchen/molmoact_libero_cot",
            split='train', 
            download_config=DownloadConfig(local_files_only=True)
            #cache_dir=args.cache_dir,
        )

target_instruction = "pick up the book and place it in the back compartment of the caddy."
matched_ds = ds.filter(lambda ex: ex.get("instruction") == target_instruction)
print(f'matched instruction count: {len(matched_ds)}')
