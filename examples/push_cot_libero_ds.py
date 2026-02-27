import argparse
import re
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

def extract_cot(example):
    assert len(example['conversations']['value']) == 2

    instruction_full = example['conversations']['value'][0]
    assert 'The task is' in instruction_full  
    assert  'What is the action that the robot should take' in instruction_full

    cot_full = example['conversations']['value'][1].split('.')[:-1]
    
    assert len(cot_full) == 3, f'cot_full===>{example['conversations']['value'][1]}'
    depth, eef_traj, action_chunk = cot_full[0], cot_full[1], cot_full[2]

    instruction = instruction_full.split('What is the action that the robot should take')[0].strip()
    
    return {"action_chunk": action_chunk, 
            "instruction": instruction.replace('The task is','').strip(), 
            'eef_traj': eef_traj, 
            'depth':depth }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default='')
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--shuffle", action='store_true')
    parser.add_argument("--subsample_per_subset", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    parts = []
    for subset_name in subsets:
        ds_part = load_dataset(
            "allenai/libero",
            subset_name,
            split='train',
            #cache_dir=args.cache_dir,
        )
        if args.shuffle:
            ds_part = ds_part.shuffle()
        if args.subsample_per_subset > 0:
            ds_part = ds_part.select(range(args.subsample_per_subset))
        
        parts.append(ds_part)
    base_ds = concatenate_datasets(parts)

    base_ds = base_ds.filter(
        lambda ex: ex.get("annotation") is None,
        desc="Dropping rows with null annotation",
    )
    base_ds = base_ds.map(extract_cot, 
                        remove_columns=["annotation", 'conversations'],
                        desc="Adding action_chunk and instruction")


    print("base_ds info:", base_ds, "\n", base_ds.features)
    print("action_chunk dtype:", base_ds.features["action_chunk"])
    print("instruction dtype:", base_ds.features["instruction"])
    print("example action_chunk:", base_ds[0]["action_chunk"])
    print("example instruction:", base_ds[0]["instruction"])
    
    # print("saved to:", args.output_dir)
    # out_dir = Path(args.output_dir)
    # if out_dir.exists():
    #     shutil.rmtree(out_dir)
    # base_ds.save_to_disk(args.output_dir) 
    
    
    
    # `datasets==4.1.1` doesn't support `tags`/`push_videos`/`license` kwargs here.
    # Add tags/license in the dataset card (README) on the Hub if needed.
    base_ds.push_to_hub(
        f"yananchen/{args.repo_id}",
        private=False,
        embed_external_files=True,
    )
    
    
if __name__ == "__main__":
    main()
