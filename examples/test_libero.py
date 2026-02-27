import argparse
import re
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

MARKER = "the action that the robot should take is"


def extract_action_chunk(example):
    conv = example.get("conversations", {})
    from_list = conv.get("from", []) or []
    value_list = conv.get("value", []) or []

    gpt_text = ""
    for role, text in zip(from_list, value_list):
        if isinstance(role, str) and role.lower() == "gpt" and isinstance(text, str):
            gpt_text = text
            break
    assert gpt_text, "Missing GPT response in conversations"
    assert MARKER in gpt_text.lower(), f"Missing marker in GPT response: {MARKER}"

    match = re.search(
        r"the action that the robot should take is\s*(.*)$",
        gpt_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    action_chunk = match.group(1).strip() if match else ""

    assert value_list and isinstance(value_list[0], str), "Missing user prompt in conversations['value'][0]"
    user_prompt = value_list[0]
    ask_marker = "what is the action that the robot should take"
    assert ask_marker in user_prompt.lower(), f"Missing marker in user prompt: {ask_marker}"
    m_inst = re.search(
        r"the task is\s*(.*?)\.\s*what is the action that the robot should take",
        user_prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    instruction = " ".join(m_inst.group(1).strip().split()) if m_inst else ""

    return {"action_chunk": action_chunk, "instruction": instruction}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default='/mnt/disk1t/molmoact_libero')
    parser.add_argument("--output_dir", type=str, default="/mnt/disk1t/molmoact_libero_map")
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
            cache_dir=args.cache_dir,
        )
        if args.shuffle:
            ds_part = ds_part.shuffle()
        if args.subsample_per_subset > 0:
            ds_part = ds_part.select(range(args.subsample_per_subset))
        
        parts.append(ds_part)
    base_ds = concatenate_datasets(parts)

    base_ds = base_ds.filter(
        lambda ex: ex.get("annotation") is not None,
        desc="Dropping rows with null annotation",
    )
    base_ds = base_ds.map(extract_action_chunk, desc="Adding action_chunk and instruction")
    

    print("base_ds info:", base_ds, "\n", base_ds.features)
    print("action_chunk dtype:", base_ds.features["action_chunk"])
    print("instruction dtype:", base_ds.features["instruction"])
    print("example action_chunk:", base_ds[0]["action_chunk"])
    print("example instruction:", base_ds[0]["instruction"])
    print("saved to:", args.output_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    base_ds.save_to_disk(args.output_dir) 
    
    
    
    base_ds.push_to_hub( 'yananchen/molmoact_libero_map_sample',
        tags=["libero", "panda", "franka"],
        private=False,
        push_videos=True,
        license="apache-2.0",
    )
    
    
if __name__ == "__main__":
    main()
