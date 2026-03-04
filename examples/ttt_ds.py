import argparse
import re,os
import shutil
from pathlib import Path
from collections import Counter

from datasets import concatenate_datasets, load_dataset, DownloadConfig





meta = load_dataset(
    'physical-intelligence/libero',
    data_files="meta/tasks.jsonl",
    split="train",
)
task_map = {row["task_index"]: row["task"] for row in meta}
tasks_desc_set = set()
for ii, task_desc in task_map.items():
    # print(f'{ii}===>{task_desc}')
    tasks_desc_set.add(task_desc)
# 

def add_instruction(example):
    instruction = task_map[example['task_index']]
    return {'instruction': instruction}

ds_libero = load_dataset(
    'physical-intelligence/libero',
    split="train",
)
ds_libero = ds_libero.map(add_instruction, num_proc=8, desc='add instruction')
print(ds_libero.num_rows)
print(ds_libero.features)

os._exit(0)



ds_molmoact = load_dataset(
            "yananchen/molmoact_libero_cot",
            split='train', 
            download_config=DownloadConfig(local_files_only=True)
            #cache_dir=args.cache_dir,
        )
num_proc = max(1, os.cpu_count() or 1)
ds_molmoact = ds_molmoact.map(
    lambda ex: {"instruction": ex["instruction"].rstrip(".")},
    num_proc=num_proc,
)

instruction_list = ds_molmoact["instruction"]
counts = Counter(instruction_list)
instructions = set(counts)
missing = instructions - tasks_desc_set
assert len(missing) == 0 , f"Found instructions not in tasks_desc_set: {sorted(missing)}"

for task in sorted(instructions):
    print(f'{task}===> {counts[task]}')

'''
open the middle drawer of the cabinet===> 5564
open the top drawer and put the bowl inside===> 6973
pick up the alphabet soup and place it in the basket===> 6400
pick up the bbq sauce and place it in the basket===> 6434
pick up the black bowl between the plate and the ramekin and place it on the plate===> 4408
pick up the black bowl from table center and place it on the plate===> 4759
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate===> 6114
pick up the black bowl next to the cookie box and place it on the plate===> 5775
pick up the black bowl next to the plate and place it on the plate===> 5238
pick up the black bowl next to the ramekin and place it on the plate===> 5682
pick up the black bowl on the cookie box and place it on the plate===> 4287
pick up the black bowl on the ramekin and place it on the plate===> 4021
pick up the black bowl on the stove and place it on the plate===> 4471
pick up the black bowl on the wooden cabinet and place it on the plate===> 5573
pick up the book and place it in the back compartment of the caddy===> 6679
pick up the butter and place it in the basket===> 6938
pick up the chocolate pudding and place it in the basket===> 7658
pick up the cream cheese and place it in the basket===> 5981
pick up the ketchup and place it in the basket===> 6457
pick up the milk and place it in the basket===> 6158
pick up the orange juice and place it in the basket===> 6291
pick up the salad dressing and place it in the basket===> 5957
pick up the tomato sauce and place it in the basket===> 5396
push the plate to the front of the stove===> 4990
put both moka pots on the stove===> 11350
put both the alphabet soup and the cream cheese box in the basket===> 11182
put both the alphabet soup and the tomato sauce in the basket===> 9296
put both the cream cheese box and the butter in the basket===> 11948
put the black bowl in the bottom drawer of the cabinet and close it===> 7842
put the bowl on the plate===> 4097
put the bowl on the stove===> 4472
put the bowl on top of the cabinet===> 4475
put the cream cheese in the bowl===> 3963
put the white mug on the left plate and put the yellow and white mug on the right plate===> 9313
put the white mug on the plate and put the chocolate pudding to the right of the plate===> 8813
put the wine bottle on the rack===> 5869
put the wine bottle on top of the cabinet===> 5004
put the yellow and white mug in the microwave and close it===> 9723
turn on the stove===> 4187
turn on the stove and put the moka pot on it===> 10565
'''