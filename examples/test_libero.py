from datasets import load_dataset
libero = load_dataset("Sylvest/libero_plus_lerobot", split="train[:1000]")



print('base_ds info:', base_ds, '\n', base_ds.features)
print(type(base_ds[0]["observation.images.front"]))

# 示例读取视频数组
video_frames = libero["train"]["observation.images.front"][0]
print(video_frames)