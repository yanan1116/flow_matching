# Train flow matching action expert for VLA robotic manipulation in simulation environment

![pipeline](images/overall.png "overall")


<p align="center">
<img src="images/flow.gif" width="900" height="270"/>
</p>

## Key components
🔬 **This repo contains** \
Training and evaluation examples of using flow matching on Robomimic, PushT, Libero and Franka Kitchen benchmarks. Modified from paper of `Affordance-based Robot Manipulation with Flow Matching`( https://hri-eu.github.io/flow-matching-policy/) with bugs fixed and more features.

🌷 **Getting Started**
🚀 Install the Python dependencies: `pip install -r requirements.txt`

🧪 **Run training and evaluation on benchmarks**
- `pusht`: `examples/flow_pusht.py`
- `franka kitchen`: `examples/flow_kitchen.py`
- `libero`:
  - **👁️ Vision + 🤖 Robot State → 🎯 Action**: `examples/flow_libero_unet_ablation_text.py`
  - **👁️ Vision + 📝 Textual Instruction + 🤖 Robot State → 🎯 Action**: `examples/flow_libero_unet_qwen.py`
  - **👁️ Vision + 📝 Textual Instruction + 🤖 Robot State → 🧠 VLM → 🎯 Action**: `examples/flow_libero_vlm.py`
  - **👁️ Vision + 📝 Textual Instruction + 🤖 Robot State → 🧠 VLM → CoT → 🎯 Action**: `examples/flow_libero_unet_qwen_cot.py`
example :
The training procedure will save checkpoints at some intervals to local folder `checkpoints`

```bash
python examples/flow_libero_unet_qwen.py \
--save_cp 	\
--cp_name unfrozen_text_model
```

Then with the saved checkpoints, do evaluation on libero or libero-plus
```bash
python examples/flow_libero_unet_qwen.py \
--eval_cp checkpoints/libero/unet_qwen/cp-frozen_text_model-100.pth
```


Train with CoT (parameters need to be tuned)
```bash
python examples/flow_libero_unet_qwen_cot.py \
--batchsize 256 \
--save_cp 	\
--frozen_text_model \
--cp_name cot_frozen_text_model  \
--lambda_fm_start  0.1 \
--lambda_fm_end  1.0 \
--lambda_depth_start  1.0 \
--lambda_depth_end  0.1 \
--lambda_eef_start  1.0 \
--lambda_eef_end  0.1 \
--lambda_plateau_min_epochs 100 \
--lambda_ramp_epochs  60 \
--lambda_force_ramp_epoch 240 \
--lambda_force_action_epoch  360

python examples/flow_libero_unet_qwen_cot.py \
--batchsize 128 \
--save_cp 	\
--frozen_text_model \
--cp_name cot_frozen_text_model_cot_first  \
--lambda_fm_start 0.0  \
--lambda_fm_end 1.0  \
--lambda_depth_start 1.0  \
--lambda_depth_end 0.1  \
--lambda_eef_start 1.0  \
--lambda_eef_end 0.1  \
--lambda_plateau_min_epochs 120  \
--lambda_ramp_epochs 80  \
--lambda_force_ramp_epoch 280  \
--lambda_force_action_epoch 420
```





🧱 **VLA Datasets on huggingface 🤗**

on top of the vanilla libero VLA training set (`https://huggingface.co/datasets/physical-intelligence/libero`)
here is the enhanced dataset version with additional two columns: `depth` and `eef_traj`: `https://huggingface.co/datasets/yananchen/libero_cot_contious`

Please note that it is not Lerobot format anymore, it is vanilla huggingface datasets.

Refer to the original MolmoAct dataset: `https://huggingface.co/datasets/allenai/libero` where the CoT is extracted from.



📈 **Performance on vanilla LIBERO**
Without using VLM, just vanilla flow-matching model without fine-tuning of textual encoder, can reach ~85% success rate average across four benchmark suites: 
```
libero_spatial
libero_object
libero_goal
libero_10
```

| Epoch | Success Rate (%) |
|------:|-----------------:|
| 100   | 38.1             |
| 200   | 42.7             |
| 300   | 62.3             |
| 400   | 64.3             |
| 500   | 63.9             |
| 600   | 75.2             |
| 800   | 79.1             |
| 1000  | 78.6             |
| 1200  | 81.3             |
| 1400  | 84.65            |
| 1600  | 84.95            |
| 1800  | 84.5             |
| 2000  | 85.4             |



However, if the vanilla LIBERO is replaced with `LIBERO PLUS`([https://github.com/sylvestf/LIBERO-plus]) then the success rate is very low.

