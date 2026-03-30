# Train flow matching action expert for VLA robotic manipulation in simulation environment

## Key components
🔬 **This repo contains** \
Training and evaluation examples of using flow matching on Robomimic, PushT, Libero and Franka Kitchen benchmarks. Modified from paper of `Affordance-based Robot Manipulation with Flow Matching`( https://hri-eu.github.io/flow-matching-policy/) with bugs fixed, more features and more comprehensive experiments.

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


Kick off training:
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
Here just the `eval_cp` controls if it goes to training mode or evaluation mode.



---

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



⚙️ **Benchmark: LIBERO**

In the experiments, `LIBERO` is used as benchmark to evaluate the success rate on the rollouts.  Each time, an LIBERO environment is initialized with some randomness seed to make it a bit various. Then the model checkpoint is loaded and return action (7-dimension) given state observations. Finally, either the trajectory is judged as success or break the max iteration limit which is flagged as failure (model does not finish the job given limited steps). 

To fully reconstruct the anaconda environment, refer to file `environment.yml`.


📈 **Findings**

Without using VLM, just vanilla flow-matching model without pre-training or fine-tuning, equipped with an off-the-shelf text encoder LLM ( for example, `Qwen/Qwen3-0.6B` ), can easily reach ~85% success rate average across four benchmark suites: `libero_spatial libero_object libero_goal libero_10` from original LIBERO benchmark (`https://libero-project.github.io/datasets`) which is widely tested in current VLAs papers.


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


To systematically expose the hidden vulnerabilities of VLA models through comprehensive robustness evaluation across seven perturbation dimensions, we can also test the same model checkpoint on this enhanced benchmark  `LIBERO PLUS`("In-depth Robustness Analysis for Vision-Language-Action Models" [https://github.com/sylvestf/LIBERO-plus]).

In vanilla LIBERO, before evaluation, we can set `export PYTHONPATH=$PYTHONPATH:~/robotics/LIBERO; ln -sf ~/.libero/config_libero.yaml ~/.libero/config.yaml`
to use LIBERO-PLUS, just install it from source and set `export PYTHONPATH=$PYTHONPATH:~/robotics/LIBERO-plus; ln -sf ~/.libero/config_libero_plus.yaml ~/.libero/config.yaml`

However, This is a more challenging benchmark that introduces more tasks spanning multiple dimensions: Objects Layout, Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise. 

(results are calculating, coming soon)
