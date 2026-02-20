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
  - **vision + robot state → action**: `examples/flow_libero_unet.py`
  - **vision + textual instruction + robot state → concat → action**: `examples/flow_libero_unet_qwen.py`
  - **vision + textual instruction + robot state → VLM → action**: `examples/flow_libero_vlm.py`


