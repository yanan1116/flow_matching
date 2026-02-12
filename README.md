# Train flow matching action expert for VLA robotic manipulation in simulation environment

![pipeline](images/overall.png "overall")


<p align="center">
<img src="images/flow.gif" width="900" height="270"/>
</p>

## Key components
🔬 **This repo contains** \
Training and evaluation examples of using flow matching on Robomimic, PushT and Franka Kitchen benchmarks. Modified from paper of `Affordance-based Robot Manipulation with Flow Matching`( https://hri-eu.github.io/flow-matching-policy/) with bugs fixed.

🌷 **Getting Started**
- Install the Python dependencies: `pip install -r requirements.txt`

- run train and evaluation on benchmark of 
  * `pusht`: `CUDA_VISIBLE_DEVICES=0 python examples/flow_pusht.py  --net ConditionalUnet1D`
  * `franka kitchen`: `CUDA_VISIBLE_DEVICES=0 python examples/flow_kitchen.py`

