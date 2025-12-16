This project is an modification of [original python deployment code provided by unitree](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_real)

## Installation

- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [pytorch](https://pytorch.org/)
- scipy
- pyyaml

## Usage

- TS Controller (sim)
    
  ```bash
  python deploy.py --config=ts.yaml --type=ts
  ```
- TS Controller (real)
  
  ```bash
  python deploy.py --config=ts.yaml --type=ts --interface=your_ethernet
  ```
  The interface should be the name of your ethernet card. It can be seen by running `ifconfig` in the terminal.
- EE Controller (sim)

  ```bash
  python deploy.py --config=ee.yaml --type=ee
  ```

## Demo

| Controller Type | GIF | Training Code |
|--- | --- | --- |
|  Teacher-Student  |  ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ts_demo.gif)   |   [genesis_lr/go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts)  |
| Explicit Estimator | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/ee_demo.gif) | [genesis_lr/go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) |
| DreamWaQ | ![](https://raw.githubusercontent.com/lupinjia/demo_imgs/refs/heads/master/dreamwaq_demo.gif) | [genesis_lr/go2_dreamwaq](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_dreamwaq) |
