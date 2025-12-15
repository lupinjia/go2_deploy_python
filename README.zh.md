此项目为[宇树提供的原本的python部署代码](https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_real)的修改版.

## 安装依赖

- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [pytorch](https://pytorch.org/)
- scipy
- pyyaml

## 使用

- TS Controller (sim)
    
  ```bash
  python deploy.py --config=ts.yaml --type=ts
  ```
- TS Controller (real)
  
  ```bash
  python deploy.py --config=ts.yaml --type=ts --interface=your_ethernet
  ```
  interface参数应当为你的电脑的以太网卡名称. 该名称可以通过`ifconfig`命令查询.
- EE Controller (sim)

  ```bash
  python deploy.py --config=ee.yaml --type=ee
  ```


## 支持的控制器

| 控制器类型 | GIF | 训练代码 |
|--- | --- | --- |
|  Teacher-Student  | ![](https://github.com/lupinjia/demo_imgs/blob/master/ts_demo.gif) | [genesis_lr/go2_ts](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ts) |
| Explicit Estimator |  | [genesis_lr/go2_ee](https://github.com/lupinjia/genesis_lr/tree/main/legged_gym/envs/go2/go2_ee) |