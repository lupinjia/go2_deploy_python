from typing import Union
import numpy as np
import time
import torch
import threading
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from common.command_helper import create_damping_cmd, create_zero_cmd
from common.rotation_helper import get_gravity_orientation
from common.remote_controller import RemoteController, KeyMap
from config import Config

locker = threading.Lock()

class TSController:
    def __init__(self, config: Config, interface: str) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        print("Loading policy network from:", config.policy_path)
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs_deque = deque(maxlen=config.frame_stack)
        for _ in range(config.frame_stack):
            self.obs_deque.append(np.zeros(config.num_single_obs, dtype=np.float32))
        self.cur_obs = np.zeros(config.num_single_obs, dtype=np.float32) # current obs
        self.euler = np.zeros(3, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])

        # State Machine
        self.state = "zero_torque"  # initial state
        state_transition_total_time = 2.0 # seconds
        self.state_transition_total_steps = int(state_transition_total_time / self.config.control_dt)
        self.transition2sit_step = 0
        self.transition2sit_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.transition2stand_step = 0
        self.transition2stand_init_dof_pos = np.zeros(self.config.num_actions, dtype=np.float32)
        self.control_step_count = 0
        
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.InitLowCmd()
        
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
        self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        
        if interface != "lo":
            # Disable MCF mode to enable custom control
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(5.0)
            self.sport_client.Init()
            
            self.motion_switcher_client = MotionSwitcherClient()
            self.motion_switcher_client.SetTimeout(5.0)
            self.motion_switcher_client.Init()
            
            status, result = self.motion_switcher_client.CheckMode()
            while result['name']:
                self.sport_client.StandDown()
                self.motion_switcher_client.ReleaseMode()
                status, result = self.motion_switcher_client.CheckMode()
                time.sleep(1.0)
            
            print("Release mcf mode")
        
        self.lowCmdThread = RecurrentThread(
            interval=self.config.communication_dt, 
            target=self.LowCmdHandler,
            name="LowCmdThread")
        self.lowCmdThread.Start()
        
        self.mainControlThread = RecurrentThread(
            interval=self.config.control_dt,
            target=self.mainControlStep,
            name="MainControlThread")
        self.mainControlThread.Start()
    
    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        PosStopF = 2.146e9
        VelStopF = 16000.0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        
    def LowCmdHandler(self):
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
    
    def damping_state(self):
        create_damping_cmd(self.low_cmd)

    def zero_torque_state(self):
        create_zero_cmd(self.low_cmd)
    
    def move_to_sit_pos(self):
        dof_idx = self.config.leg_joint2motor_idx
        sit_pos = self.config.sit_angles
        
        # move to sit pos
        alpha = min(self.transition2sit_step / self.state_transition_total_steps, 1)
        for j in range(self.config.num_actions):
            motor_idx = dof_idx[j]
            target_pos = sit_pos[j]
            self.low_cmd.motor_cmd[motor_idx].q = self.transition2sit_init_dof_pos[j] * (1 - alpha) + target_pos * alpha
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stand_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.stand_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        
        self.transition2sit_step += 1

    def move_to_stand_pos(self):
        dof_idx = self.config.leg_joint2motor_idx
        stand_pos = self.config.default_angles
        
        # move to stand pos
        alpha = min(self.transition2stand_step / self.state_transition_total_steps, 1)
        for j in range(self.config.num_actions):
            motor_idx = dof_idx[j]
            target_pos = stand_pos[j]
            self.low_cmd.motor_cmd[motor_idx].q = self.transition2stand_init_dof_pos[j] * (1 - alpha) + target_pos * alpha
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stand_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.stand_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.transition2stand_step += 1
    
    def updateStateMachine(self):
        if self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R1].on_press:
            if self.state == "damping": # damping to sit
                print("Moving to sit pos.")
                # record the current pos
                dof_idx = self.config.leg_joint2motor_idx
                for i in range(self.config.num_actions):
                    self.transition2sit_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                # reset step counter
                self.transition2sit_step = 0
                self.state = "sit"
            else:
                raise NotImplementedError("Cannot transition from damping to states other than sit.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.R2].on_press:
            if self.state == "sit": # sit to stand
                print("Moving to stand pos.")
                # record the current pos
                dof_idx = self.config.leg_joint2motor_idx
                for i in range(self.config.num_actions):
                    self.transition2stand_init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
                # reset step counter
                self.transition2stand_step = 0
                self.state = "stand"
            else:
                raise NotImplementedError("Cannot transition from sit to states other than stand.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.A].on_press:
            if self.state == "stand": # stand to ctrl
                print("Entering control state.")
                self.state = "ctrl"
            else:
                raise NotImplementedError("Cannot transition from stand to states other than ctrl.")
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.Y].on_press:
            print("Enter damping state.")
            self.state = "damping"
            # back to damping from any state
        elif self.remote_controller.button[KeyMap.L1].pressed and self.remote_controller.button[KeyMap.X].on_press:
            if self.state == "damping":
                print("Enter zero torque state.")
                self.state = "zero_torque"
            else:
                raise NotImplementedError("Can only enter zero torque state from damping state.")
        else:
            pass

    def mainControlStep(self):
        if self.control_step_count == 50:
            print(f"Current State: {self.state}")
            self.control_step_count = 0
        # Update remote controller state
        self.remote_controller.set(self.low_state.wireless_remote)
        # Update State Machine
        self.updateStateMachine()
        locker.acquire()
        if self.state == "zero_torque":
            self.zero_torque_state()
        elif self.state == "damping":
            self.damping_state()
        elif self.state == "sit":
            self.move_to_sit_pos()
        elif self.state == "stand":
            self.move_to_stand_pos()
        elif self.state == "ctrl":
            self.calculate()
        else:
            raise ValueError("Invalid state.")
        locker.release()
        
        self.control_step_count += 1
        
    def calculate(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        self.euler = self.low_state.imu_state.rpy

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.cur_obs[:3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.cur_obs[3:6] = gravity_orientation
        self.cur_obs[6:9] = ang_vel
        self.cur_obs[9 : 9 + num_actions] = qj_obs
        self.cur_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.cur_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        self.obs_deque.append(self.cur_obs.copy())
        self.obs_history = np.concatenate([self.obs_deque[i] for i in range(len(self.obs_deque))], 
                                  axis=0)

        # Get the action from the policy network
        cur_obs_tensor = torch.from_numpy(self.cur_obs).unsqueeze(0)
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        self.action = self.policy(cur_obs_tensor, obs_history_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.ctrl_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.ctrl_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0
    
# Explicit Estimator Controller
class EEController(TSController):
    # ovveride calculate function
    def calculate(self):
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        self.euler = self.low_state.imu_state.rpy

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.cur_obs[:3] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.cur_obs[3:6] = gravity_orientation
        self.cur_obs[6:9] = ang_vel
        self.cur_obs[9 : 9 + num_actions] = qj_obs
        self.cur_obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.cur_obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        
        self.obs_deque.append(self.cur_obs.copy())
        self.obs_history = np.concatenate([self.obs_deque[i] for i in range(len(self.obs_deque))], 
                                  axis=0)

        # Get the action from the policy network
        obs_history_tensor = torch.from_numpy(self.obs_history).unsqueeze(0)
        self.action = self.policy(obs_history_tensor).detach().numpy().squeeze()
        
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.ctrl_kp
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.ctrl_kd
            self.low_cmd.motor_cmd[motor_idx].tau = 0

class WaQController(TSController):
    pass