# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import signal
import sys
import threading
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd


def _obs_allegro2hora(obses):
    obs_index = obses[0:4]
    obs_middle = obses[4:8]
    obs_ring = obses[8:12]
    obs_thumb = obses[12:16]
    obses = np.concatenate([obs_index, obs_thumb, obs_middle, obs_ring]).astype(
        np.float32
    )
    return obses


def _action_hora2allegro(actions):
    cmd_act = actions.copy()
    cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
    cmd_act[[12, 13, 14, 15]] = actions[[4, 5, 6, 7]]
    cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
    return cmd_act


class HoraPoseSubscriber(Node):
    def __init__(self, topic):
        super().__init__("hora_pose_subscriber")
        self.topic = topic

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.hand_T_object = None
        # 20Hz inference timer
        self.timer = self.create_timer(0.05, self.on_timer)

    def on_timer(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "hand_base_link", self.topic, rclpy.time.Time()
            )
            obj_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
            obj_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
            obj_quat = np.array(obj_quat, dtype=np.float32)
            obj_pos = np.array(obj_pos, dtype=np.float32)
            # Process the pose data as needed
            self.hand_T_object = np.concatenate([obj_pos, obj_quat]).astype(np.float32)

        except Exception as e:
            self.get_logger().error(f"Failed to get transform: {e}")

    def get_pose(self):
        if self.hand_T_object is not None:
            return self.hand_T_object
        else:
            self.get_logger().warning("Object pose not available yet.")
            return None


class HardwarePlayer(object):
    def __init__(self, config):
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cuda"

        obs_shape = (96,)
        net_config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "actor_units": [512, 256, 128],
            "priv_mlp_units": [256, 128, 8],
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((30, 32)).to(self.device)
        self.sa_mean_std.eval()

        self.pose_topic = "object_marker"

        self.noisy_obj_pose_mean_std = RunningMeanStd((30, 7)).to(self.device)
        self.noisy_obj_pose_mean_std.eval()
        # hand setting
        self.init_pose = [
            0.0627,
            1.2923,
            0.3383,
            0.1088,
            0.0724,
            1.1983,
            0.1551,
            0.1499,
            0.1343,
            1.1736,
            0.5355,
            0.2164,
            1.1202,
            1.1374,
            0.8535,
            -0.0852,
        ]
        self.allegro_dof_lower = torch.from_numpy(
            np.array(
                [
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                    0.2630,
                    -0.1050,
                    -0.1890,
                    -0.1620,
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                    -0.4700,
                    -0.1960,
                    -0.1740,
                    -0.2270,
                ]
            )
        ).to(self.device)
        self.allegro_dof_upper = torch.from_numpy(
            np.array(
                [
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                    1.3960,
                    1.1630,
                    1.6440,
                    1.7190,
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                    0.4700,
                    1.6100,
                    1.7090,
                    1.6180,
                ]
            )
        ).to(self.device)

    def deploy(self, keyboard_interactive=False, pose_topic="object_marker"):
        import rclpy
        from gum.devices.metahand.allegro_client import AllegroRobot

        self.pose_topic = pose_topic

        print(f"Pose topic: {self.pose_topic}")

        rclpy.init()
        executor = rclpy.executors.SingleThreadedExecutor()
        allegro = AllegroRobot()

        def disconnect(signum, frame):
            print("ctrl-C received")
            allegro.disconnect()
            sys.exit(-1)

        signal.signal(signal.SIGINT, disconnect)

        executor.add_node(allegro)
        pose_subscriber = None
        if self.pose_topic is not None:
            pose_subscriber = HoraPoseSubscriber(self.pose_topic)
            executor.add_node(pose_subscriber)

        threading.Thread(target=executor.spin, daemon=True).start()

        hz = 20
        ros_rate = allegro.create_rate(hz)

        if keyboard_interactive:
            input("press to move to initial position")

        obses, _ = allegro.poll_joint_position(wait=True)
        step = hz * 4
        for i in range(step):
            target_joints = i / step * np.array(self.init_pose) + (
                1 - i / step
            ) * np.array(obses)
            allegro.command_joint_position(target_joints)
            ros_rate.sleep()

        if keyboard_interactive:
            input("press to deploy policy")

        obses, _ = allegro.poll_joint_position(wait=True)
        obses = _obs_allegro2hora(obses)
        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 16 * 3 * 2)).astype(np.float32)).cuda()
        proprio_hist_buf = torch.from_numpy(
            np.zeros((1, 30, 16 * 2)).astype(np.float32)
        ).cuda()
        noisy_obj_pose_buf = torch.from_numpy(
            np.zeros((1, 30, 7)).astype(np.float32)
        ).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.allegro_dof_lower, self.allegro_dof_upper)[
            None
        ]

        for i in range(3):
            obs_buf[:, i * 16 + 0 : i * 16 + 16] = cur_obs_buf.clone()  # joint position
            obs_buf[:, i * 16 + 16 : i * 16 + 32] = (
                prev_target.clone()
            )  # current target (obs_t-1 + s * act_t-1)

        if self.pose_topic is not None and pose_subscriber is not None:
            while pose_subscriber.get_pose() is None:
                print("Waiting for object pose...")
                time.sleep(0.1)
            noisy_obj_pose_buf[:, :, :7] = torch.from_numpy(pose_subscriber.get_pose())[
                None, None
            ].cuda()

        proprio_hist_buf[:, :, :16] = cur_obs_buf.clone()
        proprio_hist_buf[:, :, 16:32] = prev_target.clone()

        while True:
            obs = self.running_mean_std(obs_buf.clone())
            input_dict = {
                "obs": obs,
                "proprio_hist": self.sa_mean_std(proprio_hist_buf.clone()),
            }
            if self.pose_topic is not None:
                input_dict["object_pose_hist"] = self.noisy_obj_pose_mean_std(
                    noisy_obj_pose_buf.clone()
                )

            action = self.model.act_inference(input_dict)
            action = torch.clamp(action, -1.0, 1.0)

            target = prev_target + self.action_scale * action
            target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
            prev_target = target.clone()
            # interact with the hardware
            commands = target.cpu().numpy()[0]
            commands = _action_hora2allegro(commands)
            allegro.command_joint_position(commands)
            ros_rate.sleep()  # keep 20 Hz command
            # get o_{t+1}
            obses, torques = allegro.poll_joint_position(wait=True)
            obses = _obs_allegro2hora(obses)
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            cur_obs_buf = unscale(
                obses, self.allegro_dof_lower, self.allegro_dof_upper
            )[None]
            prev_obs_buf = obs_buf[:, 32:].clone()
            obs_buf[:, :64] = prev_obs_buf
            obs_buf[:, 64:80] = cur_obs_buf.clone()
            obs_buf[:, 80:96] = target.clone()

            priv_proprio_buf = proprio_hist_buf[:, 1:30, :].clone()
            cur_proprio_buf = torch.cat([cur_obs_buf, target.clone()], dim=-1)[:, None]
            proprio_hist_buf[:] = torch.cat([priv_proprio_buf, cur_proprio_buf], dim=1)

            if self.pose_topic is not None:
                prev_noisy_obj_pose_buf = noisy_obj_pose_buf[:, 1:30, :].clone()
                curr_noisy_obj_pose = torch.from_numpy(pose_subscriber.get_pose())[
                    None, None
                ].cuda()
                print(f"current noisy obj pose: {curr_noisy_obj_pose}")
                noisy_obj_pose_buf[:] = torch.cat(
                    [prev_noisy_obj_pose_buf, curr_noisy_obj_pose], dim=1
                )

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        self.model.load_state_dict(checkpoint["model"])
        self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        self.noisy_obj_pose_mean_std.load_state_dict(checkpoint["obj_pose_mean_std"])
