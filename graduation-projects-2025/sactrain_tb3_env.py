# ==========================================================
#  TurtleBot3Env  (Supervised 없이 완전 안정화)
#  - 시작 pose 오프셋 적용
#  - 충돌 안정 보정
#  - lidar 안정화
#  - reward 완전 재설계 (성공률 90~98% 목표)
# ==========================================================

import numpy as np
import gymnasium as gym
from controller import Robot


def safe(x, default=0.0):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except:
        return default


class TurtleBot3Env(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot=None,
        collision_threshold=0.25,
        success_threshold=0.30,
        max_steps=400
    ):
        super().__init__()

        self.robot = robot if robot is not None else Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # LIDAR
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_max_range = float(self.lidar.getMaxRange())
        self.lidar_min_range = 0.05

        # Motors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        # Dead-reckoning pose
        self.x = 0
        self.y = 0
        self.theta = 0

        self.last_linear = 0
        self.last_angular = 0
        self.path_length = 0
        self.prev_lidar_min = self.lidar_max_range

        # 3개 goal
        self.goal_list = [
            np.array([0.0, 5.0], dtype=np.float32),
            np.array([-0.5, 0.05], dtype=np.float32),
            np.array([-0.6, -0.6], dtype=np.float32),
        ]
        self.current_goal = None
        self.current_goal_index = None
        self.max_goal_dist = 6.0

        # threshold
        self.collision_threshold = collision_threshold
        self.success_threshold = success_threshold
        self.max_episode_steps = max_steps
        self.step_count = 0
        self.initial_ignore_steps = 5

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32
        )

        obs_dim = self.lidar_resolution + 2
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(obs_dim,),
            dtype=np.float32
        )

        print("✅ TurtleBot3Env initialized.")


    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 물리 warm-up
        for _ in range(5):
            self.robot.step(self.time_step)

        # 초기 위치 (뒤로 1m)
        self.x = -1.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0
        self.last_angular = 0
        self.path_length = 0
        self.prev_lidar_min = self.lidar_max_range

        # 목표 선택
        idx = self.np_random.integers(0, len(self.goal_list))
        self.current_goal_index = int(idx)
        self.current_goal = self.goal_list[idx].copy()

        dist = float(np.linalg.norm(self.current_goal - np.array([self.x, self.y])))
        self.last_dist = dist

        obs = self._get_state()
        return obs, {}


    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1, 1)

        v_max = 0.25
        w_max = 1.0

        linear = safe(action[0] * v_max)
        angular = safe(action[1] * w_max)

        self.last_linear = linear
        self.last_angular = angular
        self.path_length += abs(linear) * self.dt

        # Motor control
        wheel_dist = 0.160
        wheel_r = 0.033
        max_motor = 6.67

        v_l = (linear - 0.5 * angular * wheel_dist) / wheel_r
        v_r = (linear + 0.5 * angular * wheel_dist) / wheel_r

        v_l = float(np.clip(v_l, -max_motor, max_motor))
        v_r = float(np.clip(v_r, -max_motor, max_motor))

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

        self.robot.step(self.time_step)

        # Dead-reckoning update
        self.theta = (self.theta + angular * self.dt + np.pi) % (2*np.pi) - np.pi
        self.x += linear * np.cos(self.theta) * self.dt
        self.y += linear * np.sin(self.theta) * self.dt

        obs = self._get_state()
        reward, terminated, info = this_reward = self._get_reward()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info


    # =====================================================
    # STATE
    # =====================================================
    def _get_state(self):

        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar = np.nan_to_num(lidar_raw, nan=self.lidar_max_range)
        lidar = np.clip(lidar, self.lidar_min_range, self.lidar_max_range)

        lidar_norm = (lidar / self.lidar_max_range).astype(np.float32)
        self.last_lidar_min = float(np.min(lidar))

        pos = np.array([self.x, self.y], dtype=np.float32)
        diff = self.current_goal - pos
        dist = safe(np.linalg.norm(diff))
        angle = safe(np.arctan2(diff[1], diff[0]) - self.theta)
        angle = (angle + np.pi) % (2*np.pi) - np.pi

        norm_dist = np.clip(dist / self.max_goal_dist, 0, 1)
        norm_angle = np.clip(angle / np.pi, -1, 1)

        self._last_raw_dist = dist
        self._last_raw_angle = angle

        return np.concatenate([lidar_norm, [norm_dist, norm_angle]]).astype(np.float32)


    # =====================================================
    # REWARD (성공률 90~98% 성능 목표)
    # =====================================================
    def _get_reward(self):

        dist = safe(self._last_raw_dist)
        angle = safe(self._last_raw_angle)
        lidar_min = safe(self.last_lidar_min)
        prev_lidar_min = safe(self.prev_lidar_min)

        reward = 0.0

        # 초기 step 충돌 무시
        if self.step_count <= self.initial_ignore_steps:
            self.last_dist = dist
            return 0.0, False, {
                "success": False, "collision": False,
                "goal_dist": dist, "path_length": float(self.path_length)
            }

        # 1) 충돌
        if lidar_min < self.collision_threshold:
            reward = -100
            return reward, True, {
                "success": False, "collision": True,
                "goal_dist": dist, "path_length": float(self.path_length)
            }

        # 2) 성공
        if dist < self.success_threshold:
            reward = +300
            return reward, True, {
                "success": True, "collision": False,
                "goal_dist": dist, "path_length": float(self.path_length)
            }

        # 3) distance shaping
        reward += 6.0 * (self.last_dist - dist)

        # 4) angle shaping
        reward += 2.0 * (1 - abs(angle) / np.pi)

        # 5) obstacle avoidance shaping
        if lidar_min < 0.3:
            reward -= (0.3 - lidar_min) * 30.0

        if lidar_min > prev_lidar_min:
            reward += (lidar_min - prev_lidar_min) * 20.0

        # 6) forward reward
        if self.last_linear > 0.05:
            reward += 0.5 * self.last_linear

        # 7) time penalty
        reward -= 0.002

        self.last_dist = dist
        self.prev_lidar_min = lidar_min

        return reward, False, {
            "success": False,
            "collision": False,
            "goal_dist": dist,
            "path_length": float(self.path_length)
        }
