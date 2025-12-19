# ==========================================================
#  TurtleBot3Env (supervisor 없이 충돌되던 문제 해결 버전)
#  - 시작 pose를 뒤로 1.0m 오프셋
#  - 초기 5 step 충돌 무시
#  - info에 goal_dist, path_length 추가 (평가용)
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

        # LiDAR
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
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # Goals
        self.goal_list = [
            np.array([0.0, 5.0], dtype=np.float32),
            np.array([-0.5, 0.05], dtype=np.float32),
            np.array([-0.6, -0.6], dtype=np.float32),
        ]
        self.current_goal = None
        self.current_goal_index = None
        self.max_goal_dist = 6.0

        self.collision_threshold = collision_threshold
        self.success_threshold = success_threshold
        self.max_episode_steps = max_steps
        self.step_count = 0
        self.initial_ignore_steps = 5  # 초기 5 step 충돌 무시

        # Gym spaces
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

        # 물리 warmup
        for _ in range(5):
            self.robot.step(self.time_step)

        # 🔥 Dead-reckoning start pose 뒤로 1m 이동
        self.x = -1.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # Goal 선택
        idx = self.np_random.integers(0, len(self.goal_list))
        self.current_goal_index = int(idx)
        self.current_goal = self.goal_list[idx].copy()

        dist = float(np.linalg.norm(self.current_goal - np.array([self.x, self.y])))
        self.last_dist = dist

        obs = self._get_state()
        info = {
            "success": False,
            "collision": False,
            "goal_dist": dist,
            "path_length": float(self.path_length),
            "goal_index": self.current_goal_index,
        }
        return obs, info

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

        # Motor commands
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
        reward, terminated, info = self._get_reward()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    # =====================================================
    # OBSERVATION
    # =====================================================
    def _get_state(self):

        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar = np.nan_to_num(lidar_raw, nan=self.lidar_max_range)
        lidar = np.clip(lidar, self.lidar_min_range, self.lidar_max_range)

        self.last_lidar_min = float(np.min(lidar))

        lidar_norm = (lidar / self.lidar_max_range).astype(np.float32)

        pos = np.array([self.x, self.y], dtype=np.float32)
        diff = self.current_goal - pos
        dist = safe(np.linalg.norm(diff))
        angle = safe(np.arctan2(diff[1], diff[0]) - self.theta)
        angle = (angle + np.pi) % (2*np.pi) - np.pi

        norm_dist = np.clip(dist / self.max_goal_dist, 0, 1)
        norm_angle = np.clip(angle / np.pi, -1, 1)

        obs = np.concatenate([lidar_norm, [norm_dist, norm_angle]]).astype(np.float32)

        self._last_raw_dist = dist
        self._last_raw_angle = angle

        return obs

    # =====================================================
    # REWARD
    # =====================================================
    def _get_reward(self):

        dist = self._last_raw_dist
        angle = self._last_raw_angle
        lidar_min = self.last_lidar_min

        reward = 0.0
        done = False

        # 초기 step 충돌 무시
        if self.step_count <= self.initial_ignore_steps:
            info = {
                "success": False,
                "collision": False,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), done, info

        # 충돌 판정
        if lidar_min < self.collision_threshold:
            reward -= 50.0
            info = {
                "success": False,
                "collision": True,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), True, info

        # 성공
        if dist < self.success_threshold:
            reward += 100.0
            info = {
                "success": True,
                "collision": False,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), True, info

        # shaping
        if hasattr(self, "last_dist") and self.last_dist is not None:
            reward += 10.0 * (self.last_dist - dist)
        reward += 2.0 * (1.0 - abs(angle) / np.pi)
        if self.last_linear > 0:
            reward += 0.5 * self.last_linear

        reward -= 0.001
        self.last_dist = dist

        info = {
            "success": False,
            "collision": False,
            "goal_dist": float(dist),
            "path_length": float(self.path_length),
        }
        return float(reward), False, info

    # =====================================================
    # 기타
    # =====================================================
    def render(self):
        pass

    def close(self):
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except:
            pass
        return# ==========================================================
#  TurtleBot3Env (supervisor 없이 충돌되던 문제 해결 버전)
#  - 시작 pose를 뒤로 1.0m 오프셋
#  - 초기 5 step 충돌 무시
#  - info에 goal_dist, path_length 추가 (평가용)
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

        # LiDAR
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
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # Goals
        self.goal_list = [
            np.array([0.0, 5.0], dtype=np.float32),
            np.array([-0.5, 0.05], dtype=np.float32),
            np.array([-0.6, -0.6], dtype=np.float32),
        ]
        self.current_goal = None
        self.current_goal_index = None
        self.max_goal_dist = 6.0

        self.collision_threshold = collision_threshold
        self.success_threshold = success_threshold
        self.max_episode_steps = max_steps
        self.step_count = 0
        self.initial_ignore_steps = 5  # 초기 5 step 충돌 무시

        # Gym spaces
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

        # 물리 warmup
        for _ in range(5):
            self.robot.step(self.time_step)

        # 🔥 Dead-reckoning start pose 뒤로 1m 이동
        self.x = -1.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # Goal 선택
        idx = self.np_random.integers(0, len(self.goal_list))
        self.current_goal_index = int(idx)
        self.current_goal = self.goal_list[idx].copy()

        dist = float(np.linalg.norm(self.current_goal - np.array([self.x, self.y])))
        self.last_dist = dist

        obs = self._get_state()
        info = {
            "success": False,
            "collision": False,
            "goal_dist": dist,
            "path_length": float(self.path_length),
            "goal_index": self.current_goal_index,
        }
        return obs, info

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

        # Motor commands
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
        reward, terminated, info = self._get_reward()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    # =====================================================
    # OBSERVATION
    # =====================================================
    def _get_state(self):

        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar = np.nan_to_num(lidar_raw, nan=self.lidar_max_range)
        lidar = np.clip(lidar, self.lidar_min_range, self.lidar_max_range)

        self.last_lidar_min = float(np.min(lidar))

        lidar_norm = (lidar / self.lidar_max_range).astype(np.float32)

        pos = np.array([self.x, self.y], dtype=np.float32)
        diff = self.current_goal - pos
        dist = safe(np.linalg.norm(diff))
        angle = safe(np.arctan2(diff[1], diff[0]) - self.theta)
        angle = (angle + np.pi) % (2*np.pi) - np.pi

        norm_dist = np.clip(dist / self.max_goal_dist, 0, 1)
        norm_angle = np.clip(angle / np.pi, -1, 1)

        obs = np.concatenate([lidar_norm, [norm_dist, norm_angle]]).astype(np.float32)

        self._last_raw_dist = dist
        self._last_raw_angle = angle

        return obs

    # =====================================================
    # REWARD
    # =====================================================
    def _get_reward(self):

        dist = self._last_raw_dist
        angle = self._last_raw_angle
        lidar_min = self.last_lidar_min

        reward = 0.0
        done = False

        # 초기 step 충돌 무시
        if self.step_count <= self.initial_ignore_steps:
            info = {
                "success": False,
                "collision": False,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), done, info

        # 충돌 판정
        if lidar_min < self.collision_threshold:
            reward -= 50.0
            info = {
                "success": False,
                "collision": True,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), True, info

        # 성공
        if dist < self.success_threshold:
            reward += 100.0
            info = {
                "success": True,
                "collision": False,
                "goal_dist": float(dist),
                "path_length": float(self.path_length),
            }
            return float(reward), True, info

        # shaping
        if hasattr(self, "last_dist") and self.last_dist is not None:
            reward += 10.0 * (self.last_dist - dist)
        reward += 2.0 * (1.0 - abs(angle) / np.pi)
        if self.last_linear > 0:
            reward += 0.5 * self.last_linear

        reward -= 0.001
        self.last_dist = dist

        info = {
            "success": False,
            "collision": False,
            "goal_dist": float(dist),
            "path_length": float(self.path_length),
        }
        return float(reward), False, info

    # =====================================================
    # 기타
    # =====================================================
    def render(self):
        pass

    def close(self):
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except:
            pass
        return
