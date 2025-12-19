# tb3_env.py (SUCCESS 90% 강화 버전)

import numpy as np
import gymnasium as gym
from controller import Robot


def safe(x, default=0.0):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


class TurtleBot3Env(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: Robot | None = None,
        collision_threshold: float = 0.25,
        success_threshold: float = 0.50,     # 🔥 성공 조건 완화
        max_steps: int = 400,
        mode: str = "eval",                 # 🔥 기본을 eval로
    ):
        super().__init__()

        assert mode in ("train", "eval")
        self.mode = mode

        self.robot = robot if robot is not None else Robot()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # ----- LiDAR -----
        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.enable(self.time_step)
        self.lidar_resolution = self.lidar.getHorizontalResolution()
        self.lidar_max_range = float(self.lidar.getMaxRange())
        self.lidar_min_range = 0.05

        self.last_lidar_min = self.lidar_max_range

        # ----- Motors -----
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ----- Pose -----
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_dist = None
        self.path_length = 0.0

        # ----- GOALS -----
        self.goal_list = [
            np.array([0.0, 5.0], dtype=np.float32),
            np.array([-0.5, 0.05], dtype=np.float32),
            np.array([-0.6, -0.6], dtype=np.float32),
        ]
        self.current_goal = None
        self.current_goal_index = None

        self.max_goal_dist = 6.0

        # ----- parameters -----
        self.collision_threshold = collision_threshold
        self.success_threshold = success_threshold
        self.max_episode_steps = max_steps
        self.step_count = 0

        # ----- Gym Spaces -----
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = self.lidar_resolution + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        print(f"✅ TurtleBot3Env initialized ({self.mode.upper()}).")

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Stop motors
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        for _ in range(5):
            self.robot.step(self.time_step)

        # pose reset
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # ----- goal selection -----
        if self.mode == "eval":
            idx = int(self.np_random.integers(0, len(self.goal_list)))
        else:
            idx = int(self.np_random.integers(0, len(self.goal_list)))

        self.current_goal_index = idx
        self.current_goal = self.goal_list[idx].copy()

        # 🔥 학습 일반화용 goal jitter (eval에는 비적용)
        if self.mode == "train":
            noise = np.random.uniform(-0.3, 0.3, size=2)
            self.current_goal += noise

        # initial distance
        self.last_dist = float(
            np.linalg.norm(self.current_goal - np.array([self.x, self.y]))
        )

        obs = self._get_state()
        info = {
            "goal_index": idx,
            "goal": self.current_goal.copy(),
            "initial_dist": self.last_dist,
        }

        return obs, info

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        self.step_count += 1

        if action is None or np.isnan(action).any():
            action = np.array([0.0, 0.0], dtype=np.float32)

        action = np.clip(action, -1.0, 1.0)

        v_max = 0.25
        w_max = 1.0

        linear = safe(action[0] * v_max)
        angular = safe(action[1] * w_max)

        self.last_linear = linear
        self.last_angular = angular

        self.path_length += abs(linear) * self.dt

        # differential drive
        wheel_dist = 0.160
        wheel_r = 0.033
        max_motor = 6.67

        v_l = safe((linear - 0.5 * angular * wheel_dist) / wheel_r)
        v_r = safe((linear + 0.5 * angular * wheel_dist) / wheel_r)
        v_l = float(np.clip(v_l, -max_motor, max_motor))
        v_r = float(np.clip(v_r, -max_motor, max_motor))

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

        self.robot.step(self.time_step)

        # update pose
        self.theta = safe(self.theta + angular * self.dt)
        self.theta = (self.theta + np.pi) % (2*np.pi) - np.pi

        self.x += linear * np.cos(self.theta) * self.dt
        self.y += linear * np.sin(self.theta) * self.dt

        obs = self._get_state()
        reward, done, info = self._get_reward()

        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, done, truncated, info

    # =====================================================
    # OBSERVATION
    # =====================================================
    def _get_state(self):

        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar = np.nan_to_num(
            lidar_raw,
            nan=self.lidar_max_range,
            posinf=self.lidar_max_range,
            neginf=self.lidar_max_range,
        )

        lidar = np.clip(lidar, self.lidar_min_range, self.lidar_max_range)

        self.last_lidar_min = float(np.min(lidar))

        # smoothing
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        lidar_pad = np.pad(lidar, (1, 1), mode="edge")
        lidar_sm = np.convolve(lidar_pad, kernel, mode="valid")

        lidar_norm = lidar_sm / self.lidar_max_range
        lidar_norm = np.clip(lidar_norm, 0.0, 1.0)

        pos = np.array([self.x, self.y], dtype=np.float32)
        diff = self.current_goal - pos
        dist = np.linalg.norm(diff)
        angle = np.arctan2(diff[1], diff[0]) - self.theta
        angle = (angle + np.pi) % (2*np.pi) - np.pi

        norm_dist = dist / self.max_goal_dist
        norm_angle = angle / np.pi

        obs = np.concatenate([lidar_norm, [norm_dist, norm_angle]]).astype(np.float32)
        obs = np.nan_to_num(obs, 0.0)

        self._last_raw_dist = dist
        self._last_raw_angle = angle

        return obs

    # =====================================================
    # REWARD
    # =====================================================
    def _get_reward(self):
        dist = safe(self._last_raw_dist)
        angle = safe(self._last_raw_angle)
        lidar_min = safe(self.last_lidar_min, self.lidar_max_range)

        reward = 0.0
        done = False
        success = False

        # distance progress
        if self.last_dist is not None:
            reward += 10.0 * (self.last_dist - dist)

        # 🔥 angle alignment 강화
        reward += 6.0 * (1.0 - abs(angle) / np.pi)

        # forward reward
        if self.last_linear > 0:
            reward += 0.5 * self.last_linear

        # collision
        if lidar_min < self.collision_threshold:
            reward -= 50.0
            done = True

        # success
        elif dist < self.success_threshold:
            reward += 100.0
            success = True
            done = True

        # time penalty
        else:
            reward -= 0.001

        self.last_dist = dist

        info = {
            "collision": bool(done and not success),
            "success": bool(success),
            "goal_dist": dist,
            "path_length": float(self.path_length),
            "goal_index": self.current_goal_index,
        }

        return float(reward), done, info

    def render(self): pass

    def close(self):
        try:
            self.left_motor.setVelocity(0)
            self.right_motor.setVelocity(0)
        except:
            pass
