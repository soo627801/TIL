# tb3_env.py (TRAINING)

import numpy as np
import gym
from controller import Robot


def safe(x, default=0.0):
    """NaN / inf 방지용"""
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


class TurtleBot3Env(gym.Env):
    """
    Webots + TurtleBot3 + TD3 학습용 환경
    - start pose: Webots 초기 위치/자세와 일치 (0, 0, theta=0)
    - goal_list: 평가용과 동일 (3개 고정 지점)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: Robot | None = None,
        collision_threshold: float = 0.25,
        success_threshold: float = 0.30,
        max_steps: int = 400,   # 너무 짧지 않게 (학습 안정용)
    ):
        super().__init__()

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

        # ----- Pose (dead-reckoning 기준 좌표계) -----
        # Webots 초기 translation: (0, 0, 0.01), rotation: (0 1 0 0) → yaw = 0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_dist = None
        self.path_length = 0.0

        # ----- Goals (eval과 동일하게 맞춤) -----
        # 0: 정면 먼 목표, 1/2: 좌/우 근거리 목표
        self.goal_list = [
            np.array([0.0, 5.0], dtype=np.float32),
            np.array([-0.5, 0.05], dtype=np.float32),
            np.array([-0.6, -0.6], dtype=np.float32),
        ]
        self.current_goal = None
        self.current_goal_index = None

        self.max_goal_dist = 6.0

        # 파라미터
        self.collision_threshold = collision_threshold
        self.success_threshold = success_threshold
        self.max_episode_steps = max_steps
        self.step_count = 0

        # ----- Gym spaces -----
        # action: [linear(-1~1), angular(-1~1)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # obs: [lidar_norm... , norm_dist, norm_angle]
        obs_dim = self.lidar_resolution + 2
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        print("✅ TurtleBot3Env initialized (TRAIN).")


    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # 모터 정지
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Webots 물리 안정화 + LiDAR warm-up
        for _ in range(5):
            self.robot.step(self.time_step)

        # ▶ 시작 위치: Webots 초기 위치와 동일 (랜덤 X)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_length = 0.0
        self.last_lidar_min = self.lidar_max_range

        # ▶ goal 랜덤 선택 (3개 중 1개)
        idx = self.np_random.integers(0, len(self.goal_list))
        self.current_goal_index = int(idx)
        self.current_goal = self.goal_list[idx].copy()

        # 초기 거리
        self.last_dist = float(
            np.linalg.norm(self.current_goal - np.array([self.x, self.y]))
        )

        obs = self._get_state()
        info = {
            "goal_index": self.current_goal_index,
            "goal": self.current_goal.copy(),
            "initial_dist": self.last_dist,
        }
        return obs, info


    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):
        self.step_count += 1

        # action 안전 처리
        if action is None or np.isnan(action).any():
            action = np.array([0.0, 0.0], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        v_max = 0.25
        w_max = 1.0

        linear = safe(action[0] * v_max)
        angular = safe(action[1] * w_max)

        self.last_linear = linear
        self.last_angular = angular

        # 이동거리 누적
        self.path_length += abs(linear) * self.dt

        # differential drive 속도 변환
        wheel_dist = 0.160
        wheel_r = 0.033
        max_motor = 6.67

        v_l = safe((linear - 0.5 * angular * wheel_dist) / wheel_r)
        v_r = safe((linear + 0.5 * angular * wheel_dist) / wheel_r)

        v_l = float(np.clip(v_l, -max_motor, max_motor))
        v_r = float(np.clip(v_r, -max_motor, max_motor))

        self.left_motor.setVelocity(v_l)
        self.right_motor.setVelocity(v_r)

        # Webots 시뮬레이션 한 step 진행
        self.robot.step(self.time_step)

        # dead-reckoning 갱신
        self.theta = safe(self.theta + angular * self.dt)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        self.x = safe(self.x + linear * np.cos(self.theta) * self.dt)
        self.y = safe(self.y + linear * np.sin(self.theta) * self.dt)

        obs = self._get_state()
        reward, terminated, info = self._get_reward()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info


    # =====================================================
    # OBSERVATION
    # =====================================================
    def _get_state(self):
        # LiDAR 원본
        lidar_raw = np.array(self.lidar.getRangeImage(), dtype=np.float32)

        lidar = np.nan_to_num(
            lidar_raw,
            nan=self.lidar_max_range,
            posinf=self.lidar_max_range,
            neginf=self.lidar_max_range,
        )
        lidar = np.clip(lidar, self.lidar_min_range, self.lidar_max_range)

        if len(lidar) != self.lidar_resolution:
            lidar = np.resize(lidar, self.lidar_resolution)

        self.last_lidar_min = float(np.min(lidar))

        # 간단 smoothing (선택적으로)
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        lidar_padded = np.pad(lidar, (1, 1), mode="edge")
        lidar_smoothed = np.convolve(lidar_padded, kernel, mode="valid")

        lidar_norm = (lidar_smoothed / self.lidar_max_range).astype(np.float32)
        lidar_norm = np.clip(lidar_norm, 0.0, 1.0)

        # goal까지 상대 거리/각도
        pos = np.array([self.x, self.y], dtype=np.float32)
        diff = self.current_goal - pos

        dist = safe(np.linalg.norm(diff))
        angle = safe(np.arctan2(diff[1], diff[0]) - self.theta)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        norm_dist = np.clip(dist / self.max_goal_dist, 0.0, 1.0)
        norm_angle = np.clip(angle / np.pi, -1.0, 1.0)

        obs = np.concatenate(
            [lidar_norm, np.array([norm_dist, norm_angle], dtype=np.float32)]
        ).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        # reward 계산용 raw 값 저장
        self._last_raw_dist = dist
        self._last_raw_angle = angle

        return obs


    # =====================================================
    # REWARD
    # =====================================================
    def _get_reward(self):
        dist = safe(getattr(self, "_last_raw_dist", 0.0))
        angle = safe(getattr(self, "_last_raw_angle", 0.0))
        lidar_min = safe(self.last_lidar_min, default=self.lidar_max_range)

        reward = 0.0
        done = False

        # 1) progress (거리가 줄어들면 +)
        if self.last_dist is not None:
            progress = safe(self.last_dist - dist)
            reward += 10.0 * progress  # 중요 shaping

        # 2) 각도 정렬 (goal 방향으로 머리 향하면 +)
        reward += 2.0 * (1.0 - abs(angle) / np.pi)

        # 3) 전진 보상 (뒤로만 가는 정책 방지)
        if self.last_linear > 0:
            reward += 0.5 * self.last_linear

        # 4) 충돌 패널티
        if lidar_min < self.collision_threshold:
            reward -= 50.0
            done = True
            info = {
                "collision": True,
                "success": False,
                "goal_dist": dist,
                "path_length": float(self.path_length),
            }
            self.last_dist = dist
            return float(reward), done, info

        # 5) 성공 보상
        if dist < self.success_threshold:
            reward += 100.0
            done = True
            info = {
                "collision": False,
                "success": True,
                "goal_dist": dist,
                "path_length": float(self.path_length),
            }
            self.last_dist = dist
            return float(reward), done, info

        # 6) 타임 패널티 (시간 끌면 -)
        reward -= 0.001

        self.last_dist = dist

        info = {
            "collision": False,
            "success": False,
            "goal_dist": dist,
            "path_length": float(self.path_length),
        }
        return float(reward), done, info


    def render(self):
        pass

    def close(self):
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except Exception:
            pass
        return
