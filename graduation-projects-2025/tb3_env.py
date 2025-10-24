import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class TurtleBot3Env(Node, gym.Env):
    def __init__(self):
        super().__init__('turtlebot3_env')
        # ROS2 pubs/subs
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub_odom = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.sub_scan = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)

        # Executor for handling callbacks within environment
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)

        # 기본 상태
        self.lidar = np.ones(24, dtype=np.float32) * 1.0
        self.position = np.zeros(3, dtype=np.float32)  # x, y, yaw
        self.goal = self._sample_goal()

        # 환경 파라미터
        self.max_range = 3.5
        self.num_lidar = 24
        self.max_steps = 200

        # 관측공간: lidar(24, 0~1) + position(x,y, yaw bounded) + goal(x,y)
        low = np.concatenate([np.zeros(self.num_lidar), np.array([-5.0, -5.0, -np.pi]), np.array([-2.0, -2.0])])
        high = np.concatenate([np.ones(self.num_lidar), np.array([5.0, 5.0, np.pi]), np.array([2.0, 2.0])])
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        # 행동공간 (linear_vel, angular_vel)
        self.action_space = spaces.Box(low=np.array([-0.3, -1.0], dtype=np.float32),
                                       high=np.array([0.3, 1.0], dtype=np.float32),
                                       dtype=np.float32)

        # 내부 상태 트래킹
        self.prev_dist = None
        self.step_count = 0

        self.get_logger().info("✅ TurtleBot3Env (goal-aware + sync + safe lidar) initialized.")

    def _sample_goal(self):
        gx = np.random.uniform(-1.5, 1.5)
        gy = np.random.uniform(-1.5, 1.5)
        return np.array([gx, gy], dtype=np.float32)

    def odom_callback(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        # yaw 추출 (안정적 계산)
        q = msg.pose.pose.orientation
        qw, qx, qy, qz = q.w, q.x, q.y, q.z
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        self.position[2] = float(np.arctan2(siny, cosy))

    def scan_callback(self, msg):
        data = np.array(msg.ranges, dtype=np.float32)
        # NaN/Inf 처리: 기본을 max_range로
        data = np.nan_to_num(data, nan=self.max_range, posinf=self.max_range, neginf=self.max_range)
        # 일부 드라이버가 0.0을 'no measurement'로 쓸 수 있으므로 0 또는 아주 작은 값은 max_range로 대체
        data[data <= 0.01] = self.max_range
        # 균일 샘플링으로 고정 길이 생성
        if len(data) >= self.num_lidar:
            idx = np.linspace(0, len(data)-1, self.num_lidar).astype(int)
            sampled = data[idx]
        else:
            sampled = np.interp(np.linspace(0, len(data)-1, self.num_lidar), np.arange(len(data)), data)
        sampled = np.clip(sampled, 0.0, self.max_range)
        # 0~1 정규화
        self.lidar = (sampled / self.max_range).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.goal = self._sample_goal()
        self.get_logger().info(f"🎯 New goal: {self.goal}")
        self._stop_robot()
        self.prev_dist = None
        self.step_count = 0

        # 센서 초기화 안정화를 위해 약간 반복 대기
        try:
            for _ in range(5):
                self._executor.spin_once(timeout_sec=0.05)
        except Exception:
            pass

        # 디버그용 초기 센서값 로깅
        try:
            min_l = float(np.min(self.lidar) * self.max_range)
            pos = self.position.copy()
            self.get_logger().info(f"[reset] pos=({pos[0]:.3f},{pos[1]:.3f}) min_lidar={min_l:.3f}")
        except Exception:
            pass

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # 안전하게 action 클리핑
        action = np.clip(np.asarray(action, dtype=np.float32), self.action_space.low, self.action_space.high)
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.publisher_.publish(twist)

        # 퍼블리시 후 콜백 반영을 위해 반복 spin 호출 (예: 3회)
        try:
            for _ in range(3):
                self._executor.spin_once(timeout_sec=0.05)
        except Exception:
            pass

        # 상태 업데이트 관측
        obs = self._get_obs()

        # 목표 거리 계산
        dx = self.goal[0] - self.position[0]
        dy = self.goal[1] - self.position[1]
        dist = float(np.sqrt(dx**2 + dy**2))

        # 보상: 거리 감소 기반 + 시간 패널티
        if self.prev_dist is None:
            self.prev_dist = dist
        delta = self.prev_dist - dist
        reward = 1.0 * float(delta) - 0.01  # 시간 패널티
        self.prev_dist = dist

        terminated = False
        truncated = False
        reached = False

        # 도달 보상
        if dist < 0.2:
            reward += 10.0
            terminated = True
            reached = True

        # 충돌 판정 (라이다 최소값)
        min_lidar = float(np.min(self.lidar) * self.max_range)
        if min_lidar < 0.25:
            reward -= 5.0
            terminated = True

        # 스텝 카운트 증가 및 truncated 처리
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True

        info = {"reached": reached, "min_lidar": min_lidar, "dist": dist}
        done = bool(terminated or truncated)
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        # lidar(24, 0~1) + position(x,y,yaw) + goal(x,y)
        obs = np.concatenate([self.lidar, self.position, self.goal]).astype(np.float32)
        return obs

    def _stop_robot(self):
        twist = Twist()
        self.publisher_.publish(twist)

    def close(self):
        self._stop_robot()
        try:
            self._executor.remove_node(self)
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass
        self.get_logger().info("🛑 Environment closed")
