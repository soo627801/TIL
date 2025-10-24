import torch
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tb3_env import TurtleBot3Env
from train_dqn import DQN

# -------------------------------
# 로봇 자율주행 실행 (학습된 모델 사용)
# -------------------------------

class DQNRunner(Node):
    def __init__(self):
        super().__init__('dqn_runner')

        # 환경 설정
        self.env = TurtleBot3Env()
        state_dim = self.env.observation_space.shape[0]
        action_dim = 3

        # 학습된 모델 불러오기
        self.model = DQN(state_dim, action_dim)
        model_path = "/home/soo/ros2_ws/src/tb3_rl/tb3_rl/trained_dqn.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.get_logger().info("✅ Trained DQN model loaded successfully!")

        # 실행 루프 타이머
        self.timer = self.create_timer(0.1, self.run_step)
        self.state = self.env.reset()

    def run_step(self):
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(self.state))
            action_idx = torch.argmax(q_values).item()

        if action_idx == 0:
            action = [-0.5, 0.1]   # 왼쪽 회전
        elif action_idx == 1:
            action = [0.0, 0.15]   # 직진
        else:
            action = [0.5, 0.1]    # 오른쪽 회전

        next_state, reward, done, _ = self.env.step(action)
        self.state = next_state

        if done:
            self.get_logger().info("🏁 Episode finished. Resetting environment.")
            self.state = self.env.reset()


def main(args=None):
    rclpy.init(args=args)
    node = DQNRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.env.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

