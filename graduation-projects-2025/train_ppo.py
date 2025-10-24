import os
import time
import rclpy
from tb3_env import TurtleBot3Env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train():
    rclpy.init()
    env = TurtleBot3Env()
    check_env(env)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/soo/ros2_ws/src/tb3_rl/tb3_rl/exp_ppo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=512, batch_size=64, gamma=0.99)
    print("🚀 PPO 학습 시작")

    model.learn(total_timesteps=50000)
    model_path = os.path.join(save_dir, "trained_ppo.zip")
    model.save(model_path)
    print(f"✅ PPO 모델 저장 완료: {model_path}")

    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    train()

