import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tb3_env import TurtleBot3Env
import rclpy

# 🔹 자동으로 모델 파일 탐색
BASE_DIR = "/home/soo/ros2_ws/src/tb3_rl/tb3_rl"
MODEL_PATH = os.path.join(BASE_DIR, "trained_dqn.pth")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

# 🔹 평가 함수
def evaluate(num_episodes=50):
    rclpy.init()
    env = TurtleBot3Env()
    state_dim = env.observation_space.shape[0]
    action_dim = 3

    # DQN 모델 구조 (train_dqn.py와 동일해야 함)
    class DQN(torch.nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, action_dim)
            )
        def forward(self, x):
            return self.fc(x)

    model = DQN(state_dim, action_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print(f"✅ Loaded model from: {MODEL_PATH}")

    rewards = []
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state))
                action_idx = torch.argmax(q_values).item()

            # 행동 매핑
            if action_idx == 0:
                action = [-0.5, 0.1]
            elif action_idx == 1:
                action = [0.0, 0.15]
            else:
                action = [0.5, 0.1]

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        rewards.append(total_reward)
        print(f"[Evaluation] Episode {ep+1}: Reward {total_reward:.2f}")

    env.close()
    rclpy.shutdown()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f"\n📈 평균 보상: {mean_r:.2f} ± {std_r:.2f}")

    # 🔹 결과 저장
    result_dir = os.path.join(BASE_DIR, "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "reward_log.csv")
    plot_path = os.path.join(result_dir, "reward_plot.png")

    np.savetxt(csv_path, rewards, delimiter=",", fmt="%.4f")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Evaluation Rewards")
    plt.savefig(plot_path)
    print(f"✅ 결과 저장 완료:\n - CSV: {csv_path}\n - Plot: {plot_path}")

if __name__ == "__main__":
    evaluate()

