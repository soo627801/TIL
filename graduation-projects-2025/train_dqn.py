import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import rclpy
from collections import deque
from tb3_env import TurtleBot3Env
import os, time, csv, matplotlib.pyplot as plt, numpy as np

# ✅ 로그 디렉토리 자동 생성 (매 실행마다 새로운 폴더)
ts = time.strftime("%Y%m%d_%H%M%S")
out_dir = f"/home/soo/ros2_ws/src/tb3_rl/tb3_rl/exp_dqn_{ts}"
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "episode_reward.csv")
model_path = os.path.join(out_dir, "trained_dqn.pth")
plot_path = os.path.join(out_dir, "reward_plot.png")

# -------------------------------
# Hyperparameters
# -------------------------------
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
NUM_EPISODES = 300  # 학습 에피소드 수

# -------------------------------
# DQN Network
# -------------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -------------------------------
# Training Loop
# -------------------------------
def train():
    rclpy.init()
    env = TurtleBot3Env()
    state_dim = env.observation_space.shape[0]
    action_dim = 3  # [좌회전, 직진, 우회전]

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPSILON_START
    steps_done = 0

    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action_idx = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    action_idx = torch.argmax(q_values).item()

            # Action mapping
            if action_idx == 0:
                action = [-0.5, 0.1]   # 왼쪽 회전
            elif action_idx == 1:
                action = [0.0, 0.15]   # 직진
            else:
                action = [0.5, 0.1]    # 오른쪽 회전

            # Step environment
            next_state, reward, done, _ = env.step(action)
            memory.push(state, action_idx, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Training step
            if len(memory) > BATCH_SIZE:
                s, a, r, ns, d = memory.sample(BATCH_SIZE)
                s = torch.FloatTensor(s)
                ns = torch.FloatTensor(ns)
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                d = torch.FloatTensor(d)

                q_values = policy_net(s)
                q_val = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
                next_q = target_net(ns).max(1)[0]
                expected_q = r + GAMMA * next_q * (1 - d)

                loss = (q_val - expected_q.detach()).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            steps_done += 1
            if steps_done % 200 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"[Episode {episode+1}/{NUM_EPISODES}] Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

    # -------------------------------
    # Save the trained model
    # -------------------------------
    save_path = "/home/soo/ros2_ws/src/tb3_rl/tb3_rl/trained_dqn.pth"
    torch.save(policy_net.state_dict(), save_path)
    print(f"\n✅ Model saved successfully at: {save_path}\n")

    env.close()
    rclpy.shutdown()

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train()

