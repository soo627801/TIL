import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from stable_baselines3 import PPO
from tb3_env import TurtleBot3Env

def evaluate(episodes=30, max_steps=200, debug_episodes=3):
    # ROS2 초기화
    rclpy.init()
    env = TurtleBot3Env()

    MODEL_PATH = "/home/soo/ros2_ws/src/tb3_rl/tb3_rl/exp_ppo_20251024_222413/trained_ppo.zip"
    model = PPO.load(MODEL_PATH)
    print(f"\n🚀 Evaluating PPO model: {MODEL_PATH}")

    rewards, success, lengths = [], [], []

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            obs = np.asarray(obs, dtype=np.float32)
            terminated = False
            truncated = False
            total_reward = 0.0
            reached = False
            step_count = 0

            # 디버그 로그을 위해 에피소드 초반 상세 출력 여부
            debug = (ep <= debug_episodes)

            while not (terminated or truncated) and step_count < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                # action clipping
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = np.asarray(obs, dtype=np.float32)
                total_reward += float(reward)
                reached = reached or bool(info.get("reached", False))
                step_count += 1

                if debug:
                    pos = env.position.copy()
                    min_lidar = info.get("min_lidar", None)
                    dist = info.get("dist", None)
                    print(f"[DEBUG][Ep{ep}] step {step_count} | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | goal=({env.goal[0]:.2f},{env.goal[1]:.2f}) | dist={dist:.3f} | min_lidar={min_lidar:.3f} | action=({action[0]:.3f},{action[1]:.3f}) | r={reward:.3f}")

            print(f"[Eval] Episode {ep}/{episodes} | Reward: {total_reward:.2f} | Reached: {reached} | Steps: {step_count}")
            rewards.append(total_reward)
            success.append(1 if reached else 0)
            lengths.append(step_count)

        avg_reward = np.mean(rewards)
        success_rate = np.mean(success) * 100
        avg_length = np.mean(lengths)
        print("\n✅ Evaluation Complete")
        print(f"📈 Average Reward: {avg_reward:.2f}")
        print(f"🏁 Success Rate: {success_rate:.1f}%")
        print(f"⏱️ Average Steps: {avg_length:.1f}")

        # 결과 저장
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"/home/soo/ros2_ws/src/tb3_rl/tb3_rl/eval_ppo_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(f"{out_dir}/ppo_rewards.csv", rewards, delimiter=",")
        np.savetxt(f"{out_dir}/ppo_success.csv", success, delimiter=",")
        np.savetxt(f"{out_dir}/ppo_lengths.csv", lengths, delimiter=",")

        plt.plot(rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("PPO Evaluation Rewards")
        plt.legend()
        plt.savefig(f"{out_dir}/ppo_eval_plot.png")
        plt.close()

    finally:
        try:
            env.close()
            env.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    evaluate(episodes=30, max_steps=200, debug_episodes=3)
