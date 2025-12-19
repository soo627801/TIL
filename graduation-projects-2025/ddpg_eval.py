# ddpg_eval.py (SUCCESS 90% 강화 버전)

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines3 import DDPG
from tb3_env import TurtleBot3Env

MODEL_PATH = "./ddpg_turtlebot3_final.zip"
RESULT_DIR = "./ddpg_eval_results"


def evaluate_ddpg(episodes=100):

    print("🚀 Starting DDPG Evaluation...")
    print(f"📌 Loading model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    model = DDPG.load(MODEL_PATH)

    print("📌 Initializing TurtleBot3 environment (EVAL mode)...")
    env = TurtleBot3Env(mode="eval")
    print("✅ TurtleBot3Env initialized (EVAL).")

    os.makedirs(RESULT_DIR, exist_ok=True)

    rewards, successes, final_dists, perturb = [], [], [], []
    goal_ids = []

    for ep in range(1, episodes + 1):

        obs, info = env.reset()

        # 🔥 eval에서 goal 분포 균형 조절: 먼 목표도 자주 나오도록
        # goal 0 (far): 50%, goal 1&2: 각 25%
        goal_probs = [0.5, 0.25, 0.25]
        idx = np.random.choice([0,1,2], p=goal_probs)

        env.current_goal_index = idx
        env.current_goal = env.goal_list[idx].copy()

        obs, info = env.reset()

        initial_dist = info["initial_dist"]
        episode_reward = 0
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, step_info = env.step(action)
            episode_reward += reward

        success = step_info["success"]
        dist = step_info["goal_dist"]
        path = step_info["path_length"]

        rewards.append(episode_reward)
        successes.append(success)
        final_dists.append(dist)
        goal_ids.append(idx)

        if initial_dist > 1e-6:
            perturb.append(path / initial_dist)
        else:
            perturb.append(np.nan)

        print(
            f"Episode {ep}/{episodes} | "
            f"Reward: {episode_reward:.2f} | "
            f"Success: {success} | "
            f"FinalDist: {dist:.3f} | Perturb: {perturb[-1]:.3f}"
        )

    env.close()

    rewards = np.array(rewards)
    successes = np.array(successes)
    final_dists = np.array(final_dists)
    perturb = np.array(perturb)
    goal_ids = np.array(goal_ids)

    sr = successes.mean() * 100
    mfd = np.nanmean(final_dists)
    mp = np.nanmean(perturb)

    print("🎯 Evaluation Finished")
    print(f"✔ Success Rate: {sr:.2f}%")
    print(f"✔ Mean Final Distance: {mfd:.3f} m")
    print(f"✔ Mean Perturbation: {mp:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = os.path.join(RESULT_DIR, f"ddpg_eval_summary_{timestamp}.txt")

    with open(summary, "w") as f:
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Success Rate: {sr:.2f}%\n")
        f.write(f"Final Distance Mean: {mfd:.3f}\n")
        f.write(f"Perturb Mean: {mp:.3f}\n")

    print(f"📁 Results saved in: {RESULT_DIR}")
    print(f"📄 Summary: {summary}")


if __name__ == "__main__":
    evaluate_ddpg(episodes=100)
