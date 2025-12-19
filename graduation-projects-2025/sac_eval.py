# sac_eval.py
# ==========================================================
#  SAC Evaluation for TurtleBot3
# ==========================================================

import os
import csv
import numpy as np
from stable_baselines3 import SAC
from tb3_env import TurtleBot3Env


def run_eval(model, env, episodes=10, noise_std=0.0, tag="eval"):
    success_count = 0
    collision_count = 0
    steps_list, dist_list, path_list, reward_list = [], [], [], []

    print(f"🔍 Evaluating {episodes} episodes (noise={noise_std}, tag={tag})")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            if noise_std > 0:
                action = action + np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action, -1, 1)

            obs, reward, done, truncated, info2 = env.step(action)

            ep_reward += reward
            ep_steps += 1

            if done or truncated:
                success = info2.get("success", False)
                collision = info2.get("collision", False)
                dist = info2.get("goal_dist", 0)
                path = info2.get("path_length", 0)

                if success:
                    success_count += 1
                if collision:
                    collision_count += 1

                steps_list.append(ep_steps)
                dist_list.append(dist)
                path_list.append(path)
                reward_list.append(ep_reward)

                print(f"[{tag} | EP {ep:02d}] Success={success}, Collision={collision}, "
                      f"Steps={ep_steps}, FinalDist={dist:.3f}, Path={path:.3f}")
                break

    total = episodes
    result = {
        "tag": tag,
        "episodes": total,
        "success_rate": 100 * success_count / total,
        "collision_rate": 100 * collision_count / total,
        "avg_steps": float(np.mean(steps_list)),
        "avg_final_dist": float(np.mean(dist_list)),
        "avg_path": float(np.mean(path_list)),
        "avg_reward": float(np.mean(reward_list)),
    }

    print("==============================")
    print(f"📊 SUMMARY ({tag})")
    print("==============================")
    print(f"Success Rate:   {result['success_rate']:.2f}%")
    print(f"Collision Rate: {result['collision_rate']:.2f}%")
    print(f"Avg Steps:      {result['avg_steps']:.2f}")
    print(f"Avg FinalDist:  {result['avg_final_dist']:.3f} m")
    print(f"Avg Path:       {result['avg_path']:.3f} m")
    print(f"Avg Reward:     {result['avg_reward']:.2f}")
    print("==============================")

    return result


def save_results(results, filename="sac_eval_results.csv"):
    keys = list(results[0].keys())
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"📁 Saved: {filename}")


def main():
    model_path = "./sac_turtlebot3.zip"

    print("📌 Loading SAC model...")
    model = SAC.load(model_path)
    print("✅ Model loaded.")

    print("📌 Creating environment...")
    env = TurtleBot3Env()
    print("✅ Env ready.")

    results = []
    results.append(run_eval(model, env, 30, 0.0, "base_30"))
    results.append(run_eval(model, env, 10, 0.0, "noise_0.0"))
    results.append(run_eval(model, env, 10, 0.1, "noise_0.1"))
    results.append(run_eval(model, env, 10, 0.2, "noise_0.2"))

    save_results(results)


if __name__ == "__main__":
    main()
