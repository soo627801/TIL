import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from tb3_env import TurtleBot3Env


def perturb_action(action, noise_std=0.05):
    """Perturbation 테스트용: 행동에 작은 노이즈 추가."""
    noise = np.random.normal(0, noise_std, size=len(action))
    return np.clip(action + noise, -1.0, 1.0)


def evaluate(
    model_path: str = "./td3_tb3_final.zip",
    n_episodes: int = 100,
    collision_threshold: float = 0.25,
    success_threshold: float = 0.30,
    max_steps: int = 200,
    csv_path: str = "td3_eval_results.csv",
    save_graph_dir: str = "eval_graphs",
):
    print("📊 Initializing evaluation environment...")
    env = TurtleBot3Env(
#        mode="eval",
        collision_threshold=collision_threshold,
        success_threshold=success_threshold,
        max_steps=max_steps,
    )

    print("📊 Loading TD3 model...")
    model = TD3.load(model_path)

    os.makedirs(save_graph_dir, exist_ok=True)

    # 기록 저장용 리스트
    results = []
    rewards_hist = []
    dist_hist = []
    success_hist = []

    # 전체 통계
    success_count = 0
    collision_count = 0

    sum_final_dist = 0.0
    sum_reward = 0.0
    sum_steps = 0
    sum_path = 0.0

    # goal별 저장 구조
    per_goal = {}

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        goal_idx = int(info.get("goal_index", -1))

        done = False
        truncated = False

        ep_reward = 0.0
        ep_steps = 0
        ep_collision = False
        ep_success = False
        ep_path = 0.0
        final_dist = info.get("initial_dist", np.nan)

        # perturbation 30% 확률
        perturb = np.random.rand() < 0.3

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if perturb:
                action = perturb_action(action)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1
            ep_path = float(info.get("path_length", ep_path))
            final_dist = float(info.get("goal_dist", final_dist))

            # 상태 체크
            if bool(info.get("collision", False)):
                ep_collision = True
            if bool(info.get("success", False)):
                ep_success = True

            done = terminated

        # 통계 누적
        sum_final_dist += final_dist
        sum_reward += ep_reward
        sum_steps += ep_steps
        sum_path += ep_path

        if ep_success:
            success_count += 1
        if ep_collision:
            collision_count += 1

        # 로그
        print(
            f"Episode {ep} | Goal={goal_idx} | Reward={ep_reward:.2f} | "
            f"Steps={ep_steps} | Final dist={final_dist:.3f} | "
            f"Success={ep_success} | Collision={ep_collision} | Path={ep_path:.3f} m"
        )

        # 기록 저장
        rewards_hist.append(ep_reward)
        dist_hist.append(final_dist)
        success_hist.append(int(ep_success))

        results.append(
            {
                "episode": ep,
                "goal_index": goal_idx,
                "reward": ep_reward,
                "steps": ep_steps,
                "final_dist": final_dist,
                "success": int(ep_success),
                "collision": int(ep_collision),
                "path_length": ep_path,
                "perturb": int(perturb),
            }
        )

        # goal별 누적
        if goal_idx not in per_goal:
            per_goal[goal_idx] = {
                "episodes": 0,
                "success": 0,
                "collision": 0,
                "sum_dist": 0.0,
                "sum_reward": 0.0,
                "sum_steps": 0,
                "sum_path": 0.0,
            }

        g = per_goal[goal_idx]
        g["episodes"] += 1
        g["sum_dist"] += final_dist
        g["sum_reward"] += ep_reward
        g["sum_steps"] += ep_steps
        g["sum_path"] += ep_path
        if ep_success:
            g["success"] += 1
        if ep_collision:
            g["collision"] += 1

    # === 총합 결과 ===
    n = len(results)
    success_rate = success_count / n
    collision_rate = collision_count / n

    avg_final_dist = sum_final_dist / n
    avg_reward = sum_reward / n
    avg_steps = sum_steps / n
    avg_path = sum_path / n

    print("\n===== Evaluation Results =====")
    print(f"총 에피소드: {n}")
    print(f"🔥 성공률: {success_rate * 100:.2f}%")
    print(f"💥 충돌률: {collision_rate * 100:.2f}%")
    print(f"📍 평균 최종 거리: {avg_final_dist:.3f} m")
    print(f"📉 평균 이동거리: {avg_path:.3f} m")
    print("================================\n")

    # === CSV 저장 ===
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "goal_index",
                "reward",
                "steps",
                "final_dist",
                "success",
                "collision",
                "path_length",
                "perturb",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"📁 CSV 저장 완료: {csv_path}")

    # === 그래프 저장 ===
    plt.figure(figsize=(12,4))
    plt.plot(success_hist, label="Success (0/1)")
    plt.title("Success Trend")
    plt.savefig(os.path.join(save_graph_dir, "success_trend.png"))
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(dist_hist, label="Final Distance")
    plt.title("Final Distance Trend")
    plt.savefig(os.path.join(save_graph_dir, "distance_trend.png"))
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(rewards_hist, label="Reward")
    plt.title("Reward Trend")
    plt.savefig(os.path.join(save_graph_dir, "reward_trend.png"))
    plt.close()

    print(f"📊 그래프 저장 완료: {save_graph_dir}/")

    # === Goal별 상세 분석 출력 ===
    print("\n===== Goal별 상세 분석 =====")
    for gid, g in sorted(per_goal.items()):
        geps = g["episodes"]
        if geps == 0:
            continue

        print(f"\n🎯 Goal {gid}")
        print(f"  성공률: {(g['success'] / geps) * 100:.2f}%")
        print(f"  평균 거리: {g['sum_dist'] / geps:.3f} m")
        print(f"  평균 스텝: {g['sum_steps'] / geps:.1f}")
        print(f"  평균 경로 길이: {g['sum_path'] / geps:.3f} m")


def main():
    evaluate(
        model_path="td3_tb3_final.zip",
        n_episodes=100,
        csv_path="td3_eval_results.csv",
        save_graph_dir="eval_graphs"
    )


if __name__ == "__main__":
    main()
