# ddpg_train.py

import os
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from tb3_env import TurtleBot3Env


def make_env():
    def _init():
        env = TurtleBot3Env(mode="train")
        env = Monitor(env)
        return env
    return _init


def main():
    LOG_DIR = "./ddpg_tensorboard"
    MODEL_PATH = "./ddpg_turtlebot3_final.zip"

    os.makedirs(LOG_DIR, exist_ok=True)

    # VecEnv
    env = DummyVecEnv([make_env()])

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    total_timesteps = 200_000  # 필요시 늘릴 수 있음

    print("🚀 DDPG 학습 시작")
    model.learn(total_timesteps=total_timesteps, log_interval=100)
    print("💾 모델 저장:", MODEL_PATH)
    model.save(MODEL_PATH)

    env.close()
    print("✅ 학습 종료")


if __name__ == "__main__":
    main()
