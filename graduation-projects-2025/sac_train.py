import os
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from tb3_env import TurtleBot3Env

class TimeLimitCallback:
    def __init__(self, max_seconds=900):
        self.max_seconds = max_seconds
        self.start = None

    def __call__(self, _locals, _globals):
        if self.start is None:
            self.start = time.time()
        return (time.time() - self.start) < self.max_seconds


def main():

    print("🚀 SAC Training Start")

    env = TurtleBot3Env()
    env = Monitor(env)

    log_dir = "./sac_tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=100_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        tau=0.02,
        ent_coef="auto",
        use_sde=False,
    )
    model.set_logger(logger)

    callback = TimeLimitCallback(max_seconds=900)

    model.learn(
        total_timesteps=int(3e6),
        callback=callback,
        progress_bar=True
    )

    model.save("sac_turtlebot3.zip")
    print("💾 Model saved.")
    print("🎉 SAC Training Finished!")


if __name__ == "__main__":
    main()
