from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from tb3_env import TurtleBot3Env
import numpy as np

def main():
    env = TurtleBot3Env()

    # 탐험 노이즈 (빠르게 환경 탐색하도록)
    action_noise = NormalActionNoise(
        mean=np.zeros(2),
        sigma=0.15 * np.ones(2)
    )

    model = TD3(
        policy="MlpPolicy",
        env=env,
        action_noise=action_noise,

        # ▶ Fast Learning 핵심 파라미터
        learning_rate=3e-4,
        batch_size=64,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=1,
        learning_starts=500,
        tau=0.02,

        # 안정성 위해 clipping
        #target_policy_smoothing=0.1,
        verbose=1,
        tensorboard_log="./fast_td3_tensorboard/"
    )

    # 약 8~12분
    model.learn(total_timesteps=80000)

    model.save("fast_td3")
    env.close()
    print("🎉 Fast TD3 Training Completed!")

if __name__ == "__main__":
    main()
