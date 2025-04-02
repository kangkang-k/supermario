import gym
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from test_obs import make_env


def main():
    # env = make_env()
    vec_env = SubprocVecEnv([make_env for _ in range(4)])
    vec_env = VecFrameStack(vec_env, 4, channels_order='last')
    eval_callback = EvalCallback(vec_env, best_model_save_path='./best_model/', log_path='./callback_logs/',
                                 eval_freq=10000 // 4)
    model = PPO('CnnPolicy', vec_env, verbose=1, tensorboard_log='logs',
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=2048,
                n_epochs=10,
                ent_coef=0.1,
                target_kl=0.2,
                gamma=0.97,
                clip_range=0.2)
    model.learn(total_timesteps=1e7, callback=eval_callback)
    model.save('ppo_mario')


if __name__ == '__main__':
    main()
