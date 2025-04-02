import gym
from numpy import shape
from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from gym.wrappers import GrayScaleObservation, ResizeObservation
import matplotlib.pyplot as plt
from my_SkipFrameWrappers import SkipFrameWrapper


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = SkipFrameWrapper(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))
    return env


env = make_env()
done = True
for step in range(8):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    plt.imshow(state, cmap='gray')
    plt.show()
env.close()
