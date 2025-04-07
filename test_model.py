import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from test_obs import make_env


def main():
    env = make_env()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    model = PPO.load('ppo_mario', weights_only=True) #修改这里的路径就可以切换模型进行测试
    obs = env.reset()
    done = False
    steps = 0
    max_steps = 1000
    while not done and steps < max_steps:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(f"Action: {action}, Reward: {rewards}, Done: {done}, Info: {info}")
        env.render()
        steps += 1
    env.close()


if __name__ == '__main__':
    main()
