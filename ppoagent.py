import numpy as np
import tensorflow as tf

from stable_baselines import PPO2


def get_agent(env):
    """
    The Proximal Policy Optimization algorithm combines ideas from A2C
    (having multiple workers) and TRPO (it uses a trust region to improve the actor)
    """

    # Custom MLP policy
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[dict(pi=[16, 16, 16, 16], vf=[16, 16, 16, 16])])

    # Create the agent
    agent = PPO2("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    return agent

def train_agent(agent, rl_logs, env=None, steps=30000):
    """
    Train the agent on the environment
    """
    agent.set_env(env)
    agent.learn(total_timesteps=steps, callback=SaveBest, tb_log_name="ppo2_event_folder")

best_mean_reward = -np.inf
n_steps = 0
reward_path = './episode_reward/'

def SaveBest(_locals, _globals):
    """
    Store neural network weights during training if the current episode's
    performance is better than the previous best performance.
    """

    global best_mean_reward, n_steps, reward_path

    self_ = _locals['self']
    episode_reward = self_.episode_reward


