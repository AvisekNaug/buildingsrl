from stable_baselines.common.vec_env import DummyVecEnv  # SubprocVecEnv,
# from stable_baselines.common import set_global_seeds
from stable_baselines.bench import Monitor

# def make_env(env, rank, seed=0):
#     """
#     Utility function for multiprocessed env.
#
#     :param env: (gym.Env) the environment
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env.seed(seed + rank)
#         return env
#     set_global_seeds(seed)
#     return _init

def wrap_env(env, rl_logs):
    # Wrap the environment for monitoring
    env = Monitor(env, rl_logs, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    # num_cpu = 2  # Number of processes to use
    # Create the vectorized environment
    # env = SubprocVecEnv([env for i in range(num_cpu)])

    return env
