from buildingenv import *
from ppoagent import *
from create_env import *
from datapreprocess import *
from plotutils import *
import os
from modelretrain import *

# %%%%%%%%%%%%%%%%%%%%%%     Specifying logging location & Parameters   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weekstart = 0  # create 12 week train and 1 week test data set
slicepoint = 12
weekend = 13
# specify logging directory
logdir = './rl_results/'
# clear old files
try:
    os.mkdir(logdir)
except FileExistsError:
    files = os.listdir(logdir)
    for f in files:
        os.remove(logdir + f)

# %%%%%%%%%%%%%%%%%%%%%%%%%%     Creating Weekly Chunks of Data       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
traindata = 'RL_relearn_data_v2.pkl'
# chunk data frame into weeks
dfchunks, spacelb, spaceub = datachunks(traindata, period=1, lag=-1, outputcolumn= 'TotalE',
                                        subsequence=True, smoothing=True, days=7, Wn =0.02)


# %%%%%%%%%%%%%%%%%%%%%    First we train a single agent for 3 months approximately      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_steps=120960*3  # training over 12 weeks or approximately 3 months for 5 times
episode_length = int(10080 / 5)
train_X, train_y, test_X, test_y, train_df, test_df = \
    dflist2array(dfchunks, weekstart, slicepoint, weekend, scaling=True)  # select data

env = Env(train_df, test_df, spacelb, spaceub, modelpath='weights.best.hdf5',
          episodelength=episode_length)   # instantiating the environment
env = wrap_env(env, logdir+'week'+str(weekend)+'.csv')  # wrapping environment for logging information


agent = get_agent(env)  # Instantiating the agent for learning the control policy
trained_model = train_agent(agent, env, steps=num_steps)  # train agent
# save fixed agent weights
trained_model.save(logdir+'fixedweights.pkl')
# save updating agent weights
trained_model.save(logdir+'updating_weights_week{}.pkl'.format(weekend))


# necessary steps to adjust the env for testing
env.env_method("testenv")

# results for updated controller
test_perf_log = test_agent(logdir+'updating_weights_week{}.pkl'.format(weekend), env)  # do testing

rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'updated_')  # Store performance of updated control

# necessary steps to adjust the env for testing
env.env_method("testenv")

# results for fixed controller
test_perf_log = test_agent(logdir+'fixedweights.pkl', env)  # do testing

rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'fixed_')  # Store performance of fixed control

# %%%%%%%%%%%%%%%%%%%%%%%%%%           Relearning Loop Begins              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Now we run a loop where we retrain the agent and compare it with fixed agent performance
weekstart = 8  # for 1 month retraining periods
common = True
while weekend<int(0.57*len(dfchunks)):

    # advance 1 week
    weekstart += 1
    slicepoint += 1
    weekend += 1

    # create new data
    train_X, train_y, test_X, test_y, train_df, test_df = \
        dflist2array(dfchunks, weekstart, slicepoint, weekend, scaling=True)  # select data
    num_steps = 2016*4*5  # 4 weeks for 5 times

    # continue LSTM model training
    lstm = load_model('weights.best.hdf5')
    retrain(lstm, train_X, train_y, test_X, test_y)

    # create the HVAC environment with new data
    env = Env(train_df, test_df, spacelb, spaceub, modelpath='weights.best.hdf5')  # instantiating the environment
    env = wrap_env(env, logdir+'week'+str(weekend)+'.csv')  # wrapping environment for logging information

    # do training
    trained_model = train_agent(agent, env, steps=num_steps)
    # save updating agent weights
    trained_model.save(logdir + 'updating_weights_week{}.pkl'.format(weekend))

    # necessary steps to adjust the env for testing
    env.env_method("testenv")

    # results for updated controller
    test_perf_log = test_agent(logdir + 'updating_weights_week{}.pkl'.format(weekend), env)  # do testing

    rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'updated_')  # Store performance of updated control

    # necessary steps to adjust the env for testing
    env.env_method("testenv")

    # results for fixed controller
    test_perf_log = test_agent(logdir + 'fixedweights.pkl', env)  # do testing

    rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'fixed_')  # Store performance of fixed control
