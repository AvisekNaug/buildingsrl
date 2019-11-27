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
logdir = './rl_results_local/'
# clear old files
try:
    os.mkdir(logdir)
except FileExistsError:
    files = os.listdir(logdir)
    for f in files:
        os.remove(logdir + f)

traindata = 'RL_relearn_dataV2.pkl'

# %%%%%%%%%%%%%%%%%%%%%%%%%%     Creating Weekly Chunks of Data       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# chunk data frame into weeks
dfchunks = datachunks(traindata, period=1, lag=-1, outputcolumn= 'TotalE',
                      subsequence=True, smoothing=True, days=7, Wn =0.02)


# %%%%%%%%%%%%%%%%%%%%%    First we train a single agent for 3 months approximately      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_steps=120960*3  # training over 12 weeks or approximately 3 months for 5 times
train_X, train_y, test_X, test_y, train_df, test_df = \
    dflist2array(dfchunks, weekstart, slicepoint, weekend, scaling=True)  # select data
env = Env(train_df, test_df, modelpath='weights.best.hdf5')  # instantiating the environment
agent = get_agent(env)  # Instantiating the agent for learning the control policy
train_metrics = train_agent(agent, env, steps=num_steps, dest=logdir + "fixed_agent_weights.h5f")  # train agent
rl_reward_save(train_metrics, logdir)  # save training metrics

env.testenv()  # set env to test mode

# results for updated controller
test_perf_log = test_agent(agent, env,
                           weights=logdir + "fixed_agent_weights.h5f")  # do testing
rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'updated_')  # Store performance of updated control

# results for fixed controller
test_perf_log = test_agent(agent, env,
                           weights=logdir + "fixed_agent_weights.h5f")  # do testing
rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'fixed_')  # Store performance of fixed control

# %%%%%%%%%%%%%%%%%%%%%%%%%%           Relearning Loop Begins              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Now we run a loop where we retrain the agent and compare it with fixed agent performance
weekstart = 8  # for 1 month retraining periods
common = True
while weekend<int(0.6*len(dfchunks)):

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
    env = Env(train_df, test_df, modelpath='weights.best.hdf5')  # instantiating the environment

    # Instantiating the agent for learning the control policy
    agent = get_agent(env)
    # if common:
    #   agent.load_weights(logdir + "fixed_agent_weights.h5f")
    #   common = False
    # else:
    #   agent.load_weights(logdir + 'Week' + str(weekend-1) + "_agent_weights.h5f")

    # do training
    train_metrics = train_agent(agent, env, steps=num_steps,
                                dest=logdir + 'Week' + str(weekend) + "_agent_weights.h5f")

    # save training metrics
    rl_reward_save(train_metrics, './rl_results_local/')

    env.testenv()  # set env to test mode
    # results for updated controller
    test_perf_log = test_agent(agent, env,
                               weights=logdir + 'Week' + str(weekend) + "_agent_weights.h5f")  # do testing
    rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'updated_')  # Store performance of updated control

    env.testenv()  # set env to test mode
    # results for fixed controller
    test_perf_log = test_agent(agent, env,
                               weights='./rl_results_local/fixed_agent_weights.h5f')  # do testing
    rl_perf_save(test_perf_log, logdir + 'Week' + str(weekend) + 'fixed_')  # Store performance of fixed control
