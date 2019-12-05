from pandas import *
import numpy as np
from keras.models import load_model

# importing the modules for setting up the environment from which the
# algorithm learns the control policy to be implemented
import gym
from gym import spaces
from gym.utils import seeding


# This class describes the formal environment which the reinforcement learning
# interacts with. It inherits some properties from the gym imported earlier
class Env(gym.Env):
    def __init__(self, train_df, test_df, spacelb, spaceub,
                 modelpath: str = 'weights.best.hdf5',
                 episodelength = int(10080 / 5)):

        '''Here we initialize the data driven model for evaluating energy
            The weights and biases of the models are stored in a file'''
        self.model = load_model(modelpath)

        '''choosing the weights of the rewardfn'''
        self.w1 = 0.6
        self.w2 = 0.4

        '''Reading the weather data+current dt data for the simulation and doing some pre processing steps'''
        self.rawdata = concat([train_df, test_df], axis=0,
                              join='outer', sort=False)  # 'OAT', 'OAH', 'Ghi', 'SAT', 'TotalE', 'AvgStpt'
        self.Stpt = self.rawdata.iloc[:, [-1]]  # separate data for AvgStpt
        self.dataSet = self.rawdata.iloc[:, :-2]  # removing last column: TotalE
        self.m, self.n = self.dataSet.shape
        # getting 0:mean 1:std 2:min 3:max--> array of shape (metric(4), number of cols(4))
        self.Stats = self.dataSet.describe().iloc[[1, 2, 3, 7], :].to_numpy()

        '''Windowed Stats: Assuming a window of 3 hours'''
        self.win_len = 36  # we look at 3.0 hr data
        self.windowMean = self.dataSet.rolling(self.win_len, min_periods=1).mean()['OAT']
        self.windowMax = self.dataSet.rolling(self.win_len, min_periods=1).max()['OAT']
        self.windowMin = self.dataSet.rolling(self.win_len, min_periods=1).min()['OAT']

        '''Standard requirements for interfacing with Keras-RL's code'''
        # spacelb = [self.Stats[2, i] for i in range(self.n)]
        # spaceub = [self.Stats[3, i] for i in range(self.n)]
        self.observation_space = spaces.Box(low=np.array(spacelb),
                                            high=np.array(spaceub),
                                            dtype=np.float32)
        # self.action_space = spaces.Box(low=np.array([self.Stats[2,3]]), high=np.array([self.Stats[3,3]]),
        #          #                              dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1]),
                                       high=np.array([1]),
                                       dtype=np.float32)  # since stable baselines does not allow scaling
        self.actionspacehigh = 75
        self.actionspacelow = 55

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # counter: counts the current step number in an episode
        # episode length: dictates number of steps in an episode
        # testing: whether we env is in testing phase or not
        # dataPtr: steps through the entire available data in a cycle- gets
        #           reset to 0 when entire trainData is used up

        self.counter = 0
        self.episodelength = episodelength
        self.testing = False
        self.dataPtr = 0

        # slicing data into train and test sequences
        self.slicepoint = 0.85
        self.traindatalimit = int(self.slicepoint * self.m)
        self.testdatalimit = self.m

        '''Resetting the environment to its initial value'''
        self.S = self.dataSet.iloc[self.dataPtr, :].to_numpy()
        self.state = self.S.flatten()

    def testenv(self):
        self.testing = True
        self.dataPtr = self.traindatalimit

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def controlmap(self, action):
        return 0.5*(action[0]+1)*(self.actionspacehigh-self.actionspacelow)+self.actionspacelow

    def step(self, controlact):
        """
        A normalized energy cost modelled for the building based on its current
        state and the control action (degrees F).
        """

        self.state = self.S.flatten()
        oldenergy = self.costfn(self.state)
        # update the state
        controlact = self.controlmap(controlact)
        self.state[3] = controlact
        rlenergy = self.costfn(self.state)

        # ambient temperature based control
        w_mean = self.windowMean.iloc[self.dataPtr]
        w_max = self.windowMax.iloc[self.dataPtr]
        w_min = self.windowMin.iloc[self.dataPtr]

        # prevent cases where data is anomalous
        ideal_dt = 68
        if w_max != w_min:
            # implementing the safety heuristics based on Darren's safety recommendation
            if self.state[0] > 70:  # ie over the past few hours, temperature is over 70F -- hot weather
                ideal_dt = 10 * (2 * w_mean - self.state[0] - w_min) / (w_max - w_min) + 56
            elif self.state[0] < 58:  # ie over past few hrs temperature is in the intermediate range -- cold weather
                ideal_dt = 3 * (2 * w_mean - self.state[0] - w_min) / (w_max - w_min) + 65
            else:  # ie over the past few hours temperature is between 70 and 55 -- intermediate weather
                ideal_dt = 8 * (2 * w_mean - self.state[0] - w_min) / (w_max - w_min) + 58

        penalty = np.abs(ideal_dt - controlact)
        reward = -self.w1 * rlenergy - self.w2 * penalty

        step_info = {}
        if self.testing:
            # Update callback info with new values
            step_info = {'rl_energy': rlenergy,
                         'old_energy': oldenergy,
                         'oat': self.state[0],
                         'dat': controlact}

        self.counter += 1
        self.dataPtr += 1

        # adjust proper indexing of sequential train and test data
        if not self.testing:
            if self.dataPtr > self.traindatalimit - 1:
                self.dataPtr = 0
        else:
            if self.dataPtr > self.testdatalimit - 1:
                self.dataPtr = self.traindatalimit

        # see if episode has ended
        done = False
        if self.counter > self.episodelength - 1:
            done = True

        # proceed to the next state
        self.S = self.dataSet.iloc[self.dataPtr, :].to_numpy()
        self.state = self.S.flatten()
        self.state[3] = controlact

        return self.state, float(reward), done, step_info

    # Resetting the state of the environment after a pre specified amount of time has passed
    # This interval corresponds to the data that is available for that time period.
    def reset(self):
        self.S = self.dataSet.iloc[self.dataPtr, :].to_numpy()
        self.state = self.S.flatten()
        self.counter = 0
        self.steps_beyond_done = None
        return self.state

    def costfn(self, inputvector):
        # if using min max scaled LSTM
        inputvector = np.divide(np.subtract(inputvector, self.Stats[2, :]),
                                np.subtract(self.Stats[3, :], self.Stats[2, :]))
        inputvector = inputvector.reshape(1, 1, inputvector.shape[0])  # reshape to (batchsize, timesteps, features)
        return self.model.predict(inputvector, batch_size=1)[0, 0]