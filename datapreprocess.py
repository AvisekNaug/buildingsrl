from helperfunctions import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Helper methods for the LSTM data processing
def subsequencing(df, period=1):
    counter = 0
    timegap = period * 5
    dflist = []
    for i in range(len(df.index)):
        if df.index[i] + DateOffset(minutes=timegap) not in df.index:
            dflist.append(df.iloc[counter:(i + 1), :])
            counter = i + 1
    return dflist


def inputreshaper(X, time_steps=1, outputsequence=6):
    totalArray = []
    m, n = X.shape

    for i in range(time_steps):
        # copying
        temp = np.copy(X)
        # shifting
        temp = temp[i:m + i - time_steps + 1, :]
        # appending
        totalArray.append(temp)

    # reshaping collated array to (samples, time_steps, dimensions)
    collatedArray = np.concatenate(totalArray, axis=1)
    X_reshaped = collatedArray.reshape((collatedArray.shape[0], time_steps, n))
    if outputsequence != 1:
        X_reshaped = X_reshaped[0:1 - outputsequence, :, :]
    # ^^We are removing last outputsequence-1 data due to predicting sequence of length outputsequence

    return X_reshaped


def outputreshaper(y, outputsequence=6, outputfeatures=1, time_steps=1):
    N = outputsequence
    totalArray = []
    if outputfeatures == 1:
        m = y.shape[0]  # since y is (m, )
    else:
        m, n = y.shape

    for i in range(N):
        # copying
        temp = np.copy(y)
        # shifting
        temp = temp[i + time_steps - 1: m + i - N + 1]
        # appending
        totalArray.append(temp.reshape(-1, 1))  # reshaping needed for concatenating along axis=1

    # reshaping collated array to (samples, time_steps)
    collatedArray = np.concatenate(totalArray, axis=1)
    y_reshaped = collatedArray.reshape((collatedArray.shape[0], N, outputfeatures))

    return y_reshaped


# Methods for data subsequencing


def rangesubsequencing(dflist,days=7,hours=0):
    dfchunks = []
    for h in dflist:
        chunks = (h.index[-1]-h.index[0])//Timedelta(str(days)+' days '+str(hours)+' hours')
        timeloc = h.index[0]
        for i in range(chunks):
            dfchunks.append(h.loc[timeloc : timeloc + DateOffset(days=days,hours=hours),:])
            timeloc = timeloc + DateOffset(days=days,hours=hours)
    return dfchunks


# Methods for incremental learning

def datachunks(datapath, period=12, lag=-1, smoothing=False,
               subsequence=True, outputcolumn='TotalE', days=7, hours=0, Wn = 0.015):

    # read the data set
    df = read_pickle(datapath)
    # df['TotalE'] = df['TotalE'].shift(-1) - df['TotalE']
    # df.dropna(inplace=True)
    # select part of the data
    # mask = (df.index>to_datetime('5/1/2018 23:55:00')) & (df.index<to_datetime('8/31/2019 23:55:00'))
    # df = df[mask]
    # subsample it to  timegaps
    df = df_sample(df, period)

    if smoothing:
        # window length of two hours = 120/(5*period)
        windowlength = int(120/(5*period))
        df = butterworthsmoothing(df, column_names=[outputcolumn], Wn = Wn)  # , windowlength=windowlength)

    # scale the data min-max option !
    # !!!do not scale here as data also goes to RL environment
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # df = DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)

    if subsequence:
        # extract sequences larger than L samples
        outputDFrames = subsequencing(df, period)
        minimum_seq_length = int(1440 / (5 * period))*days  # one day times days since we are
        # looking to do weekly relearning
        counteri = 0
        for i in range(len(outputDFrames)):
            if len(outputDFrames[counteri]) < minimum_seq_length:
                outputDFrames.pop(counteri)
            else:
                counteri = counteri + 1

        # reducing data to days day chunks
        outputDFrames = rangesubsequencing(outputDFrames, days=days, hours=hours)

    else: # Don't do subsequencing and don't forget state, preserve it.
        outputDFrames = [df]

    # reorganize data to predict future observation
    for i in range(len(outputDFrames)):
        outputDFrames[i][outputcolumn] = outputDFrames[i][outputcolumn].shift(lag)
        outputDFrames[i] = droprows(outputDFrames[i])

    return outputDFrames


def dflist2array(dfchunks, weekstart, slicepoint, weekend, time_steps=1, inputfeatures=5, outputfeatures=1, outputsequence=1,
                 scaling = False):

    train_df = concat(dfchunks[weekstart:slicepoint], axis=0, join='outer', sort=False)
    test_df = concat(dfchunks[slicepoint:weekend], axis=0, join='outer', sort=False)

    # Scale the data
    if scaling:
        scaler = MinMaxScaler(feature_range=(0, 1))
        temp_train_array = scaler.fit_transform(train_df.iloc[:, :-1].to_numpy())
        temp_test_array = scaler.fit_transform(test_df.iloc[:, :-1].to_numpy())

    else:
        temp_train_array = train_df.to_numpy()
        temp_test_array = test_df.to_numpy()

    # Do the initial one month of training
    train_X = inputreshaper(temp_train_array[:, :-1],
                            time_steps, outputsequence)  # (samplesize,1,4 or 5)
    train_y = outputreshaper(temp_train_array[:, -1],
                             outputsequence, outputfeatures, time_steps)  # (samplesize,1,1)
    test_X = inputreshaper(temp_test_array[:, :-1],
                           time_steps, outputsequence)  # (samplesize,1,4 or 5)
    test_y = outputreshaper(temp_test_array[:, -1],
                            outputsequence, outputfeatures, time_steps)  # (samplesize,1,1)

    return [train_X, train_y, test_X, test_y, train_df, test_df]