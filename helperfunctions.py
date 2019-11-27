from pandas import *
import scipy.signal as signal

# Helper methods for the data collection
def fileReader(pathtofile, dateheading, dtformat='%m/%d/%Y %H:%M', offset=0):
    """
    reads files in Bdx format and returns a list of data frames with parsed time
    :param pathtofile: type str; the folder or path from which we read individual .csv or .excel files
    :param dateheading: type str; the column name for date varies, so it is required
    :param dtformat: format string for datetime parsing
    :param offset: {str} -- hours to offset incase of zone adjustment
    :return: list of dataframes
    """
    # Read the files
    dlist = []
    if pathtofile.endswith('.csv'):
        dfr = read_csv(pathtofile)
    if pathtofile.endswith('.xlsx'):
        dfr = read_excel(pathtofile)
    else:
        dfr = read_pickle(pathtofile)

    # Parsing the Date column
    dfr.insert(loc=0, column='Dates',
               value=to_datetime(dfr[dateheading],
                                 format=dtformat) + DateOffset(hours=offset))

    dfr.drop(dateheading, axis=1, inplace=True)  # Drop original Time column

    # Add df to the dlist
    dlist.append(dfr)

    return dlist

def merge_df_rows(dlist):
    """
    Merge rows of dataframes sharing same columns but different time points
    Always Call merge_df_rows before calling merge_df_columns as time has not been set as
    index yet
    :param dlist: list of dataframes to be added along rows
    :return: dataframe
    """

    # Create Dataframe from the dlist files
    dframe = concat(dlist, axis=0, join='outer', sort=False)

    # Sort the df based on the datetime index
    dframe.sort_values(by='Dates', inplace=True)

    # Setting Dates as the dataframe index
    dframe.set_index(['Dates'], drop=True, inplace=True)

    # Dropiing duplicated time points that may exist in the data
    dframe = dframe[~dframe.index.duplicated()]

    return dframe

def merge_df_columns(dlist):
    """
    Merge dataframes  sharing same rows but different columns
    :param dlist: list of dataframes to be along column axis
    :return: concatenated dataframe
    """
    df = concat(dlist, axis=1, join='outer', sort=False)
    df = droprows(df)

    return df

def droprows(df):
    return df.dropna(axis=0, how='any')

def df_sample(df, period=12):
    """
    resamples dataframe at "period" 5 minute time points
    :param df:
    :param period: number of 5 min time points
    :return: sampled dataframe
    """
    timegap = period * 5
    return df[df.index.minute % timegap == 0]

def butterworthsmoothing(df, column_names: list = None, Wn = 0.015):
    """
    Smoothes the dataframe columns
    :param df: the input datafrme
    :param column_names: list of column names to be smoothed
    :return: smoothed data frame
    """

    if column_names is None:
        return df
    else:
        # First, design the Buterworth filter
        N = 2  # Filter order
        Wn = Wn  # Cutoff frequency
        B, A = signal.butter(N, Wn, output='ba')
        for i in column_names:
            df[i] = signal.filtfilt(B, A, df[i])
        df.dropna(axis=0, how='any', inplace=True)
        return df