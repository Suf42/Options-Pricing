import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from parser import dtype, col_num, description
from cnnLSTM import cnnLSTM

pd.options.display.max_columns = 8
pd.options.display.max_rows = 500
pd.options.display.width = 0

TIME_SERIES_WINDOW_LENGTH = 10
TIME_SERIES_LAG_STRIDE = 1

# python main.py '../datasets/spy_2020_2022.csv'



def read_dataset(file_path : str) -> pd.DataFrame:
    '''
    Read in the Historical End of Day Options Price Quotes for $SPY Option Chains between Q1 2019 and Q4 2022.
    Ensure correct typing and handle missing values.

    Parameters
    ----------
    file_path : str
        The file path to the dataset.

    Returns
    -------
    source_df : pd.DataFrame
        The preliminary pandas dataframe of the historical pricing data.
    '''

    ## Converter functions to replace missing values and remove trailing whitespace

    def remove_whitespace_int(col):
        if col[0] == ' ':
            return int(col[1:])

        return int(col)

    def remove_whitespace_float(col):
        if col == ' ':
            return np.nan

        return float(col[1:])

    def remove_whitespace_string(col):
        return col[1:]



    converters = {
        '[QUOTE_UNIXTIME]': remove_whitespace_int, 
        ' [EXPIRE_UNIX]': remove_whitespace_int, 
        ' [C_DELTA]': remove_whitespace_float, 
        ' [C_GAMMA]': remove_whitespace_float, 
        ' [C_VEGA]': remove_whitespace_float, 
        ' [C_THETA]': remove_whitespace_float, 
        ' [C_RHO]': remove_whitespace_float, 
        ' [C_IV]': remove_whitespace_float, 
        ' [C_VOLUME]': remove_whitespace_float, 
        ' [C_LAST]': remove_whitespace_float, 
        ' [C_BID]': remove_whitespace_float, 
        ' [C_ASK]': remove_whitespace_float, 
        ' [P_BID]': remove_whitespace_float, 
        ' [P_ASK]': remove_whitespace_float, 
        ' [P_LAST]': remove_whitespace_float, 
        ' [P_DELTA]': remove_whitespace_float, 
        ' [P_GAMMA]': remove_whitespace_float, 
        ' [P_VEGA]': remove_whitespace_float, 
        ' [P_THETA]': remove_whitespace_float, 
        ' [P_RHO]': remove_whitespace_float, 
        ' [P_IV]': remove_whitespace_float, 
        ' [P_VOLUME]': remove_whitespace_float, 
        ' [C_SIZE]': remove_whitespace_string, 
        ' [P_SIZE]': remove_whitespace_string 
    }

    parse_dates = [
        col_num['[QUOTE_UNIXTIME]'], 
        col_num[' [QUOTE_READTIME]'], 
        col_num[' [QUOTE_DATE]'], 
        col_num[' [EXPIRE_DATE]'], 
        col_num[' [EXPIRE_UNIX]'] 
    ]

    df = pd.read_csv(
        file_path, 
        sep = ',', 
        header = 0, 
        engine = 'c', 
        converters = converters, 
        # nrows = 5000, 
        parse_dates = parse_dates, 
        on_bad_lines = 'skip', 
        low_memory = False 
    )

    df = df.astype(dtype = dtype)

    return df



def preprocess_dataset(spy_df : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Preprocess the historical pricing data pandas dataframe.
    Actions:
        1. Split dataframe into call and put option lisitings
        2. Noise removal
        3. Pruning
        4. Normalisation (min-max scaling)
        5. Dimensionality reduction

    Parameters
    ----------
    spy_df : pd.DataFrame
        The preliminary pandas dataframe of the historical pricing data.

    Returns
    -------
    cleaned_spy_call_df : pd.DataFrame
        The cleaned up version of the historical call option pricing data pandas dataframe.
    cleaned_spy_put_df : pd.DataFrame
        The cleaned up version of the historical put option pricing data pandas dataframe.
    '''

    ## 1. Separate the listing of all available option contracts into call and put option listings

    call_columns = [
        col_num['[QUOTE_UNIXTIME]'], 
        col_num[' [EXPIRE_UNIX]'], 
        col_num[' [DTE]'], 
        col_num[' [C_DELTA]'], 
        col_num[' [C_GAMMA]'], 
        col_num[' [C_VEGA]'], 
        col_num[' [C_THETA]'], 
        col_num[' [C_RHO]'], 
        col_num[' [C_IV]'], 
        col_num[' [C_VOLUME]'], 
        col_num[' [C_SIZE]'], 
        col_num[' [C_BID]'], 
        col_num[' [C_ASK]'], 
        col_num[' [C_LAST]'], 
        col_num[' [UNDERLYING_LAST]'], 
        col_num[' [STRIKE]'] 
    ]

    put_columns = [
        col_num['[QUOTE_UNIXTIME]'], 
        col_num[' [EXPIRE_UNIX]'], 
        col_num[' [DTE]'], 
        col_num[' [P_DELTA]'], 
        col_num[' [P_GAMMA]'], 
        col_num[' [P_VEGA]'], 
        col_num[' [P_THETA]'], 
        col_num[' [P_RHO]'], 
        col_num[' [P_IV]'], 
        col_num[' [P_VOLUME]'], 
        col_num[' [P_SIZE]'], 
        col_num[' [P_BID]'], 
        col_num[' [P_ASK]'], 
        col_num[' [P_LAST]'], 
        col_num[' [UNDERLYING_LAST]'], 
        col_num[' [STRIKE]'] 
    ]

    spy_call_df = spy_df.iloc[:, call_columns]
    spy_put_df = spy_df.iloc[:, put_columns]

    ## 2. Noise removal

    ### Drop the following values
    ### - Missing IV values
    ### - Out of the money options
    ### - Expired options

    cleaned_spy_call_df = spy_call_df.dropna(
        axis = 0, 
        how = 'any', 
        subset = [' [C_IV]'] 
    )

    cleaned_spy_put_df = spy_put_df.dropna(
        axis = 0, 
        how = 'any', 
        subset = [' [P_IV]'] 
    )

    cleaned_spy_call_df = cleaned_spy_call_df[cleaned_spy_call_df[' [STRIKE]'] < cleaned_spy_call_df[' [UNDERLYING_LAST]']]
    cleaned_spy_put_df = cleaned_spy_put_df[cleaned_spy_put_df[' [STRIKE]'] > cleaned_spy_put_df[' [UNDERLYING_LAST]']]

    cleaned_spy_call_df = cleaned_spy_call_df[cleaned_spy_call_df[' [DTE]'] != 0.0]
    cleaned_spy_put_df = cleaned_spy_put_df[cleaned_spy_put_df[' [DTE]'] != 0.0]

    ## 3. Pruning (based on SP -- OP ratio)

    cleaned_spy_call_df = cleaned_spy_call_df[cleaned_spy_call_df[' [C_LAST]'] != 0.0]
    cleaned_spy_put_df = cleaned_spy_put_df[cleaned_spy_put_df[' [P_LAST]'] != 0.0]

    cleaned_spy_call_df[' [SP-OP]'] = cleaned_spy_call_df[' [UNDERLYING_LAST]'] / cleaned_spy_call_df[' [C_LAST]']
    cleaned_spy_put_df[' [SP-OP]'] = cleaned_spy_put_df[' [UNDERLYING_LAST]'] / cleaned_spy_put_df[' [P_LAST]']

    Q1_call = cleaned_spy_call_df[' [SP-OP]'].quantile(0.25)
    Q3_call = cleaned_spy_call_df[' [SP-OP]'].quantile(0.75)
    IQR_call = Q3_call - Q1_call
    cleaned_spy_call_df = cleaned_spy_call_df[cleaned_spy_call_df[' [SP-OP]'] >= Q1_call - 1.5 * IQR_call]
    cleaned_spy_call_df = cleaned_spy_call_df[cleaned_spy_call_df[' [SP-OP]'] <= Q3_call + 1.5 * IQR_call]

    Q1_put = cleaned_spy_put_df[' [SP-OP]'].quantile(0.25)
    Q3_put = cleaned_spy_put_df[' [SP-OP]'].quantile(0.75)
    IQR_put = Q3_put - Q1_put
    cleaned_spy_put_df = cleaned_spy_put_df[cleaned_spy_put_df[' [SP-OP]'] >= Q1_put - 1.5 * IQR_put]
    cleaned_spy_put_df = cleaned_spy_put_df[cleaned_spy_put_df[' [SP-OP]'] <= Q3_put + 1.5 * IQR_put]

    ## 3.1 Drop missing volume values

    cleaned_spy_call_df = cleaned_spy_call_df.dropna(
        axis = 0, 
        how = 'any', 
        subset = [' [C_VOLUME]'] 
    )

    cleaned_spy_put_df = cleaned_spy_put_df.dropna(
        axis = 0, 
        how = 'any', 
        subset = [' [P_VOLUME]'] 
    )

    ## 3.2 Convert string size values into integers so that we can pass them to a neural network

    def get_size1(size):
        size1 = 0

        for c in size:
            if(c == ' '):
                break

            size1 = size1 * 10 + int(c)

        return size1

    def get_size2(size):
        size2 = 0
        num_spaces = 0

        for c in size:
            if(num_spaces == 2):
                size2 = size2 * 10 + int(c)

            if(c == ' '):
                num_spaces += 1

        return size2

    cleaned_spy_call_df[' [C_SIZE1]'] = cleaned_spy_call_df[' [C_SIZE]'].apply(get_size1)
    cleaned_spy_call_df[' [C_SIZE2]'] = cleaned_spy_call_df[' [C_SIZE]'].apply(get_size2)
    cleaned_spy_call_df = cleaned_spy_call_df.drop(columns = [' [C_SIZE]'])
    cleaned_spy_call_df = cleaned_spy_call_df.astype(dtype = {' [C_SIZE1]': 'uint32', ' [C_SIZE2]': 'uint32'})

    cleaned_spy_put_df[' [P_SIZE1]'] = cleaned_spy_put_df[' [P_SIZE]'].apply(get_size1)
    cleaned_spy_put_df[' [P_SIZE2]'] = cleaned_spy_put_df[' [P_SIZE]'].apply(get_size2)
    cleaned_spy_put_df = cleaned_spy_put_df.drop(columns = [' [P_SIZE]'])
    cleaned_spy_put_df = cleaned_spy_put_df.astype(dtype = {' [P_SIZE1]': 'uint32', ' [P_SIZE2]': 'uint32'})

    ## 3.3 Sort data into subsets with same DTE by increasing quote date

    cleaned_spy_call_df = cleaned_spy_call_df.sort_values(by = [' [DTE]', '[QUOTE_UNIXTIME]'])
    cleaned_spy_put_df = cleaned_spy_put_df.sort_values(by = [' [DTE]', '[QUOTE_UNIXTIME]'])

    ## 4. Normalisation (min-max scaling)

    scaler = MinMaxScaler()

    call_columns = [
        ' [DTE]', 
        ' [C_DELTA]', 
        ' [C_GAMMA]', 
        ' [C_VEGA]', 
        ' [C_THETA]', 
        ' [C_RHO]', 
        ' [C_IV]', 
        ' [C_VOLUME]', 
        ' [C_BID]', 
        ' [C_ASK]', 
        ' [C_LAST]', 
        ' [UNDERLYING_LAST]', 
        ' [STRIKE]', 
        ' [SP-OP]', 
        ' [C_SIZE1]', 
        ' [C_SIZE2]' 
    ]

    put_columns = [
        ' [DTE]', 
        ' [P_DELTA]', 
        ' [P_GAMMA]', 
        ' [P_VEGA]', 
        ' [P_THETA]', 
        ' [P_RHO]', 
        ' [P_IV]', 
        ' [P_VOLUME]', 
        ' [P_BID]', 
        ' [P_ASK]', 
        ' [P_LAST]', 
        ' [UNDERLYING_LAST]', 
        ' [STRIKE]', 
        ' [SP-OP]', 
        ' [P_SIZE1]', 
        ' [P_SIZE2]' 
    ]

    cleaned_spy_call_df[call_columns] = scaler.fit_transform(cleaned_spy_call_df[call_columns])
    cleaned_spy_put_df[put_columns] = scaler.fit_transform(cleaned_spy_put_df[put_columns])

    ## 5. Dimensionality reduction



    ## Plot ROM of features

    ### Removes time columns

    # min_values = cleaned_spy_call_df.iloc[:, 2:].min(numeric_only = True)
    # max_values = cleaned_spy_call_df.iloc[:, 2:].max(numeric_only = True)

    # fig, ax = plt.subplots(figsize = (10, 6))
    # ax.bar(min_values.index, max_values - min_values)
    # ax.set_ylabel('Range of Motion')
    # ax.set_title('Range of Motion for Each Feature')
    # plt.xticks(rotation = 90)
    # plt.show()



    ## Final dataframe size -- call: (1247762, ), put: (747174, )

    return cleaned_spy_call_df, cleaned_spy_put_df



def produce_cleaned_call_dataframe(file_path : str) -> pd.DataFrame:
    '''
    Wraps the entire dataset preprocessing pipeline from reading to cleaning.

    Parameters
    ----------
    file_path : str
        The file path to the dataset.

    Returns
    -------
    cleaned_spy_call_df : pd.DataFrame
        The cleaned up version of the historical call option pricing data pandas dataframe.
    '''

    print()
    print('----------------------------------------------------------------- Loading dataset -----------------------------------------------------------------')
    print()

    print('Loading large dataset. This may take a few minutes...')
    start = time.time()
    spy_df = read_dataset(file_path)
    end = time.time()
    print(f'Dataset loaded successfully. Time elapsed: {(end - start) // 60:.0f} minutes and {(end - start) % 60:.2f} seconds.')

    print()
    print('----------------------------------------------------------------- Loading dataset -----------------------------------------------------------------')
    print()



    print()
    print('---------------------------------------------------------------- Dataframe summary ----------------------------------------------------------------')
    print()

    print(f'Shape: {spy_df.shape[0]} rowx x {spy_df.shape[1]} columns')
    print()

    print('Dataframe head:\n')
    print(spy_df.head(50))
    print()

    print('Column descriptions:\n')
    for column in spy_df.columns.to_list():
        print(f'{column:25}--   {description[column]}')

    # print()
    # for column in spy_df.columns.to_list():
    #     print(f'{column:25}--   {spy_df[column][0]}')

    print()
    print('---------------------------------------------------------------- Dataframe summary ----------------------------------------------------------------')
    print()



    print()
    print('---------------------------------------------------------------- Cleaned dataframe ----------------------------------------------------------------')
    print()

    cleaned_spy_call_df, cleaned_spy_put_df = preprocess_dataset(spy_df)

    print('Call options --')
    print()

    print(f'Shape: {cleaned_spy_call_df.shape[0]} rowx x {cleaned_spy_call_df.shape[1]} columns')
    print()

    print('Dataframe head:\n')
    print(cleaned_spy_call_df.head(50))
    print()

    print('Put options --')
    print()

    print(f'Shape: {cleaned_spy_put_df.shape[0]} rowx x {cleaned_spy_put_df.shape[1]} columns')
    print()

    print('Dataframe head:\n')
    print(cleaned_spy_put_df.head(50))

    print()
    print('---------------------------------------------------------------- Cleaned dataframe ----------------------------------------------------------------')
    print()

    return cleaned_spy_call_df



def prepare_training_data(cleaned_spy_call_df : pd.DataFrame) -> tuple[np.array, np.array, np.array, np.array, int]:
    '''
    Converts the historical call option pricing data pandas dataframe into numpy arrays of time-series image-like data.
    Splits up the pandas dataframe into training and testing data.

    Parameters
    ----------
    cleaned_spy_call_df : pd.DataFrame
        The cleaned up version of the historical call option pricing data pandas dataframe.

    Returns
    -------
    X_train : np.array
        Training features.
    y_train : np.array
        Training labels.
    X_test : np.array
        Testing features.
    y_test : np.array
        Testing labels.
    num_features : int
        Number of training features.
    '''

    call_feature_columns = [
        ' [DTE]', 
        ' [C_DELTA]', 
        ' [C_GAMMA]', 
        ' [C_VEGA]', 
        ' [C_THETA]', 
        ' [C_RHO]', 
        ' [C_IV]', 
        ' [C_VOLUME]', 
        ' [C_BID]', 
        ' [C_ASK]', 
        ' [UNDERLYING_LAST]', 
        ' [STRIKE]', 
        ' [SP-OP]', 
        ' [C_SIZE1]', 
        ' [C_SIZE2]' 
    ]

    call_label_columns = [
        ' [C_LAST]' 
    ]

    X = cleaned_spy_call_df.loc[:, call_feature_columns]
    y = cleaned_spy_call_df.loc[:, call_label_columns]

    ## Since we are working with time series data, we need to incorporate past label data as a feature

    X[' [C_LAST_LAG]'] = y.shift(TIME_SERIES_LAG_STRIDE)
    X[' [C_LAST_LAG]'][0: TIME_SERIES_LAG_STRIDE] = y[' [C_LAST]'][0: TIME_SERIES_LAG_STRIDE]

    num_features = len(X.columns)

    ## Reshape numpy array to become image-like data

    X = X.to_numpy()
    y = y.to_numpy()

    def create_sequences(X, y, TIME_SERIES_WINDOW_LENGTH):
        X_list, y_list = [], []
 
        for i in range(X.shape[0] - TIME_SERIES_WINDOW_LENGTH):
            ### Different DTEs

            if X[i, 0] != X[i + TIME_SERIES_WINDOW_LENGTH - 1, 0]:
                continue

            X_list.append(X[i: i + TIME_SERIES_WINDOW_LENGTH, :])
            y_list.append(y[i + TIME_SERIES_WINDOW_LENGTH - 1])

        return np.array(X_list), np.array(y_list)



    X, y = create_sequences(X, y, TIME_SERIES_WINDOW_LENGTH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    return X_train, X_test, y_train, y_test, num_features



if __name__ == '__main__':
    '''
    Main function.
    Sample usage: In terminal execute "python main.py <file_path>".

    Parameters
    ----------
    argv[1] : str
        The file path to the dataset.
    '''

    file_path = str(sys.argv[1])



    ## Training data

    cleaned_spy_call_df = produce_cleaned_call_dataframe(file_path)

    assert TIME_SERIES_WINDOW_LENGTH >= 1
    assert TIME_SERIES_LAG_STRIDE >= 1

    X_train, X_test, y_train, y_test, num_features = prepare_training_data(cleaned_spy_call_df)



    ## Training parameters

    learning_rate = 0.001
    epochs = 200
    batch_size = 128



    ## Actual cnn-lstm model

    model = cnnLSTM(num_features, TIME_SERIES_WINDOW_LENGTH)
    model.train(learning_rate, epochs, batch_size, X_train, y_train)
    predictions = model.eval(X_test, y_test)

    # print(predictions.shape, y_test.shape)

    ## Reconstruct predictions (undo min-max)