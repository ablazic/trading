import pandas as pd
import numpy as np

np.random.seed(10)

from keras.models import load_model
def zerolistmaker2(n):
    listofzeros = [0,0] * n
    return listofzeros

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def denormalize(df, elements):
    for i in range(0, len(df)):
        df[i] = (df[i] + 1) * elements[i]
    return df


def next_prediction(df, model, step, window):
    window_size = window
    no_of_sequences = 8
    window_strides = 1
    pred_step = step

    # Convert dataframe to windows for predictions ie days of data to be considered for every prediction
    openp = df.ix[:, 'Open'].tolist()
    highp = df.ix[:, 'High'].tolist()
    lowp = df.ix[:, 'Low'].tolist()
    closep = df.ix[:, 'Close'].tolist()
    volumep = df.ix[:, 'Volume (BTC)'].tolist()
    activity = df.ix[:, 'activity'].tolist()
    polarity = df.ix[:, 'polarity'].tolist()
    subjectivity = df.ix[:, 'subjectivity'].tolist()
    # hourp = df.ix[:, 'hour'].tolist()
    X, Y = [], []

    # Create window_sizes
    # Create window_sizes
    for i in range(0, len(df)-window_size+1, window_strides):
        try:
            o = openp[i:i + window_size]
            h = highp[i:i + window_size]
            l = lowp[i:i + window_size]
            c = closep[i:i + window_size]
            v = volumep[i:i + window_size]
            a = activity[i:i + window_size]
            p = polarity[i:i + window_size]
            s = subjectivity[i:i + window_size]
            # ho = hourp[i:i + window_size]

            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)
            a = (np.array(a) - np.mean(a)) / np.std(a)
            p = (np.array(p) - np.mean(p)) / np.std(p)
            s = (np.array(s) - np.mean(s)) / np.std(s)
            # ho = (np.array(ho)) / (np.array(ho)[0] + 1) - 1

            # x_i = closep[i:i + window_size]

            # x_i = close[i:i + window_size]
            # y_i = close[i + window_size + pred_step - 1]

            # last_close = x_i[-1]
            # close = x_i[-1]
            # next_close = y_i

            # if last_close < next_close:
            #     y_i = [1, 0]
            # else:
            #     y_i = [0, 1]

            x_i = np.column_stack((o, h, l, c, v, a, p, s))

        except Exception as e:
            break

        X.append(x_i)
        # Y.append(y_i)

    # Prepare dataset for prediction
    X = np.array(X)

    # load trained model
    pred = model.predict(X)
    pred = (pred == pred.max(axis=1)[:, None]).astype(int)
    return pred