import pandas as pd
import numpy as np
np.random.seed(1332)

from keras.models import Sequential, Model
from keras.optimizers import Nadam, Adam, RMSprop, SGD
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization, Dense, Activation, Flatten, Convolution1D, Dropout, concatenate, Conv1D, MaxPool1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# To shuffle windows (not the content in the window) along with y values
def shuffle(a, b):
    s_a = np.empty(a.shape, dtype=a.dtype)
    s_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.RandomState(seed=121).permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        s_a[new_index] = a[old_index]
        s_b[new_index] = b[old_index]
    return s_a, s_b

# For training create dataset from a stored csv created using data creation file
def create_dataset(X, y, split=0.8):
    split_index = int(len(X) * split)
    X_train = X[0:split_index]
    Y_train = y[0:split_index]
    X_train, Y_train = shuffle(X_train, Y_train)

    X_test = X[split_index:]
    Y_test = y[split_index:]
    return X_train, X_test, Y_train, Y_test

df = pd.read_csv("/home/gaurav/PycharmProjects/AI-Tensorflow/data/final_fulldata_next_hour_2.csv")[::-1]
df = df[3000:]
print(df.shape)

# From API
openp = df.ix[:, 'Open'].tolist()
highp = df.ix[:, 'High'].tolist()
lowp = df.ix[:, 'Low'].tolist()
closep = df.ix[:, 'Close'].tolist()
volumep = df.ix[:, 'Volume (BTC)'].tolist()
hourp = df.ix[:, 'hour'].tolist()

# From news
activity = df.ix[:, 'activity'].tolist()
polarity = df.ix[:, 'polarity'].tolist()
subjectivity = df.ix[:, 'subjectivity'].tolist()

# Hyperparameters
window_size = 10 # no. of values in each window
no_of_sequences = 8
window_strides = 1
pred_step = 1 # after 24 hrs

X, Y = [], []

# Create window_sizes
# Create window of size defined by window_size
for i in range(0, len(df), window_strides):
    try:
        o = openp[i:i+window_size]
        h = highp[i:i+window_size]
        l = lowp[i:i+window_size]
        c = closep[i:i+window_size]
        v = volumep[i:i+window_size]
        a = activity[i:i+window_size]
        p = polarity[i:i+window_size]
        s = subjectivity[i:i + window_size]

        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)
        a = (np.array(a) - np.mean(a)) / np.std(a)
        p = (np.array(p) - np.mean(p)) / np.std(p)
        s = (np.array(s) - np.mean(s)) / np.std(s)


        x_i = closep[i:i+window_size]
        y_i = closep[i+window_size+pred_step-1]

        last_close = x_i[-1]
        close = x_i[-1]
        next_close = y_i

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1]

        x_i = np.column_stack((o, h, l, c, v, a, p, s))

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)
print(len(X))
X_train, X_test, Y_train, Y_test = create_dataset(X, Y)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], no_of_sequences))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_sequences))



model = Sequential()
model.add(Convolution1D(input_shape = (window_size, no_of_sequences),
                        nb_filter=16,#16
                        filter_length=1,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.4))

model.add(Convolution1D(nb_filter=8,#8
                        filter_length=1,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64)) #64
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(32)) #32
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation('softmax'))
print(model.summary())
opt = Nadam(lr=0.001)
reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="Best_Model_NextHour.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit(X_train, Y_train, nb_epoch = 100, batch_size = 128, verbose=2, validation_split=0.05, callbacks=[checkpointer], shuffle=True)

model.load_weights("Best_Model_NextHour.hdf5")
pred = model.predict(np.array(X_test))
pred = (pred == pred.max(axis=1)[:,None]).astype(int)

from sklearn.metrics import classification_report

from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(Y_test, pred))
print(classification_report(Y_test, pred))

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
