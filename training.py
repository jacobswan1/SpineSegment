from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import data

batch_size = 128
nb_classes = 2
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28，56
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3



print('Loading Data...')

X_train, Y_train = data.load_binary_data('train')
X_test, Y_test = data.load_binary_data('test')
#X_train = np.random.permutation(X_train)
#Y_train = np.random.permutation(Y_train)
#X_test = np.random.permutation(X_test)
#Y_test = np.random.permutation(Y_test)


print('Normalizing training data...')
for i in range(X_train.shape[0]):
    for m in range(X_train.shape[1]):
        mean = np.mean(X_train[i][m])  # mean for data centering
        std = np.std(X_train[i][m])  # std for data normalization
        X_train[i][m] = (X_train[i][m] - mean)/std
        
#print('Normalizing testing data...')
#for i in range(X_test.shape[0]):
#    for m in range(X_test.shape[1]):
#        mean = np.mean(X_test[i][m])  # mean for data centering
#        std = np.std(X_test[i][m])  # std for data normalization
#	X_test[i][m] = (X_test[i][m] -mean)/std

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=( 1 , img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, shuffle = True ,validation_split=0.25)

model.save_weights('weights.hdf5', overwrite=True)
#predicted_result = model.predict_classes(X_test,verbose = 1)
#np.save('result.npy', predicted_result)

print('Model trained, saving weights and ready to prediction.')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

