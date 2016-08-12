from __future__ import print_function

import cv2
import numpy as np
import os
import dicom
import copy
from keras.utils import np_utils
from skimage.transform import rotate, warp_coords
from scipy.ndimage import map_coordinates
import random
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage.morphology import remove_small_objects
from scipy import ndimage
from sklearn import metrics, metrics
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_erosion, binary_dilation

DATA_PATH = os.getcwd()
MODE_TRAIN_PATH = '/data/metROI/train_txt'
MODE_TEST_PATH = '/data/metROI/test_txt'
MODE_SPINE_PATH = '/data/spineRegionROI/txt'
IMG_ROWS = 320
IMG_COLS = 320
RECEP_WEI = 30
RECEP_HEI = 70
nb_classes = 2
batch_size = 128
nb_classes = 2
nb_epoch = 1
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def get_input_and_output(source, spine_source):
    file = open(source, 'r')
    list = file.readlines()
    test_array = np.empty((0, len(list)), int)
    for line in list:
        # print line
        line = line.split(',')
        line[len(line) - 1] = line[len(line) - 1].replace('\n', '')
        line = map(int, line)
        test_array = np.append(test_array, np.array([line]), axis=0)

    file2 = open(spine_source, 'r')
    list2 = file2.readlines()
    test_array2 = np.empty((0, len(list2)), int)
    for line in list2:
        # print line
        line = line.split(',')
        line[len(line) - 1] = line[len(line) - 1].replace('\n', '')
        line = map(int, line)
        test_array2 = np.append(test_array2, np.array([line]), axis=0)

    dir_array = source.split('/')
    dir_name = dir_array[len(dir_array) - 1].split('_')
    dir_image_file = DATA_PATH + '/data/Sagittal-segmentation/' + \
        dir_name[0] + '/' + dir_name[1] + '/' + \
        str(dir_name[2]).replace('txt', 'dcm')
    medical_img = dicom.read_file(dir_image_file)
    return medical_img, test_array, test_array2

def getPara(predict, true, threshold, resolution, windowsize):
    (TP, FP, TN, FN, class_lable) = perf_measure(true, predict, threshold)
    if((TP + FN) == 0):
        TPR = 0
    else:
        TPR = np.float(TP) / (TP + FN)

    class_lable = class_lable.astype(bool).reshape(250,  130)
    true = true.astype(bool).reshape((250,  130))

    num = 2
    x = np.arange( -num , num+1, 1)
    xx, yy  = np.meshgrid( x, x )
    struc = (xx * xx + yy * yy)<= num * num
    class_lable = binary_dilation(class_lable, struc)
    class_lable = binary_erosion(class_lable, struc)

    # predict2 = remove_small_objects(class_lable, windowsize * resolution, in_place=False)
    predict2 = remove_small_objects(class_lable, windowsize, in_place=False)
    labeled_array1, num_features1 = label(predict2)
    labeled_array2, num_features2 = label(true)
    FP_num = num_features1 - num_features2
    if FP_num < 0:
        FP_num = 0
    return TPR, FP_num


def perf_measure(y_actual, predict, threshold):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    predict = transfer_prob(predict, threshold)
    for i in range(len(predict)):
        if y_actual[i] == predict[i] == 1:
            TP += 1
    for i in range(len(predict)):
        if y_actual[i] == 0 and y_actual[i] != predict[i]:
            FP += 1
    for i in range(len(predict)):
        if y_actual[i] == predict[i] == 0:
            TN += 1
    for i in range(len(predict)):
        if y_actual[i] == 1 and y_actual[i] != predict[i]:
            FN += 1

    return(TP, FP, TN, FN, predict)


def transfer_prob(y_score, threshold):
    y_result = []
    for i in range(len(y_score)):
        if y_score[i] >= threshold:
            y_result.append(1)
        else:
            y_result.append(0)
    return np.asarray(y_result)

def generate_plotnum(windowsize):
    y_score = np.load('predicted_image.npy')
    y_true = np.load('answer_image.npy')
    reso = np.load('resolution.npy')

    y_score = y_score[0:y_score.shape[0], 1]
    y_true = y_true[0:y_true.shape[0], 1]

    scores = []
    trues = []
    next_start = 0
    for i in range(len(reso)):
        ysize = 250
        xsize = 130
        scores.append(y_score[next_start: next_start + np.int(ysize * xsize)])
        trues.append(y_true[next_start: next_start + np.int(ysize * xsize)])
        next_start = np.int(next_start + ysize * xsize)


    y_score = np.asarray(scores)
    y_true = np.asarray(trues)
    set_size = y_score.shape[0]

    count = 1


    thresholds = []
    tmp = 0
    for m in range(1, 10, 1):
        tmp += 1
        # print(m/np.float(100))
        thresholds.append(m / np.float(100))
    for i in range(1, 10, 1):
        thresholds.append(i / np.float(10))
    for m in range(90, 100, 1):
        thresholds.append(m / np.float(100))
    tmp = 0
    for m in range(900, 1000, 1):
        tmp += 1
        if(tmp % 10 == 0):
            thresholds.append(m / np.float(1000))
    thresholds = sorted(thresholds, reverse=True)
    thresholds = np.asarray(thresholds)


    TPR_list = []
    FP_num_list = []

    for t in range(1, thresholds.size):
        tpr_sum = 0
        fp_sum = 0
        for i in range(set_size):
            TPR, FP_num = getPara(y_score[i], y_true[i].astype(np.int),  thresholds[t], reso[i], windowsize)
            tpr_sum += TPR
            fp_sum += FP_num
            print(TPR, FP_num)
        print(t, thresholds.size, thresholds[t], fp_sum / np.float(set_size ), tpr_sum / np.float(set_size ))
        TPR_list.append(tpr_sum / np.float(set_size ))
        FP_num_list.append(fp_sum / np.float(set_size ))
        #print( t, thresholds.size, fp_sum/np.float(set_size ), tpr_sum/np.float(set_size ))
        #TPR_list.append(tpr_sum/np.float(set_size ))
        #FP_num_list.append(fp_sum/np.float(set_size ))
    # np.save('./resolution/TPR_list_'+ str(windowsize) +'.npy', TPR_list)
    # np.save('./resolution/FP_num_list_'+ str(windowsize) +'.npy', FP_num_list)
    np.save('TPR_list_'+ str(windowsize) +'.npy', TPR_list)
    np.save('FP_num_list_'+ str(windowsize) +'.npy', FP_num_list)

nb_classes = 2
RECEP_HEI = 70
RECEP_WEI = 30
IMG_COLS = 320
IMG_ROWS = 320
ratio_of_outside = 0.2
count_img = 0
count = 0
index = 0
count_pos = 0
class_lable = 2
x_start = RECEP_WEI
y_start = RECEP_HEI
y_max = IMG_COLS
x_max = IMG_ROWS / 2

TRAIN_DATA_TXT_PATH = DATA_PATH + MODE_TRAIN_PATH
TRAIN_SPINE_TXT_PATH = DATA_PATH + MODE_SPINE_PATH

IMG_LIST = os.listdir(TRAIN_DATA_TXT_PATH)
TRAIN_NUM = len(IMG_LIST)

X = []
Y = []
TRAIN_IN = []
TRAIN_OUT = []

for name in IMG_LIST:
    med, test_array, spine_region = get_input_and_output(
        TRAIN_DATA_TXT_PATH + '/' + name, TRAIN_SPINE_TXT_PATH + '/' + name)
    med_img = med.pixel_array.astype('float')
    test_array = test_array.astype('float')
    spine_region = spine_region.astype('float')
    print(med_img.shape)

    X.append(med_img)
    Y.append([test_array, spine_region, med.PixelSpacing[0]])
    X.append(cv2.flip(med_img, 1))
    Y.append([cv2.flip(test_array, 1), cv2.flip(
        spine_region, 1), med.PixelSpacing[0]])
    X.append(cv2.flip(med_img, 0))
    Y.append([cv2.flip(test_array, 0), cv2.flip(
        spine_region, 0), med.PixelSpacing[0]])
    X.append(cv2.flip(med_img, -1))
    Y.append([cv2.flip(test_array, -1),
              cv2.flip(spine_region, -1), med.PixelSpacing[0]])

X = np.asarray(X)
Y = np.asarray(Y)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(
#     X, Y, test_size=0.25, random_state=42)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(
#      X, Y, test_size=0.25)
# Xtrain = X[np.int((1./4) * len(X)) : len(X)]
# Ytrain = Y[np.int((1./4) * len(X)) : len(Y)]
# Xtest = X[0: np.int((1./4) * len(X)) ]
# Ytest = Y[0: np.int((1./4) * len(Y)) ]
Xtrain = X
Ytrain = Y

# shuffle training data to get better performance



for i in range(Xtrain.shape[0]):
    med_img, test_array, spine_region = Xtrain[
        i], Ytrain[i][0], Ytrain[i][1]
    test_array = (test_array > 0.5).astype('int')
    spine_region = (spine_region > 0.5).astype('int')

    test_array = test_array + 0.001
    spine_region = spine_region + 0.001

    med_img = cv2.resize(med_img, (IMG_ROWS, IMG_COLS),
                         interpolation=cv2.INTER_AREA)
    test_array = cv2.resize(
        test_array, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)
    spine_region = cv2.resize(
        spine_region, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)

    cropImg1 = med_img[0:IMG_ROWS, IMG_COLS / 4: 3 * IMG_ROWS / 4]
    cropImg2 = test_array[0:IMG_ROWS, IMG_COLS / 4: 3 * IMG_ROWS / 4]
    cropImg3 = spine_region[0:IMG_ROWS, IMG_COLS / 4: 3 * IMG_ROWS / 4]
    sum_pix = cropImg1.shape[0] * cropImg1.shape[1]
    spine_pix = 0
    pos_pix = 0
    count_inside = 0
    count_outside = 0
    for line in cropImg3:
        for ele in line:
            if(ele > 0.02):
                spine_pix += 1
    for line in cropImg2:
        for ele in line:
            if(ele > 0.02):
                pos_pix += 1

    inside_num = (1 - ratio_of_outside) * pos_pix
    outside_num = ratio_of_outside * pos_pix
    inside_len = (int(round(spine_pix / inside_num)))
    outside_len = int(round((sum_pix - spine_pix) / outside_num))

    for i in range(y_start, y_max):
        for m in range(x_start, x_max):
            class_lable = cropImg2[i - RECEP_HEI / 2][m - RECEP_WEI / 2]
            inside_lable = cropImg3[i - RECEP_HEI / 2][m - RECEP_WEI / 2]
            if (class_lable > 0.02):
                region = cropImg1[i - RECEP_HEI: i, m - RECEP_WEI: m]
                TRAIN_IN.append(region.reshape(1, RECEP_HEI, RECEP_WEI))
                TRAIN_OUT.append(
                    np_utils.to_categorical([1], nb_classes)[0])
                count_pos += 1
                index += 1
                # print(index)

            elif(inside_lable > 0.02 and class_lable <= 0.02):
                count_inside += 1
                if(count_inside % inside_len == 0):
                    region = cropImg1[i - RECEP_HEI: i, m - RECEP_WEI: m]
                    TRAIN_IN.append(region.reshape(
                        1, RECEP_HEI, RECEP_WEI))
                    TRAIN_OUT.append(
                        np_utils.to_categorical([0], nb_classes)[0])
                    index += 1

            elif(inside_lable < 0.02 and class_lable <= 0.02):
                count_outside += 1
                if(count_outside % outside_len == 0):
                    region = cropImg1[i - RECEP_HEI: i, m - RECEP_WEI: m]
                    TRAIN_IN.append(region.reshape(
                        1, RECEP_HEI, RECEP_WEI))
                    TRAIN_OUT.append(
                        np_utils.to_categorical([0], nb_classes)[0])
                    index += 1
            count += 1
    if(index % 1000 == 0):
        print('Processed training set:' + str(index))

    count_img += 1
    print('Processed images:' + str(count_img) + '.Positive num:' +
          str(count_pos) + '.Total size:' + str(index))

X_train = np.asarray(TRAIN_IN, dtype=np.float)
Y_train = np.asarray(TRAIN_OUT, dtype=np.float)

print('Normalizing training data...')
for i in range(X_train.shape[0]):
    for m in range(X_train.shape[1]):
        mean = np.mean(X_train[i][m])  # mean for data centering
        std = np.std(X_train[i][m])  # std for data normalization
        X_train[i][m] = (X_train[i][m] - mean) / std

#print('Normalizing testing data...')
# for i in range(X_test.shape[0]):
#    for m in range(X_test.shape[1]):
#        mean = np.mean(X_test[i][m])  # mean for data centering
#        std = np.std(X_test[i][m])  # std for data normalization
#	X_test[i][m] = (X_test[i][m] -mean)/std

print('X_train shape:', X_train.shape)

#training and predicting part

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(1, RECEP_HEI, RECEP_WEI)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
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
          verbose=1, shuffle=True, validation_split=0.25)

model.save_weights('weights.hdf5', overwrite=True)
#predicted_result = model.predict_classes(X_test,verbose = 1)
#np.save('result.npy', predicted_result)

print('Model trained, saving weights and ready to prediction.')



#generate testing data

predict_train_in = []
predict_train_out = []
count_pos = 0
count_img = 0
index = 0
TEST_DATA_TXT_PATH = DATA_PATH + MODE_TEST_PATH
IMG_LIST = os.listdir(TEST_DATA_TXT_PATH)
Xtest = []
Ytest = []
Resolution = []

for name in IMG_LIST:
    med, test_array, spine_region = get_input_and_output(
        TEST_DATA_TXT_PATH + '/' + name, TRAIN_SPINE_TXT_PATH + '/' + name)
    med_img = med.pixel_array.astype('float')
    test_array = test_array.astype('float')
    print(med_img.shape)
    Xtest.append(med_img)
    Ytest.append([test_array, med.PixelSpacing[0]])
Xtest = np.asarray(Xtest)
Ytest = np.asarray(Ytest)
Resolution.append(med.PixelSpacing[0])
Resolution = np.asarray(Resolution)


for i in range(Xtest.shape[0]):

    med_img, test_array = Xtest[i], Ytest[i][0]
    med_img = cv2.resize(med_img, (320, 320),
                         interpolation=cv2.INTER_AREA)
    test_array = test_array + 0.001
    test_array = cv2.resize(
        test_array, (320, 320), interpolation=cv2.INTER_AREA)

    cropImg1 = med_img[0:320, 320 / 4: 3 * 320 / 4]
    cropImg2 = test_array[0:320, 320 / 4: 3 * 320 / 4]

    y_max = 320
    x_max = 320 / 2

    for i in range(y_start, y_max):
        for m in range(x_start, x_max):
            region = cropImg1[i - RECEP_HEI: i, m - RECEP_WEI: m]
            class_lable = cropImg2[
                i - RECEP_HEI / 2][m - RECEP_WEI / 2]
            if(class_lable <= 0.002):
                predict_train_in.append(region.reshape(
                    1, RECEP_HEI, RECEP_WEI))
                predict_train_out.append(
                    np_utils.to_categorical([0], nb_classes)[0])
                index += 1
                # print(index)

            else:
                predict_train_in.append(region.reshape(
                    1, RECEP_HEI, RECEP_WEI))
                predict_train_out.append(
                    np_utils.to_categorical([1], nb_classes)[0])
                count_pos += 1
                index += 1
                # print(index)

            count += 1
    if(index % 1000 == 0):
        print('Processed training set:' + str(index))

    count_img += 1
    print('Processed images:' + str(count_img) + '.Positive num:' +
          str(count_pos) + '.Total size:' + str(index))
predict_train_in = np.asarray(predict_train_in, dtype=np.float)
predict_train_out = np.asarray(predict_train_out, dtype=np.float)

print('Normalizing testing data...')
for i in range(predict_train_in.shape[0]):
    for m in range(predict_train_in.shape[1]):
        mean = np.mean(predict_train_in[i][m])  # mean for data centering
        std = np.std(predict_train_in[i][m])  # std for data normalization
        predict_train_in[i][m] = (predict_train_in[i][m] - mean) / std

print('Predicting...')

predict_result = model.predict_proba(predict_train_in)

np.save('answer_image' + '.npy', predict_train_out)
np.save('predicted_image' + '.npy', predict_result)
np.save('input_image'+'.npy',Xtest)
np.save('resolution' + '.npy', Resolution)

#evaluate part
generate_plotnum(300)
generate_plotnum(400)
generate_plotnum(500)
generate_plotnum(600)

