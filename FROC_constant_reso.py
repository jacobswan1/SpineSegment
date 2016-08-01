import cv2
import numpy as np
import os
import dicom
import data
import copy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy import ndimage
from keras.optimizers import SGD
from keras.utils import np_utils
from skimage.morphology import remove_small_objects
from scipy import ndimage
from sklearn import metrics, metrics
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_erosion, binary_dilation


def getPara(predict, true, threshold, resolution, windowsize):
    (TP, FP, TN, FN, class_lable) = perf_measure(true, predict, threshold)
    if((TP + FN) == 0):
        TPR = 0
    else:
        TPR = np.float(TP) / (TP + FN)

    class_lable = class_lable.astype(bool).reshape(264,  132)
    true = true.astype(bool).reshape((264,  132))

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
    y_score = np.load('./resolution/predicted_image.npy')
    y_true = np.load('./resolution/answer_image.npy')
    reso = np.load('./resolution/resolution.npy')

    y_score = y_score[0:y_score.shape[0], 1]
    y_true = y_true[0:y_true.shape[0], 1]

    scores = []
    trues = []
    next_start = 0
    for i in range(len(reso)):
        ysize = 264
        xsize = 132
        scores.append(y_score[next_start: next_start + np.int(ysize * xsize)])
        trues.append(y_true[next_start: next_start + np.int(ysize * xsize)])
        next_start = np.int(next_start + ysize * xsize)


    y_score = np.asarray(scores)
    y_true = np.asarray(trues)
    set_size = 22

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
    #delete = [6, 7, 8, 19, 20,21]

    delete = [6,7,8,9,10,11,12,13,19,20,21]
    for t in range(1, thresholds.size):
        tpr_sum = 0
        fp_sum = 0
        for i in range(set_size):
            if i not in delete:
                TPR, FP_num = getPara(y_score[i], y_true[i].astype(np.int),  thresholds[t], reso[i], windowsize)
                tpr_sum += TPR
                fp_sum += FP_num
                print(TPR, FP_num)
        print(t, thresholds.size, thresholds[t], fp_sum / np.float(set_size - len(delete)), tpr_sum / np.float(set_size - len(delete)))
        TPR_list.append(tpr_sum / np.float(set_size - len(delete)))
        FP_num_list.append(fp_sum / np.float(set_size - len(delete)))
        #print( t, thresholds.size, fp_sum/np.float(set_size ), tpr_sum/np.float(set_size ))
        #TPR_list.append(tpr_sum/np.float(set_size ))
        #FP_num_list.append(fp_sum/np.float(set_size ))
    # np.save('./resolution/TPR_list_'+ str(windowsize) +'.npy', TPR_list)
    # np.save('./resolution/FP_num_list_'+ str(windowsize) +'.npy', FP_num_list)
    np.save('./resolution/TPR_list_'+ 'special' +'.npy', TPR_list)
    np.save('./resolution/FP_num_list_'+ 'special' +'.npy', FP_num_list)

if __name__ == '__main__':
    generate_plotnum(500)


