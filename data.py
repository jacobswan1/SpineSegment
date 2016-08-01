from __future__ import print_function

import cv2
import numpy as np
import os
import dicom
import copy
from keras.utils import np_utils

DATA_PATH =  os.getcwd()
MODE_TRAIN_PATH = '/data/metROI/train_txt'
MODE_TEST_PATH = '/data/metROI/test_txt'
MODE_SPINE_PATH = '/data/spineRegionROI/txt'
IMG_ROWS = 320
IMG_COLS = 320
RECEP_WEI = 28
RECEP_HEI = 56
nb_classes = 2


def get_input_and_output(source,spine_source):
    file = open(source, 'r')
    list = file.readlines()
    test_array = np.empty((0,len(list)), int)
    for line in list:
        #print line
        line = line.split(',')
        line[len(line)-1] = line[len(line)-1].replace('\n','')
        line = map(int, line)
        test_array = np.append(test_array,np.array([line]),axis = 0)
    
    file2 = open(spine_source, 'r')
    list2 = file2.readlines()
    test_array2 = np.empty((0,len(list2)), int)
    for line in list2:
        #print line
        line = line.split(',')
        line[len(line)-1] = line[len(line)-1].replace('\n','')
        line = map(int, line)
        test_array2 = np.append(test_array2,np.array([line]),axis = 0)
        
    dir_array = source.split('/')
    dir_name = dir_array[len(dir_array)-1].split('_')
    dir_image_file = DATA_PATH+'/data/Sagittal-segmentation/'+dir_name[0]+'/'+dir_name[1]+'/'+str(dir_name[2]).replace('txt','dcm')
    medical_img=dicom.read_file(dir_image_file)
    return medical_img.pixel_array , test_array, test_array2

def generate_data(MODE):
    nb_classes = 2
    RECEP_HEI = 56
    RECEP_WEI = 28
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
    x_max = IMG_ROWS/2

    if MODE == 'train':
        TRAIN_DATA_TXT_PATH = DATA_PATH + MODE_TRAIN_PATH
        TRAIN_SPINE_TXT_PATH = DATA_PATH + MODE_SPINE_PATH
    elif MODE == 'test':
        TRAIN_DATA_TXT_PATH = DATA_PATH + MODE_TEST_PATH
        TRAIN_SPINE_TXT_PATH = DATA_PATH + MODE_SPINE_PATH
    IMG_LIST = os.listdir(TRAIN_DATA_TXT_PATH)
    IMG_LIST = os.listdir(TRAIN_DATA_TXT_PATH)
    TRAIN_NUM = len(IMG_LIST)
    TRAIN_OUT = []
    TRAIN_IN = []
    spine_region = []
    for name in IMG_LIST:
            med_img , test_array ,spine_region = get_input_and_output(TRAIN_DATA_TXT_PATH + '/' + name, TRAIN_SPINE_TXT_PATH + '/' + name)
            test_array = test_array + 0.01
            spine_region = spine_region + 0.01
            med_img = cv2.resize(med_img, (IMG_ROWS,IMG_COLS), interpolation = cv2.INTER_AREA)
            test_array = cv2.resize(test_array, (IMG_ROWS,IMG_COLS), interpolation = cv2.INTER_AREA)
            spine_region = cv2.resize(spine_region, (IMG_ROWS,IMG_COLS), interpolation = cv2.INTER_AREA)
            cropImg1 = med_img[0:IMG_ROWS, IMG_COLS/4 : 3*IMG_ROWS/4 ]
            cropImg2 = test_array[0:IMG_ROWS, IMG_COLS/4 : 3*IMG_ROWS/4 ]
            cropImg3 = spine_region[0:IMG_ROWS, IMG_COLS/4 : 3*IMG_ROWS/4 ]
            sum_pix = cropImg1.shape[0] * cropImg1.shape[1]
            spine_pix = 0
            pos_pix = 0
            count_inside = 0
            count_outside = 0
            for line in cropImg3:
                for ele in line:
                    if(ele == 1.01):
                        spine_pix += 1
            for line in cropImg2:
                for ele in line:
                    if(ele == 1.01):
                        pos_pix += 1

            inside_num = (1 - ratio_of_outside) * pos_pix
            outside_num = ratio_of_outside * pos_pix
            inside_len = (int(round(spine_pix / inside_num)))
            outside_len = int(round((sum_pix-spine_pix) / outside_num))

            for i in range(y_start, y_max):
                for m in range(x_start, x_max):
                    class_lable = cropImg2[i- RECEP_HEI/2][m- RECEP_WEI/2]
                    inside_lable = cropImg3[i- RECEP_HEI/2][m- RECEP_WEI/2]    
                    if (class_lable == 1.01):
                        region = cropImg1[i- RECEP_HEI: i, m- RECEP_WEI: m]
                        TRAIN_IN.append(region.reshape(1,RECEP_HEI, RECEP_WEI))
                        TRAIN_OUT.append(np_utils.to_categorical([1], nb_classes)[0])
                        count_pos += 1
                        index += 1
                        #print(index)
                        
                    elif(inside_lable == 1.01 and class_lable != 1.01 ):
                        count_inside +=1
                        if(count_inside % inside_len == 0 ):
                            region = cropImg1[i- RECEP_HEI: i, m- RECEP_WEI: m]
                            TRAIN_IN.append(region.reshape(1,RECEP_HEI, RECEP_WEI))
                            TRAIN_OUT.append(np_utils.to_categorical([1], nb_classes)[0])
                            index += 1
                        
                    elif(inside_lable != 1.01 and class_lable != 1.01):
                        count_outside += 1
                        if(count_outside % outside_len == 0):
                            region = cropImg1[i- RECEP_HEI: i, m- RECEP_WEI: m]
                            TRAIN_IN.append(region.reshape(1,RECEP_HEI, RECEP_WEI))
                            TRAIN_OUT.append(np_utils.to_categorical([1], nb_classes)[0])
                            index += 1
                    count += 1
            if(index%1000 == 0):
                    print('Processed training set:'+str(index))

            count_img += 1
            print('Processed images:'+str(count_img)+'.Positive num:'+str(count_pos)+'.Total size:'+str(index))
    TRAIN_IN = np.asarray(TRAIN_IN, dtype=np.float)
    TRAIN_OUT = np.asarray(TRAIN_OUT, dtype=np.float)


    print('Saving '+ MODE +' data binary files.')
    np.save('X_42'+ MODE +'.npy', TRAIN_IN)
    np.save('Y_42'+ MODE +'.npy', TRAIN_OUT)
    print('Transfered '+ MODE +' data into binary file.')
    print('data set:'+str(index))

def load_binary_data(MODE):
    imgs_train = np.load('X_42'+ MODE +'.npy')
    imgs_out_train = np.load('Y_42'+ MODE +'.npy')
    return imgs_train, imgs_out_train



if __name__ == '__main__':
    #generate_data('test')
    generate_data('train')
