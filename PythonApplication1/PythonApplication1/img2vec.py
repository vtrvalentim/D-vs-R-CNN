import cv2 as cv2
import glob as glob
import numpy as np

def DataShuffle(data,label):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(label)

def img2input():

    #declaring empty list
    X_train_temp = []
    
    #creating file path lists
    print("Loading data path...")
    D_train_files = glob.glob("../DATA SET/Train/D/*.png")
    D_dev_files = glob.glob("../DATA SET/Dev/D/*.png")
    D_test_files = glob.glob("../DATA SET/Test/D/*.png")

    R_train_files = glob.glob("../DATA SET/Train/R/*.png")
    R_dev_files = glob.glob("../DATA SET/Dev/R/*.png")
    R_test_files = glob.glob("../DATA SET/Test/R/*.png")

    #creating numpy array from file path lists   
    #D class training set
    print("Loading D training data...")
    for file in D_train_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    D_train = np.array(X_train_temp)
    X_train_temp.clear()

    #D class dev set
    print("Loading D dev data...")
    for file in D_dev_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    D_dev = np.array(X_train_temp)
    X_train_temp.clear()

    #D class test set
    print("Loading D test data...")
    for file in D_test_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    D_test = np.array(X_train_temp)
    X_train_temp.clear()

    #R class training set
    print("Loading R training data...")
    for file in R_train_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    R_train = np.array(X_train_temp)
    X_train_temp.clear()

    #R class dev set
    print("Loading R dev data...")
    for file in R_dev_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    R_dev = np.array(X_train_temp)
    X_train_temp.clear()

    #R class test set
    print("Loading R test data...")
    for file in R_test_files:
        image = cv2.imread(file,1)
        X_train_temp.append(image)
    R_test = np.array(X_train_temp)
    X_train_temp.clear()

    #creating label vectors
    #D class: label = 0 
    #R class: label = 1    
    train_label = np.concatenate((np.zeros(D_train.shape[0]),np.ones(R_train.shape[0]))).reshape((D_train.shape[0]+R_train.shape[0],1))
    dev_label = np.concatenate((np.zeros(D_dev.shape[0]),np.ones(R_dev.shape[0]))).reshape((D_dev.shape[0]+R_dev.shape[0],1))
    test_label = np.concatenate((np.zeros(D_test.shape[0]),np.ones(R_test.shape[0]))).reshape((D_test.shape[0]+R_test.shape[0],1))

    #concatenating input vectors
    X_train = np.concatenate((D_train,R_train))
    X_dev = np.concatenate((D_dev,R_dev))
    X_test = np.concatenate((D_test,R_test))  
    
    #Unison shuffling of X vectors and respective labels
    DataShuffle(X_train,train_label)
    DataShuffle(X_dev,dev_label)
    DataShuffle(X_test,test_label)

    return X_train, X_dev, X_test, train_label, dev_label, test_label
