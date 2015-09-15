
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pylab as plt
from pylab import *
import numpy as np
import random
import sklearn as sk
import itertools
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import csv
from sklearn import preprocessing, datasets, svm, cross_validation, linear_model
data_directory = '.'
def read_csv(p, train_test):
    """read csv input file and create array for pixel information and digit"""
    df = pd.read_csv('train.csv',header=0)
    digits= df['label'].values
    pixels=df.drop('label', axis=1).values
    return (pixels, digits)

def view_image_q1(image, label=""):
    """use pyplot imshow to convert matrix of pixels into image """
    """View a single image.
    Code reference: http://martin-thoma.com/classify-mnist-with-pybrain/"""
    #print("Label: %s" % label)
    figure()
    imshow(image, cmap=cm.gray)
    savefig(data_directory +'image'+label+'.png')
    close()

def view_images(image, label="", arr_index=0):
    """use pyplot imshow to convert matrix of pixels into image """
    """View a single image.
    Code reference: http://martin-thoma.com/classify-mnist-with-pybrain/"""
    #print("Label: %s" % label)
    figure()
    imshow(image, cmap=cm.gray)
    savefig(data_directory +"/"+label+'/image_'+str(arr_index)+"_"+label+'.png')
    close()

def read_digits_one(digit, pixel_digits):
    """convert numpy array of 784 elements into a 28 by 28 matrix to plot"""
    pixel_matrix = np.reshape(pixel_digits, (28, 28))
    view_image_q1(pixel_matrix, str(digit))
    
def read_digits(digit, pixel_digits,pixel_index ):
    """convert numpy array of 784 elements into a 28 by 28 matrix to plot"""
    #print
    pixel_matrix = np.reshape(pixel_digits, (28, 28))
    #print(pixel_matrix)
    #view_images(pixel_matrix, str(digit), pixel_index)

def hist_digits(list_digits):
    hist_bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #print("saving histogram")
    plt.figure()
    plt.hist(list_digits, normed=True)
    plt.xlabel('digit')
    plt.ylabel('prior probability')
    plt.title('histogram of normalized digit counts')
    plt.savefig(data_directory +"histogram_digits.png")
    plt.close()
    
def euclidian_dist(list1, list2):
    distmatrix = sk.metrics.pairwise.euclidean_distances(list1, list2)
    return distmatrix

def calc_L2_dist(list1, list2):
    cum_msd=0
    for i in range(len(list1)):
        msd = np.square(list1[i] - list2[i])
        cum_msd = cum_msd + msd
    rmsd = np.sqrt(cum_msd)
    return rmsd
    
    

def hist_genuine_imposter(list_zero, list_one):
    distances = []
    genuine_L2=[]
    imposter_L2=[]
    matches = []
    list_combinations_zero = itertools.combinations(list_zero, 2)
    for item in list_combinations_zero:
        L2dist = euclidian_dist(pixels[item[0]], pixels[item[1]])
        distances.append(1/float(L2dist[0]))
        genuine_L2.append(float(L2dist[0]))
        matches.append(1)
    list_combinations_one = itertools.combinations(list_one, 2)
    for item in list_combinations_one:
        L2dist = euclidian_dist(pixels[item[0]], pixels[item[1]])
        
        distances.append(1/float(L2dist[0]))
        genuine_L2.append(float(L2dist[0]))
        matches.append(1)
    for myindex0 in list_zero:
        for myindex1 in list_one:
            L2dist = euclidian_dist(pixels[myindex0], pixels[myindex1])
            distances.append(1/float(L2dist[0]))
            imposter_L2.append(float(L2dist[0]))
            matches.append(0)
  
    #print(genuine_L2, imposter_L2)
    fpr, tpr, thresholds = sk.metrics.roc_curve(matches, distances)
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operating Characterestics curve')
    plt.savefig(data_directory +"roc_curve.png")
    plt.close()
    return genuine_L2, imposter_L2

def KNearest_neighbor(all_digits, all_pixels):
    X = all_pixels
    Y = np.array(all_digits)
    for train, test in cross_validation.KFold(len(X), 3):
        
        Xtrain = X[train]
        Ytrain = Y[train]
        cls = sk.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
        cls.fit(Xtrain, Ytrain)

        print("accuracy : " +str(cls.score(X[test], Y[test])))
        ##confusion matrix
        confusionmat = sk.metrics.confusion_matrix(Y[test], cls.predict(X[test]))
        print(confusionmat)
        error_by_digit = {}
        for i in range(len(confusionmat)):
            error = 0
            for j in range(len(confusionmat)):
                if(i<>j):
                    error = error + confusionmat[i,j]
            error_by_digit[i] = error
        #print(error_by_digit)
        print("digit with max error rate: "+str( max(error_by_digit, key=error_by_digit.get)))

def KNearest_neighbor_test(all_digits, all_pixels, test_pixels):

        
        Xtrain = all_pixels
        Ytrain = np.array(all_digits)
        cls = sk.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')
        cls.fit(Xtrain, Ytrain)
        print(cls.predict(test_pixels))
    
def plot_histograms(input1, input2):
    plt.figure()
    plt.hist(input1,histtype='bar', alpha=0.6, color='b')
    plt.hist(input2, histtype='bar', alpha=0.6, color='r')
    plt.savefig(data_directory +"hist_real_imposter.png")
    plt.close()
    
if __name__ == "__main__":
    
    #define number of columns
    p = 784
    print("reading csv\n")

    (pixels, digits) = read_csv(p, "train")
    print("plotting histogram of digits\n")
    hist_digits(digits)
    dictSampleDigit={}
    from collections import defaultdict
    defdict_digits = defaultdict(list)
    for i in range(len(digits)):
        dictSampleDigit[digits[i]] = i
        defdict_digits[digits[i]].append(i)
        
    #print(dictSampleDigit)
    print("printing one sample digit of each type\n")
    for mydigit in dictSampleDigit.keys():
        print(pixels[dictSampleDigit[mydigit]])
        read_digits_one(mydigit, pixels[dictSampleDigit[mydigit]])
    '''for mydigit in defdict_digits.iterkeys():
        for mypixelindex in defdict_digits[mydigit]:
            #print(mydigit, mypixelindex)
            read_digits(mydigit, pixels[mypixelindex], mypixelindex)'''
    
    print("printing nearest neighbor for randomly selected instance of each digit\n")
    for mydigitkey in defdict_digits.keys():
        mydigit = random.choice(defdict_digits[mydigitkey])
        #print(mydigit, digits[mydigit])
        dict_L2dist = {}
        for mypixelindex in range(len(pixels)):
            if mydigit <> mypixelindex:
                #print(mydigit, dictSampleDigit[mydigit], pixels[mypixelindex])
                L2dist = euclidian_dist(pixels[mypixelindex], pixels[mydigit])
                L2dist_rmsd = calc_L2_dist(pixels[mypixelindex], pixels[mydigit])
        
                dict_L2dist[mypixelindex]=L2dist[0]
        min_rmsd=min(dict_L2dist, key=dict_L2dist.get)
        #print(euclidian_dist(pixels[min_rmsd],pixels[mydigit] ))
        match_correct=''
        if(digits[mydigit]<>digits[min_rmsd]):
            match_correct='*'
        
        print(digits[mydigit],digits[min_rmsd], list(dict_L2dist[min_rmsd])[0], match_correct )
                
    list_zeroes = defdict_digits[0]
    list_ones = defdict_digits[1]
    print("computing real and imposter matches for zero and one digits only\n")
    real, imposter = hist_genuine_imposter(list_zeroes, list_ones)
    #print(real, imposter)
    print("plotting histogram of real and imposter matches\n")
    plot_histograms(real, imposter)
    
    
    print("performing K-nearest neighbor classification on all data\n")
    KNearest_neighbor(digits, pixels)
    print("Reading test")
    (test_pixels, empty_digits) = read_csv(p, "test")
    KNearest_neighbor_test(digits, pixels, test_pixels)


# In[ ]:




# In[ ]:



