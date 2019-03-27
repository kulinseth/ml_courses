
# coding: utf-8

# In[1]:

#get_ipython().magic(u'pylab inline')


# In[ ]:

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as P

def display_digit(df):
    num = df.drop_duplicates('label')
    v = num.drop('label', axis=1).values
    r, _ = num.shape
    imgs = map(lambda x :np.reshape(x, (28,28)), v)
    fig = P.figure()
    for i in range(r):
        a = fig.add_subplot(2, r/2, i+1)
        P.imshow(imgs[i])
    P.savefig('numbers')
    P.close()
    return num.values

### This function computes minimum distance
### between 2 matrices passed between all pairs in those matrices
def l2norm(a1, a2):
    tp = np.empty((0,2))
    fp = np.empty((0,2))
    for i in xrange(a1.shape[0]):
        mindist = 0.0
        is_tp = 0
        for j in xrange(a2.shape[0]):
            dist = np.linalg.norm(a1[i][1:] - a2[j][1:])
            if ((i != j) and (dist < mindist)):
                mindist = dist
                if (a1[i][0] == a2[j][0]):
                    is_tp = 1
                else:
                    is_tp = 0
        if (is_tp == 1):
            tp = np.append(tp, np.array([[is_tp, mindist]]), axis = 0)
            print mindist
        else:
            fp = np.append(fp, np.array([[is_tp, mindist]]), axis = 0)
    return (tp, fp)
    
if __name__ == '__main__': 
    df = pd.read_csv('train.csv', header = 0)
    
    #Write a function to display an MNIST digit. Display one of each digit
    #  v = display_digit(df)
    
    #Examine the prior probability of the classes in the training data. 
    #Is it uniform across the digits?
    # Display a normalized histogram of digit counts. Is it even?
    
    # P.figure()
    #df['label'].plot(kind='hist', orientation='horizontal', cumulative=True)
    #df['label'].hist()
    #P.savefig('hist_plot')
    #P.close()
    #Pick one example of each digit from your training data. Then, for each sample digit, compute
    #and show the best match (nearest neighbor) between your chosen sample and the rest of
    #the training data. Use L2 distance between the two images’ pixel values as the metric. This
    #probably won’t be perfect, so add an asterisk next to the erroneous examples.

    """train = df.values

    empty_shape = np.shape(train[0][1:])

    for i in xrange(v.shape[0]):
        mindist = np.linalg.norm(v[i][1:])
        val = np.empty(empty_shape)
        for j in xrange(train.shape[0]):
            dist = np.linalg.norm(v[i][1:] - train[j][1:])
            if (dist > 0.0 and dist < mindist):
                mindist = dist
                val = train[j]
        if (v[i][0] != val[0]):
            print "Min Distance " + str(dist) + " *Val " + str(v[i][0]) + " " + str(val[0])
        else:
            print "Min Distance " + str(dist) + " Val " + str(v[i][0]) + " " + str(val[0])

    """

    #Consider the case of binary comparison between the digits 0 and 1. Ignoring all the other
    #digits, compute the pairwise distances for all genuine matches and all impostor matches,
    #again using the L2 norm. Plot histograms of the genuine and impostor distances on the same
    #set of axes. 
    df_0 = df.loc[df['label'] == 0]
    df_1 = df.loc[df['label'] == 1]
    df_0_1 = df_0.append(df_1).values
    tp, fp = l2norm(df_0_1, df_0_1)
    


# In[ ]:




# In[ ]:



