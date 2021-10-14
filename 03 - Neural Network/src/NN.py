import numpy as np

def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)

x = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])
y = np.array([ [0], [1], [1], [0] ])
np.random.seed(1)

#Randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in range(60000):
    #Feed forward through layers 0, 1, and 2
    10 = x
    11 = nonlin(np.dot(10, syn0))
    12 = nonlin(np.dot(11, syn1))
    
    #How much did we miss the target value?
    error_12 = y - 12
    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(error_12))))

# in what direction is the target value?
# were we really sure? if so, don't change too much
delta_12 = error_12 * nonlin(12, deriv=True)

# how much did each 11 value contribute to the 12 error
# (according to the weights) ?
error_11 = delta_12.dot(syn1.T)

# in what direction is the target 11?
# were we really sure? if so, don't change too much
delta_11 = error_11 * nonlin(11, deriv=True)

syn1 += 11.T.dot(delta_12)
syn0 += 10.T.dot(delta_11)

"""
NB : Mungkin terdapat error pada syntax dikarenakan penulis
buku menggunakan python versi 2.7.9 saat menulis bukunya.
"""