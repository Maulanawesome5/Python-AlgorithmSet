from numpy import array
import kNN

"""
group, labels = kNN.createDataSet()
print("array",group)
print("\nlabels",labels)
"""

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()