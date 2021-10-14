import numpy as np
import matplotlib.pyplot as plt

limitIpk = np.array([1.00, 4.00])
ipk = np.array([3.38, 2.54, 2.32, 2.33, 1.91, 1.93, 1.89, 1.89])

plt.ylabel('IPK Limit')
plt.xlabel('Semester')
plt.plot(ipk)
plt.show()