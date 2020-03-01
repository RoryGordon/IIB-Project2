from numpy import genfromtxt
import numpy as np
import csv
import matplotlib.pyplot as plt


def despiker(data):
    #gather the data and put the columns into seperate variables
    ringing_data = genfromtxt(data, delimiter=',')
    times = ringing_data[:,0]
    masses = ringing_data[:,1]


    #shift the data down by 1 reading
    first_value = masses[0]
    shifted = masses[0:-1]
    shifted = np.insert(shifted,0,first_value)

    #find difference of data and shifted data to identify outliers
    spikes = abs(masses-shifted)
    # index the spikes
    indices = np.array(np.where(spikes>10))
    indices = indices.transpose()
    # at each spike, replace the error by an interpolation. i+3 since some spikes are over 2 readings
    for i in indices:
        masses[i] = (masses[i-1] + masses[i+3])/2
    #concatenate the timestamps with the new smoothed mass data
    despiked = np.column_stack((times,masses))

    return despiked

file = '/Users/shuowanghe/github/IIB-Project/rawdata 7:12:19/arms.csv'
a = despiker(file)
r = genfromtxt(file, delimiter=',')
np.savetxt("/Users/shuowanghe/github/IIB-Project/processed 7:12:19/smooth_arms.csv", a, delimiter=",")
flipped = a
flipped[:,1] = 77*np.ones(len(a))-a[:,1]
a = despiker(file)
np.savetxt("/Users/shuowanghe/github/IIB-Project/processed 7:12:19/flipped_arms.csv", flipped, delimiter=",")
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(a[:,0],r[:,1])
plt.show()
plt.plot(a[:,0],a[:,1])
plt.show()
plt.plot(a[:,0],flipped[:,1])
plt.show()
