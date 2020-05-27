###-------------------------------------------------------------------------###
###---------------------------------IMPORTS---------------------------------###
###-------------------------------------------------------------------------###
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from matplotlib import animation
###-------------------------------------------------------------------------###
###--------------------------------FUNCTIONS--------------------------------###
###-------------------------------------------------------------------------###
def despiker(data):
    #gather the data and put the columns into seperate variables
    times = data[:,0]
    masses = data[:,1]
    #shift the data down by 1 reading
    shifted = masses[0:-1]
    shifted = np.insert(shifted,0,masses[0])
    #find difference of data and shifted data to identify outliers
    spikes = abs(masses-shifted)
    # index the spikes
    indices = np.array(np.where(spikes>10)).transpose()
    # at each spike, replace the error by an interpolation. i+3 since some spikes are over 2 readings
    for i in indices:
        masses[i] = (masses[i-1] + masses[i+3])/2
    #concatenate the timestamps with the new smoothed mass data
    despiked = np.column_stack((times,masses))
    return despiked
def mass_to_force(despiked_data,standing_mass_start,standing_mass_end):
    standing_mass = np.mean(despiked_data[standing_mass_start:standing_mass_end,1])
    pull_force = standing_mass-despiked_data[:,1]
    return pull_force
###-------------------------------------------------------------------------###
###-------------------------------UNPACK DATA-------------------------------###
###-------------------------------------------------------------------------###
#Gather and unpack data from CSV

me_file = '/Users/shuowanghe/github/IIB-Project2/data/rawdata 12:11:19 GSM/follow.csv'
yangsheng_file = '/Users/shuowanghe/github/IIB-Project2/data/rawdata 22:11:19 Benet/yangsheng.csv'
sam_file = '/Users/shuowanghe/github/IIB-Project2/data/rawdata 12:11:19 GSM/sam.csv'
arms_file = '/Users/shuowanghe/github/IIB-Project2/data/rawdata 19:11:19 GSM/arms.csv'

me_data = genfromtxt(me_file,delimiter=',')
yangsheng_data = genfromtxt(yangsheng_file,delimiter=',')
sam_data = genfromtxt(sam_file,delimiter=',')
arms_data = genfromtxt(arms_file,delimiter=',')

me_despiked_data = despiker(me_data)
yangsheng_despiked_data = despiker(yangsheng_data)
sam_despiked_data = despiker(sam_data)
arms_despiked_data = despiker(arms_data)

me_timestamps,me_force = me_despiked_data[:,0],me_despiked_data[:,1]
yangsheng_timestamps,yangsheng_force = yangsheng_despiked_data[:,0],yangsheng_despiked_data[:,1]
sam_timestamps,sam_force = sam_despiked_data[:,0],sam_despiked_data[:,1]
arms_timestamps,arms_force = arms_despiked_data[:,0],arms_despiked_data[:,1]

me_pull_force = mass_to_force(me_despiked_data,262,556)
yangsheng_pull_force = mass_to_force(yangsheng_despiked_data,5780,7000)
sam_pull_force = mass_to_force(sam_despiked_data,215,895)
arms_pull_force = mass_to_force(arms_despiked_data,307,563)

me_smooth_pull_force = savgol_filter(me_pull_force,window_length=21,polyorder=3)*9.81
yangsheng_smooth_pull_force = savgol_filter(yangsheng_pull_force,window_length=21,polyorder=3)*9.81
sam_smooth_pull_force = savgol_filter(sam_pull_force,window_length=21,polyorder=3)*9.81
arms_smooth_pull_force = savgol_filter(arms_pull_force,window_length=21,polyorder=3)*9.81

plt.plot(me_force)
plt.title('me')
plt.show()

plt.plot(yangsheng_force)
plt.title('yangsheng')
plt.show()

plt.plot(sam_force)
plt.title('sam')
plt.show()

plt.plot(arms_force)
plt.title('arms')
plt.show()

# plt.plot(me_data[:,0],me_pull_force)
plt.plot(me_data[:,0],me_smooth_pull_force)
plt.xlabel(r'time (s)')
plt.ylabel(r'Pull force (N)')
plt.axis([81, 101, -100, 500])
plt.show()

# plt.plot(me_data[:,0],me_pull_force)
plt.plot(yangsheng_data[:,0],yangsheng_smooth_pull_force)
plt.xlabel(r'time (s)')
plt.ylabel(r'Pull force (N)')
plt.axis([130, 150, -100, 500])
plt.show()

# plt.plot(sam_data[:,0],sam_pull_force)
plt.plot(sam_data[:,0],sam_smooth_pull_force)
plt.xlabel(r'time (s)')
plt.ylabel(r'Pull force (N)')
plt.axis([68.5, 88.5, -100, 500])
plt.show()

# plt.plot(sam_data[:,0],sam_pull_force)
plt.plot(arms_data[:,0],arms_smooth_pull_force)
plt.xlabel(r'time (s)')
plt.ylabel(r'Arm intertial force measured (N)')
plt.axhline(y=0 , linestyle='--')
plt.axis([7.5, 23, -300, 300])
plt.show()
