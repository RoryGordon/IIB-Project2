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
from scipy.optimize import basinhopping
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.integrate import cumtrapz
from matplotlib import animation
from statsmodels.graphics import tsaplots
###-------------------------------------------------------------------------###
###------------------------UNPACK DATA AND GET ZEROS------------------------###
###-------------------------------------------------------------------------###
#Gather and unpack data from CSV
userinput_file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitmay5th/userinput.csv'
userinput = genfromtxt(userinput_file,delimiter=',')
freeswing_file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitmay5th/freeswing.csv'
freeswing = genfromtxt(freeswing_file,delimiter=',')
timestamps,a_r,a_theta,theta_dot = userinput[:,0], userinput[:,1], userinput[:,2], userinput[:,3]
#Differentiate gyro signal to get angular acceleration, then smooth with Sav-Gol filter
theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = savgol_filter(theta_double_dot,window_length=25, polyorder=3)
#Smooth the radial acceleration signal to find theta=zeros
filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
start = theta_zeros[0]
free_start_zeros,_ = find_peaks(savgol_filter(freeswing[:,1],window_length=21, polyorder=2),prominence=5)
free_start = free_start_zeros[0]
###-------------------------------------------------------------------------###
###--------------------------------FUNCTIONS--------------------------------###
###-------------------------------------------------------------------------###
#Function for getting angle from gyro by re-integrating at every theta=0 and distributing the drift
def get_theta(data):
    #find theta=zeros
    timestamps = data[:,0]
    a_r = data[:,1]
    theta_dot = data[:,3]
    filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
    start = theta_zeros[0]
    #integrate theta and distribute drift before every zero
    theta_int_once = cumtrapz(theta_dot[start:],timestamps[start:],initial=0)
    theta_fix = np.zeros(len(timestamps)) #generate an array to hold the fixed theta
    theta_fix[start:] = theta_int_once #put the hard integrated theta between the first 2 zeros
    prev_zero = start #initiate the last theta=0 as the first one for the loop
    for _ in theta_zeros[1:]: #reintegrate and correct drift
        time_section = timestamps[prev_zero:_+1] #carve out the section between the 2 zeros
        theta_dot_section = theta_dot[prev_zero:_+1]
        theta_section = cumtrapz(theta_dot_section,time_section,initial=0) #make the integration
        drift = theta_section[-1] #find the drift at the end of the section
        drift_vec = np.linspace(start=0,stop=drift,num=_-prev_zero+1) #generate a vector increasing steadily from 0 to the drift over that time frame
        theta_fix[prev_zero:_] = theta_section[:-1]-drift_vec[:-1] #make the correction so the last theta=0
        prev_zero = _ #store the zero point for the next loop
    return theta_fix #returns the drift-fixed theta

#Function for getting the gradient in the sin(theta) vs ang accel equation
def get_gradient(data):
    theta = get_theta(data) #get drift corrected theta from the data    #find theta=zeros
    timestamps = data[:,0]
    theta_dot = data[:,3]
    a_r = data[:,1]
    filtered_a_r = savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = find_peaks(filtered_a_r,prominence=5)
    start = theta_zeros[0]
    theta_double_dot = np.gradient(theta_dot,timestamps)
    x = np.sin(theta)[start:]
    y = savgol_filter(theta_double_dot,window_length=25, polyorder=3)[start:]
    p = np.polyfit(x,y,deg=1)[0] #fit a line through all of the data points
    return p

#Function for finding where force is being applied
def forcefinder(force):
    smooth_force = savgol_filter(force,window_length=45,polyorder=3)
    low_smooth_force = savgol_filter(force,window_length=35,polyorder=3)
    peaks,_ = find_peaks(abs(low_smooth_force),prominence=1)
    peak_matrix = np.zeros((len(peaks),3))
    peak_matrix[:,1] = peaks
    peaks = np.append(peaks,len(smooth_force))
    prev_peak = 0
    count = 0
    for i in peaks:
        mini_peaks_before,_ = find_peaks(abs(low_smooth_force)[prev_peak:i],prominence=0)
        begin = mini_peaks_before[-1]+prev_peak
        prev_end = mini_peaks_before[0]+prev_peak
        if count != len(peaks)-1:
            peak_matrix[count,0] = begin
        if count != 0:
            peak_matrix[count-1,2] = prev_end
        prev_peak = i
        count+=1
    return peak_matrix

###-------------------------------------------------------------------------###
###-------------------------------OPTIMISATION------------------------------###
###-------------------------------------------------------------------------###
#Optimise correction parameters for freeswing
def force_func(corrections):
    theta_correction_factor,gradient_correction_factor,theta_offset = corrections[0],corrections[1],corrections[2]
    theta = get_theta(freeswing)*theta_correction_factor+theta_offset
    theta_double_dot = np.gradient(freeswing[:,3],freeswing[:,0])
    filtered_theta_double_dot = savgol_filter(theta_double_dot,window_length=25, polyorder=3)*theta_correction_factor
    p = get_gradient(freeswing)*gradient_correction_factor
    force = filtered_theta_double_dot[start:]-p*np.sin(theta[start:])
    return sum(abs(force))
res=least_squares(fun=force_func, x0=[1,1,0])
print("theta factor: ",res.x[0],"\ngradient factor: ",res.x[1],"\ntheta offset: ",res.x[2],"\ncost: ",res.fun)
theta_correction_factor,gradient_correction_factor,theta_offset = res.x[0],res.x[1],res.x[2]
#get the -mlg/J gradient from fitting the userinput graph
p = get_gradient(userinput)
#Use re-integrated, drift corrected theta from now on
theta = get_theta(userinput)
#Calculate some force quantity T/J using just the userinput data
force = filtered_theta_double_dot[start:]-p*np.sin(theta[start:])
smooth_force = savgol_filter(force,window_length=25, polyorder=3)
#Calculate it post corrections from freeswing data
p_corrected = p*gradient_correction_factor
theta_corrected = theta*theta_correction_factor+theta_offset
theta_dot_corrected = theta_dot*theta_correction_factor
force_corrected = theta_correction_factor*filtered_theta_double_dot[start:]-p_corrected*np.sin(theta_corrected[start:])
smooth_force_corrected = savgol_filter(force_corrected,window_length=25, polyorder=3)

#get the indices where force is being applied and released
peak_matrix = forcefinder(force_corrected).astype(int)
theta_dot_zeros,_ = find_peaks(abs(theta_dot[start:]),prominence=1)
#get following data but when force isnt applied
no_force_times = timestamps[start:]
no_force_force = force
no_force_theta = theta[start:]
no_force_filtered_theta_double_dot = filtered_theta_double_dot[start:]
for i in range(len(peak_matrix)):
    force_range = range(peak_matrix[(len(peak_matrix)-1-i),0],peak_matrix[(len(peak_matrix)-1-i),2])
    no_force_times = np.delete(no_force_times,force_range)
    no_force_theta = np.delete(no_force_theta,force_range)
    no_force_force = np.delete(no_force_force,force_range)
    no_force_filtered_theta_double_dot =  np.delete(no_force_filtered_theta_double_dot,force_range)
#Optimise correction parameters again but only using no force data
def force_func2(corrections):
    theta_correction_factor,gradient_correction_factor,theta_offset = corrections[0],corrections[1],corrections[2]
    theta = no_force_theta*theta_correction_factor+theta_offset
    filtered_theta_double_dot = no_force_filtered_theta_double_dot*theta_correction_factor
    p = get_gradient(userinput)*gradient_correction_factor
    force = filtered_theta_double_dot-p*np.sin(theta)
    return sum(abs(force))

res2=least_squares(fun=force_func2, x0=[1,1,0])
print("theta factor 2: ",res2.x[0],"\ngradient factor 2: ",res2.x[1],"\ntheta offset 2: ",res2.x[2],"\ncost 2: ",res2.fun)
theta_correction_factor2,gradient_correction_factor2,theta_offset2 = res2.x[0],res2.x[1],res2.x[2]
theta_corrected2 = theta*theta_correction_factor2+theta_offset2
p_corrected2 = p*gradient_correction_factor2
theta_dot_corrected2 = theta_dot*theta_correction_factor2
force_corrected2 = filtered_theta_double_dot[start:]*theta_correction_factor2-p_corrected2*np.sin(theta_corrected2[start:])
smooth_force_corrected2 = savgol_filter(force_corrected2,window_length=25, polyorder=3)

###-------------------------------------------------------------------------###
###-------------------------------UNCERTAINTY-------------------------------###
###-------------------------------------------------------------------------###
no_bins = 20
means_variance = np.zeros([no_bins,2])
free_theta = get_theta(freeswing)[free_start:]
free_p = get_gradient(freeswing)
free_theta_double_dot = np.gradient(freeswing[:,3][free_start:],freeswing[:,0][free_start:])
free_filtered_theta_double_dot = savgol_filter(free_theta_double_dot,window_length=25, polyorder=3)
free_force = free_filtered_theta_double_dot-free_p*np.sin(free_theta)
fixed_force = np.zeros(len(free_force))
for bin in range(no_bins):
    bin_start = (bin-no_bins/2)*2*np.pi/no_bins
    bin_end = (bin-no_bins/2+1)*2*np.pi/no_bins
    bin_indices = np.where((free_theta>=bin_start)&(free_theta<=bin_end))
    if np.size(bin_indices)>0:
        means_variance[bin,0] = np.mean(free_force[bin_indices])
        means_variance[bin,1] = np.var(free_force[bin_indices])
        fixed_force[bin_indices] = free_force[bin_indices] - means_variance[bin,0]
free_force_corrected = theta_correction_factor*free_filtered_theta_double_dot-gradient_correction_factor*free_p*np.sin(free_theta*theta_correction_factor+theta_offset)
###-------------------------------------------------------------------------###
###----------------------------------PLOTS----------------------------------###
###-------------------------------------------------------------------------###
#Plot means with bins
plt.plot(range(no_bins),means_variance[:,0])
plt.title(r'Mean force measured in freeswing for each $\theta$ section')
plt.xlabel(r'$\theta$ bin')
plt.show()
#Plot variances with bins
plt.plot(range(no_bins),means_variance[:,1])
plt.title(r'Variance of force measured in freeswing for each $\theta$ section')
plt.xlabel(r'$\theta$ bin')
plt.show()
#Plot freeswing force with the mean corrected version
plt.plot(freeswing[:,0][free_start:],free_theta,label=r'$\theta$')
plt.plot(freeswing[:,0][free_start:],free_force,label='Force measured')
plt.plot(freeswing[:,0][free_start:],free_force_corrected,label='Force with parameter corrections')
plt.plot(freeswing[:,0][free_start:],fixed_force,label='Measured for corrected with averages')
plt.title('Force measured in freeswing and mean corrected version')
plt.xlabel('times (s)')
plt.legend()
plt.show()
#Plot line fit through theta double dot vs sin(theta) relationship
x = np.sin(theta[start:])
y = filtered_theta_double_dot[start:]
x_corrected=np.sin(theta_corrected2[start:])
y_corrected=filtered_theta_double_dot[start:]*theta_correction_factor2
plt.plot(x,y,'.',label='Raw measurements') #plot all the data points
plt.plot(x_corrected,y_corrected,'.',label='Corrected data') #plot all the data points
plt.plot(x,p*x,label='Line fit through origin of original points') #plot the fitted line, through the origin
plt.plot(x_corrected,p_corrected2*x_corrected,label='Line fit through origin of corrected points')
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$) with a line fitted through')
plt.legend()
plt.show()

#Force, theta and theta_dot vs time
plt.plot(timestamps[start:],smooth_force,label="Force measurement (Smoothed)")
plt.plot(timestamps[start:],smooth_force_corrected,label="Corrected force measurement (Smoothed)")
plt.plot(timestamps[start:],theta_corrected[start:],label=r'$\theta$')
plt.plot(timestamps[start:],theta_dot_corrected[start:],label=r'$\dot{\theta}$')
plt.axhline(0,color='b',linestyle='--')
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.legend()
plt.show()

#Plot Force vs (mlg/J)sin(theta)
plt.plot(-p*np.sin(theta[start:]),smooth_force,label="As measured")
plt.plot(-p_corrected2*np.sin(theta_corrected2[start:]),smooth_force_corrected2,label="Corrected")
plt.xlabel(r'$\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'$\frac{T}{J}$ vs $\frac{mgl}{J}sin(\theta)$')
plt.legend()
plt.show()

#Plot Force vs velocity
plt.plot(theta_dot[start:],smooth_force,label="As measured")
plt.plot(theta_dot_corrected[start:],smooth_force_corrected,label="Corrected")
plt.xlabel(r'$\dot{\theta}$')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'Force vs $\dot{\theta}$')
plt.legend()
plt.show()
