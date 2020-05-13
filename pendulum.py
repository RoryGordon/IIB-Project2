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
    start = find_peaks(savgol_filter(data[:,1],window_length=21, polyorder=2),prominence=5)[0][0]
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
    return peak_matrix.astype(int)

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
peak_matrix = forcefinder(force_corrected)
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
smooth_force_corrected2 = savgol_filter(force_corrected2,window_length=35, polyorder=3)
#find zero crossings of theta_dot, i.e. when bell is at extremes
theta_dot_zeros,_ = find_peaks(-abs(theta_dot[start:]))
#Produce clean force curve
noisy_peaks = find_peaks(-abs(smooth_force_corrected2),height=-0.5)[0]
force_curve = np.zeros(len(force_corrected2))
starts_and_ends = np.zeros([len(peak_matrix),2])
force_in = np.zeros(len(smooth_force_corrected2))
force_out = np.zeros(len(smooth_force_corrected2))
energy_in = np.zeros(len(peak_matrix))
energy_out = np.zeros(len(peak_matrix))
power_in = np.zeros(len(peak_matrix))
power_out = np.zeros(len(peak_matrix))
count = 0
for _ in peak_matrix[:,1]:
    force_start = max(noisy_peaks[noisy_peaks<_])
    force_end = min(noisy_peaks[noisy_peaks>_])
    # starts_and_ends[count,0],starts_and_ends[count,1] = force_start,force_end
    force_curve[force_start:force_end] = smooth_force_corrected2[force_start:force_end]
    if force_start<theta_dot_zeros[count]:
        force_out[force_start:theta_dot_zeros[count]] = force_curve[force_start:theta_dot_zeros[count]]
    if force_end>theta_dot_zeros[count]:
        force_in[theta_dot_zeros[count]:force_end] = force_curve[theta_dot_zeros[count]:force_end]
    energy_in[count] = max(cumtrapz(abs(force_in)[force_start:force_end],initial=0))
    energy_out[count] = max(cumtrapz(abs(force_out)[force_start:force_end],initial=0))
    power_in[count] = sum(np.multiply(abs(theta_dot[start:])[force_start:force_end],abs(force_in)[force_start:force_end]))
    power_out[count] = sum(np.multiply(abs(theta_dot[start:])[force_start:force_end],abs(force_out)[force_start:force_end]))
    count += 1
print(power_in)
print(power_out)
print(power_in-power_out)

###-------------------------------------------------------------------------###
###-------------------------------UNCERTAINTY-------------------------------###
###-------------------------------------------------------------------------###
def remove_mean(data):
    start = find_peaks(savgol_filter(data[:,1],window_length=21, polyorder=2),prominence=5)[0][0]
    theta_double_dot = np.gradient(data[:,3],data[:,0])[start:]
    filtered_theta_double_dot = savgol_filter(theta_double_dot,window_length=25, polyorder=3)
    p = get_gradient(data)
    theta = get_theta(data)[start:]
    force = filtered_theta_double_dot-p*np.sin(theta)
    no_bins = 20
    means_variance = np.zeros([no_bins,2])
    free_theta = get_theta(freeswing)[free_start:]
    free_p = get_gradient(freeswing)
    free_theta_double_dot = np.gradient(freeswing[:,3][free_start:],freeswing[:,0][free_start:])
    free_filtered_theta_double_dot = savgol_filter(free_theta_double_dot,window_length=25, polyorder=3)
    free_force = free_filtered_theta_double_dot-free_p*np.sin(free_theta)
    fixed_force = filtered_theta_double_dot-p*np.sin(theta)
    for bin in range(no_bins):
        bin_start = (bin-no_bins/2)*2*np.pi/no_bins
        bin_end = (bin-no_bins/2+1)*2*np.pi/no_bins
        free_bin_indices = np.where((free_theta>=bin_start)&(free_theta<bin_end))
        data_bin_indices = np.where((theta>=bin_start)&(theta<bin_end))
        if np.size(free_bin_indices)>0:
            means_variance[bin,0] = np.mean(free_force[free_bin_indices])
            means_variance[bin,1] = np.var(free_force[free_bin_indices])
            fixed_force[data_bin_indices] = force[data_bin_indices] - means_variance[bin,0]
    return fixed_force, means_variance

plt.plot(timestamps[start:][theta_dot_zeros],theta_dot[start:][theta_dot_zeros],'x')
plt.plot(timestamps[start:],abs(force_in),'g')
plt.plot(timestamps[start:],abs(force_out),'r')
plt.plot(timestamps[start:],abs(theta_dot)[start:])
# plt.plot(timestamps[start:],-abs(smooth_force_corrected2))
# plt.plot(timestamps[start:][little_peaks],-abs(smooth_force_corrected2)[little_peaks],'x')
plt.show()

# plt.plot(timestamps[start:],abs(force_corrected2))
# plt.plot(timestamps[start:],abs(smooth_force_corrected2))
# plt.plot(timestamps[start:],-abs(force_corrected2))
# plt.show()
# plt.plot(timestamps[start:][peak_matrix2[:,0]],abs(smooth_force_corrected2)[peak_matrix2[:,0]],'x')
# plt.plot(timestamps[start:][peak_matrix2[:,2]],abs(smooth_force_corrected2)[peak_matrix2[:,2]],'x')
# plt.show()

"""
###-------------------------------------------------------------------------###
###--------------------------------ANIMATIONS-------------------------------###
###-------------------------------------------------------------------------###
#THETA_DOUBLE_DOT vs SIN(THETA) ANIMATION

fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-10,10))
line, = ax.plot([], [], lw=1)
def init():
    line.set_data([], [])
    return line,
#Animation function. This is called sequentially
def animate(i):
    y = filtered_theta_double_dot[start:start+i]
    x = np.sin(theta[start:start+i])
    line.set_data(x, y)
    return line,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(theta), interval=20, blit=True)
#Save the animation
#anim.save('2sidepushfix.gif', fps=50, extra_args=['-vcodec', 'libx264'])
#Plot the animation
x = np.array([-2,2])
plt.plot(x,p*x,'r',linewidth=1,label="Fit") #plot the fitted line, through the origin
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$)')
plt.show()
"""
#PENDULUM AND FORCE ANIMATION
fig_pend = plt.figure()
ax_pend = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5,1.5))
bell, = ax_pend.plot([], [], 'bo', lw=5,label='Bell')
torque, = ax_pend.plot([], [], 'ro', lw=5, label='Force')
def init_pend():
    bell.set_data([], [])
    torque.set_data([], [])
    return bell, torque,
#Animation function. This is called sequentially
def animate_pend(i):
    y = -np.cos(theta[i+start]) #Bell's y position
    x = np.sin(theta[i+start]) #Bell's x position
    bell.set_data(x, y)
    x_torque = force_curve[i] #Torque/J at time i
    torque.set_data(x_torque/3,0)
    return bell, torque,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim_pend = animation.FuncAnimation(fig_pend, animate_pend, init_func=init_pend, frames=len(theta[start:]), interval=20, blit=True)
#anim_pend.save('2sidepush_force.gif', fps=50, extra_args=['-vcodec', 'libx264'])
plt.title('Bell swinging and force applied')
plt.legend()
plt.show()
"""
#FORCE vs VELOCITY ANIMATION
fig_force_vel = plt.figure()
ax_force_vel = plt.axes(xlim=(-5, 5), ylim=(-3,3))
line_force_vel, = ax_force_vel.plot([], [], lw=2)
def init_force_vel():
    line_force_vel.set_data([], [])
    return line_force_vel,
#Animation function. This is called sequentially
def animate_force_vel(i):
    y = smooth_force[0:i]
    x = theta_dot[start:start+i]
    line_force_vel.set_data(x, y)
    return line_force_vel,
#Call the animator. blit=True means only re-draw the parts that have changed.
anim_force_vel = animation.FuncAnimation(fig_force_vel, animate_force_vel, init_func=init_force_vel,
                               frames=len(force), interval=20, blit=True)
#Save the animation
#anim.save('2sidepushfix.gif', fps=50, extra_args=['-vcodec', 'libx264'])
#Plot the animation
plt.xlabel(r'$\dot{\theta}$')
plt.ylabel(r'Some force $\frac{T}{J}$')
plt.title(r'Force vs $\dot{\theta}$')
plt.show()

###-------------------------------------------------------------------------###
###----------------------------------PLOTS----------------------------------###
###-------------------------------------------------------------------------###
#Plot line fit through theta double dot vs sin(theta) relationship

#allowed_indices = np.where(abs(x)<20.2) #find indices where line is straight
x = np.sin(theta[start:])
y = filtered_theta_double_dot[start:]
plt.plot(x,y,'x',label='All points') #plot all the data points
#plt.plot(x[allowed_indices],y[allowed_indices],'gx',label='Points used in fitting') #plot the allowed data points for fitting
plt.plot(x,p*x,'r',label='Line fit through origin of central points') #plot the fitted line, through the origin
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$) with a line fitted through')
plt.show()

#Force, theta and theta_dot vs time
plt.plot(timestamps[start:],force,label="Force measurement")
plt.plot(timestamps[start:],smooth_force,label="Force measurement (Smoothed)")
plt.plot(timestamps[start:],theta[start:],label=r'$\theta$')
plt.plot(timestamps[start:],theta_dot[start:],label=r'$\dot{\theta}$')
plt.axhline(0,color='b',linestyle='--')
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.axhline(0,color='b',linestyle='--')
plt.legend()
plt.show()

#Force, theta and theta_dot vs time

#plt.plot(timestamps[start:][peaks],force[peaks],'x',label="Force measurement")
#plt.plot(timestamps[start:],force,label="Force measurement")
plt.plot(timestamps[start:],force,label="Force measurement")
plt.plot(timestamps[start:],smooth_force,label="Force measurement (Smoothed)")
for i in range(len(peak_matrix)):
    plt.plot(timestamps[start:][peak_matrix[i,0]:peak_matrix[i,2]],force[peak_matrix[i,0]:peak_matrix[i,2]],'r-')
    plt.plot(timestamps[start:][peak_matrix[i,0]:theta_dot_zeros[i]],force[peak_matrix[i,0]:theta_dot_zeros[i]],'b-')
    plt.plot(timestamps[start:][theta_dot_zeros[i]:peak_matrix[i,2]],force[theta_dot_zeros[i]:peak_matrix[i,2]],'g-')
#plt.plot(timestamps[start:][peak_matrix[:,0]],abs(smooth_force)[peak_matrix[:,0]],'x')
#plt.plot(timestamps[start:][peak_matrix[:,2]],abs(smooth_force)[peak_matrix[:,2]],'x')
plt.plot(timestamps[start:],theta[start:],label=r'$\theta$')
#plt.plot(timestamps[start:],abs(theta_dot[start:]),label=r'$\dot{\theta}$')
plt.axhline(0,color='b',linestyle='--')
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.axhline(0,color='b',linestyle='--')
plt.legend()
plt.show()



#Plot Force vs (mlg/J)sin(theta)
plt.plot(-p*np.sin(theta[start:]),smooth_force)
plt.xlabel(r'$\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'$\frac{T}{J}$ vs $\frac{mgl}{J}sin(\theta)$')
plt.show()

#Plot Force vs velocity
plt.plot(theta_dot[start:],smooth_force)
plt.xlabel(r'$\dot{\theta}$')
plt.ylabel(r'$\frac{T}{J}$')
plt.title(r'Force vs $\dot{\theta}$')
plt.show()
"""
