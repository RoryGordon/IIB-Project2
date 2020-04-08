#Imports
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks
from matplotlib import animation

#Gather and unpack data from CSV
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitmarch6th/ringdown.csv'
data = genfromtxt(file,delimiter=',')
timestamps = data[:,0]
a_r = data[:,1]
a_theta = data[:,2]
theta_dot = data[:,3]
#Smooth the radial acceleration signal
filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
#Differentiate gyro signal to get angular acceleration, then smooth with Sav-Gol filter
theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = scipy.signal.savgol_filter(theta_double_dot,window_length=25, polyorder=3)
#Get theta=0 time stamps from filtered radial acceleration signal
theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
#Integrate gyro signal from first theta=0 to get angle vs time, but will drift
theta = scipy.integrate.cumtrapz(theta_dot[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)

#Function for getting angle from gyro by re-integrating at every theta=0 and distributing the drift
def get_theta(data):
    timestamps = data[:,0] #unpack the data again, filter a_r and find theta=0s
    a_r = data[:,1]
    ang_vel = data[:,3]
    filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
    theta = np.zeros(len(timestamps)) #generate empty array to hold theta
    for _ in theta_zeros: #hard re-integrate at every theta=0
        time_section = timestamps[_:]
        theta_dot_section = ang_vel[_:]
        theta[_:] = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0)
    theta_int_once = scipy.integrate.cumtrapz(ang_vel[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)
    theta_fix = np.zeros(len(timestamps)) #generate an array to hold the fixed theta
    theta_fix[theta_zeros[0]:] = theta_int_once #put the hard integrated theta between the first 2 zeros
    prev_zero = theta_zeros[0] #initiate the last theta=0 as the first one for the loop
    for _ in theta_zeros[1:]: #reintegrate and correct drift
        time_section = timestamps[prev_zero:_+1] #carve out the section between the 2 zeros
        theta_dot_section = ang_vel[prev_zero:_+1]
        theta_section = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0) #make the integration
        drift = theta_section[-1] #find the drift at the end of the swing
        drift_vec = np.linspace(start=0,stop=drift,num=_-prev_zero+1) #generate a vector increasing steadily from 0 to the drift over that time frame
        theta_fix[prev_zero:_] = theta_section[:-1]-drift_vec[:-1] #make the correction so last theta=0
        prev_zero = _ #store the zero point for the next loop
    return theta,theta_fix #returns both the hard reintegrated theta and the drift-fixed theta

#Function for getting the gradient in the sin(theta) vs ang accel equation
def get_gradient(freeswing):
    data = genfromtxt(freeswing,delimiter=',') #extract data from a freeswing run
    theta = get_theta(data)[1] #get fixed theta from the freeswings
    x = np.sin(theta)
    y = scipy.signal.savgol_filter(np.gradient(data[:,3],data[:,0]),window_length=25, polyorder=3)
    allowed_indices = np.where(abs(x)<0.5) #find indices where line is straight
    p = np.polyfit(x[allowed_indices],y[allowed_indices],deg=1) #fit a line through all of the data points within the cutoff
    x = x[theta_zeros[0]:] #use points after the first theta=zero
    y = y[theta_zeros[0]:]
    plt.plot(x,y,'x') #plot all the data points
    plt.plot(x,p[0]*x,'r') #plot the fitted line, through the origin
    plt.show()
    return p[0]


#Plot to compare reintegrated theta vs once integrated theta
plt.plot(timestamps,get_theta(data)[0],label=r'Re-zeroed $\theta$',linewidth=5)
plt.plot(timestamps,get_theta(data)[1],label=r'Re-zeroed and drift corrected $\theta$',linewidth=3)
plt.plot(timestamps[theta_zeros[0]:],theta,label=r'$\theta$ intergrated from 1st zero')
plt.plot(timestamps[theta_zeros],np.zeros(len(theta_zeros)),'gx',label=r'$\theta=0$')
plt.legend(loc='lower right')
plt.title(r'Comparison of $\theta$ calculated from initial zero vs recalulated at every zero')
plt.xlabel('time(s)')
plt.ylabel(r'$\theta$(rad)')
plt.show()

#Plot to show filtered radial acceleration to find all theta=0 time stamps
plt.plot(timestamps,a_r,label=r'measured $a_r$')
plt.plot(timestamps,filtered_a_r,label=r'measured $a_r$ with Savitzky–Golay filter')
plt.plot(data[theta_zeros,0],filtered_a_r[theta_zeros],'x')
plt.legend()
plt.title(r'Filtered vs unfiltered $a_r$, used to find all $\theta=0$')
plt.xlabel('time(s)')
plt.ylabel(r'$a_r$(m/$s^2$)')
plt.show()

p = get_gradient('/Users/shuowanghe/github/IIB-Project2/data/adafruitmarch6th/freeswing.csv')

#Use re-integrated, drift corrected theta from now on
theta = get_theta(data)[1]

#Define the plot axes for the animation of sin(theta) vs ang accel
fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-10,10))
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,

#Animation function. This is called sequentially
def animate(i):
    y = filtered_theta_double_dot[theta_zeros[0]:i+theta_zeros[0]]
    x = np.sin(theta[theta_zeros[0]:i+theta_zeros[0]])
    line.set_data(x, y)
    return line,

#Call the animator. blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(theta), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#Save the animation
#anim.save('basic_animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
#Plot the animation
plt.xlabel(r'sin($\theta$)')
plt.ylabel(r'$\"{\theta}$(rad/$s^2$)')
plt.title(r'$\"{\theta}$ vs sin($\theta$)')
plt.show()

#Plot bell angle against some force calculation
force = filtered_theta_double_dot[theta_zeros[0]:]-p*np.sin(theta[theta_zeros[0]:])
plt.plot(timestamps[theta_zeros[0]:],force,label="Force measurement")
plt.plot(timestamps[theta_zeros[0]:],scipy.signal.savgol_filter(force,window_length=25, polyorder=3),label="Force measurement (Smoothed)")
plt.plot(timestamps[theta_zeros[0]:],theta[theta_zeros[0]:],label=r'$\theta$')
#plt.plot(timestamps[theta_zeros[0]:],filtered_theta_double_dot[theta_zeros[0]:])
#plt.plot(timestamps[theta_zeros[0]:],p*np.sin(theta[theta_zeros[0]:]))
plt.xlabel(r't(s)')
plt.ylabel(r'$\"{\theta}+\frac{mgl}{J}sin(\theta)$(rad/$s^2$)')
plt.title(r'$\frac{T}{J}$ vs time')
plt.legend()
plt.show()

#Animate Pendulum
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
    y = -np.cos(theta[i]) #Bell's y position
    x = np.sin(theta[i]) #Bell's x position
    bell.set_data(x, y)
    x_torque = filtered_theta_double_dot[i]-p*np.sin(theta[i]) #Torque/J at time i
    torque.set_data(x_torque/3,0)
    return bell, torque,

#Call the animator. blit=True means only re-draw the parts that have changed.
anim_pend = animation.FuncAnimation(fig_pend, animate_pend, init_func=init_pend,
                               frames=len(theta), interval=20, blit=True)

plt.title('Bell swinging and force applied')
plt.legend()
plt.show()




#Other plots for visualisation
#plt.plot(timestamps,theta_dot,label='measured ang vel')
plt.plot(timestamps,scipy.signal.savgol_filter(a_theta,window_length=21, polyorder=2),label='measured tangential acc')
#plt.plot(timestamps,a_r,label='measured radial acc')
#plt.plot(timestamps_trunc1[:-1],theta1,label='theta (from integration)')
#plt.plot(timestamps_trunc2[:-1],abs(theta2),'b')
#plt.plot(data[theta_zeros,0],theta_dot[theta_zeros],'x')
#plt.plot(timestamps,theta_double_dot,label='theta double dot (from diff)')
#plt.plot(timestamps,filtered_theta_double_dot,label='theta double dot (from diff and filtered)')
#plt.plot(timestamps_trunc1[:-1],9.81*np.cos(theta1)+0.43*np.square(theta_dot_trunc1[:-1]),label='radial acc prediction')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]-9.81*np.cos(theta1),label='r thetadot^2')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]+9.81*np.sin(theta1),label='L theta double dot')
#plt.plot(timestamps,5*np.square(theta_dot),label='thetadot^2')
#plt.plot(timestamps_trunc1[:-1],np.cos(theta1),label='cos(theta)')
#plt.plot(theta_double_dot[-len(timestamps_trunc1):-1],-np.sin(theta1))
#plt.plot(a_theta[-len(timestamps_trunc1):-861],-np.sin(theta1[0:200]))
plt.legend(loc='lower left')
plt.axhline(0,color='b',linestyle='--')
#plt.xlim(0,timestamps[-1])
#plt.ylim(-5,5)
plt.title('Other temporal plots that may be of interest for visualisation')
plt.xlabel('time(s)')
plt.show()
