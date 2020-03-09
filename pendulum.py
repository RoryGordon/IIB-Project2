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
file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitmarch6th/userinput.csv'
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

#Function for getting angle from gyro by re-integrating at every theta=0 to prevent drift
def get_theta(data):
    timestamps = data[:,0]
    a_r = data[:,1]
    ang_vel = data[:,3]
    filtered_a_r = scipy.signal.savgol_filter(a_r,window_length=21, polyorder=2)
    theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
    theta = np.zeros(len(timestamps))
    for _ in theta_zeros:
        time_section = timestamps[_:]
        theta_dot_section = ang_vel[_:]
        theta[_:] = scipy.integrate.cumtrapz(theta_dot_section,time_section,initial=0)
    return theta

#Get theta=0 time stamps from filtered radial acceleration signal
theta_zeros,_ = scipy.signal.find_peaks(filtered_a_r,prominence=5)
#Integrate gyro signal from first theta=0 to get angle vs time, but may drift
theta = scipy.integrate.cumtrapz(theta_dot[theta_zeros[0]:],timestamps[theta_zeros[0]:],initial=0)

#Plot to compare reintegrated theta vs once integrated theta
plt.plot(timestamps,get_theta(data),label=r'Re-zeroed $\theta$')
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

#Use re-integrated theta from now on
theta = get_theta(data)

#Define the plot axes for the animation of sin(theta) vs ang accel
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-2,2))
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,

#Animation function. This is called sequentially
def animate(i):
    x = filtered_theta_double_dot[theta_zeros[0]:i+theta_zeros[0]]
    y = np.sin(theta[theta_zeros[0]:i+theta_zeros[0]])
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

#Other plots for visualisation
#plt.plot(timestamps,theta_dot,label='measured ang vel')
#plt.plot(timestamps,0.53*a_theta,label='measured tangential acc')
#plt.plot(timestamps,a_r,label='measured radial acc')
#plt.plot(timestamps_trunc1[:-1],theta1,label='theta (from integration)')
#plt.plot(timestamps_trunc2[:-1],abs(theta2),'b')
#plt.plot(data[theta_zeros,0],theta_dot[theta_zeros],'x')
plt.plot(timestamps,theta_double_dot,label='theta double dot (from diff)')
plt.plot(timestamps,filtered_theta_double_dot,label='theta double dot (from diff and filtered)')
#plt.plot(timestamps_trunc1[:-1],9.81*np.cos(theta1)+0.43*np.square(theta_dot_trunc1[:-1]),label='radial acc prediction')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]-9.81*np.cos(theta1),label='r thetadot^2')
#plt.plot(timestamps_trunc1[:-1],a_r[first_theta_zero_idx:-1]+9.81*np.sin(theta1),label='L theta double dot')
#plt.plot(timestamps,5*np.square(theta_dot),label='thetadot^2')
#plt.plot(timestamps_trunc1[:-1],np.cos(theta1),label='cos(theta)')
#plt.plot(theta_double_dot[-len(timestamps_trunc1):-1],-np.sin(theta1))
#plt.plot(a_theta[-len(timestamps_trunc1):-861],-np.sin(theta1[0:200]))
plt.legend(loc='lower left')
#plt.xlabel('time(s)')
#plt.ylabel('y')
#plt.axhline(0,color='b',linestyle='--')
#plt.xlim(0,timestamps[-1])
#plt.ylim(-5,5)
plt.show()
