import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks
from matplotlib import animation

file = '/Users/shuowanghe/github/IIB-Project2/data/adafruitmarch6th/ringdown.csv'
data = genfromtxt(file,delimiter=',')
timestamps = data[:,0]
a_r = data[:,1]
a_theta = data[:,2]
theta_dot = data[:,3]
theta_zeros,_ = scipy.signal.find_peaks(abs(a_r),prominence=1)
print(timestamps[theta_zeros])
first_theta_zero_idx = theta_zeros[0]
second_theta_zero_idx = theta_zeros[2]
timestamps_trunc1 = timestamps[first_theta_zero_idx:]
theta_dot_trunc1 = theta_dot[first_theta_zero_idx:]
timestamps_trunc2 = timestamps[second_theta_zero_idx:]
theta_dot_trunc2 = theta_dot[second_theta_zero_idx:]

theta_double_dot = np.gradient(theta_dot,timestamps)
filtered_theta_double_dot = scipy.signal.savgol_filter(theta_double_dot,window_length=25, polyorder=3)

theta1 = scipy.integrate.cumtrapz(theta_dot_trunc1,timestamps_trunc1)
theta2 = scipy.integrate.cumtrapz(theta_dot_trunc2,timestamps_trunc2)

t_init = data[0,0]
t = np.linspace(t_init,30,1500)
initial = 4.5
omega = 1.8
amplitude = 9.81


fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-2,2))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = filtered_theta_double_dot[first_theta_zero_idx:i+first_theta_zero_idx]
    y = np.sin(theta1[:i])
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(theta1), interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#anim.save('basic_animation.mp4', fps=50, extra_args=['-vcodec', 'libx264'])
plt.xlabel('sin(theta)')
plt.ylabel('theta double dot')
plt.show()

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
