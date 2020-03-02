import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks
from matplotlib import animation

m = 3
g = 9.81
l = 0.5
J = 2
fric = 1
dt = 1/50
theta_init = 2
time_elapsed = 20
theta = np.zeros(int(time_elapsed/dt))
theta[0] = theta_init
theta[1] = theta_init
T = np.zeros(int(time_elapsed/dt))
for k in range(1,int(time_elapsed/dt-1)):
    theta[k+1] = (2*theta[k] + (fric*dt/(2*J)-1)*theta[k-1] -m*g*l/J*np.sin(theta[k])*dt**2)/(1+fric*dt/(2*J))

plt.plot(np.linspace(0,time_elapsed,int(time_elapsed/dt)),theta)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.show()
