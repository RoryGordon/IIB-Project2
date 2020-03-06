import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy import genfromtxt
import scipy
from scipy import integrate
from scipy.signal import find_peaks
from scipy.optimize import fmin_bfgs
from matplotlib import animation

file = '/Users/shuowanghe/github/IIB-Project/test.csv'
data = genfromtxt(file,delimiter=',')
timestamps = data[:,0]
a_r = data[:,1]
a_theta = data[:,2]
theta_dot = data[:,3]
#get max theta dot points to get theta=0 points
theta_zeros,_ = scipy.signal.find_peaks(abs(theta_dot),prominence=0)
print(timestamps[theta_zeros])
first_theta_zero_idx = theta_zeros[1]
timestamps_trunc1 = timestamps[first_theta_zero_idx:]
theta_dot_trunc1 = theta_dot[first_theta_zero_idx:]
theta_measured = scipy.integrate.cumtrapz(theta_dot_trunc1,timestamps_trunc1)
timestamps_trunc1 = timestamps_trunc1 - timestamps_trunc1[0]

ydata = np.gradient(theta_dot,timestamps)

m = 0.4
g = 9.81
l = 0.39
J = 1
fric = 0.08
popt = np.array([m,l,J,fric])
dt = 1/50
time_elapsed = 20
theta = np.zeros(int(time_elapsed/dt))
theta[0] = theta_measured[0]
theta[1] = theta_measured[1]
T = np.zeros(int(time_elapsed/dt))

def func(params,*args):
    m,l,J,fric = params
    theta,theta_prev,t = args
    g = 9.81
    theta_next = (2*theta + (fric*dt/(2*J)-1)*theta_prev -m*g*l/J*np.sin(theta)*dt**2)/(1+fric*dt/(2*J))
    error = abs(theta_next - theta_measured[t+1])
    return error

para = []

for k in range(1,len(theta_measured)-601):
    popt = fmin_bfgs(func,popt,args=(theta[k],theta[k-1],k))
    para.append(popt)
    m,l,J,fric = popt
    theta[k+1] = (2*theta[k] + (fric*dt/(2*J)-1)*theta[k-1] - m*g*l/J*np.sin(theta[k])*dt**2)/(1+fric*dt/(2*J))

print(np.mean(para,0))
popt = np.mean(para,0)

for k in range(1,int(time_elapsed/dt-1)):
    m,l,J,fric = popt
    theta[k+1] = (2*theta[k] + (fric*dt/(2*J)-1)*theta[k-1] -m*g*l/J*np.sin(theta[k])*dt**2)/(1+fric*dt/(2*J))

plt.plot(np.linspace(0,time_elapsed,int(time_elapsed/dt)),theta)
plt.plot(timestamps_trunc1[:-1],theta_measured)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.show()

print(para)
plt.plot(range(len(para[:][0])),para[:][0])
plt.plot(range(len(para[:][0])),para[:][1])
plt.plot(range(len(para[:][0])),para[:][2])
plt.plot(range(len(para[:][0])),para[:][3])
plt.show()
