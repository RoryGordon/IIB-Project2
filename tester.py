import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')
import csv


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax = plt.axes(xlim=(0, 20), ylim=(-2, 90))
#line, = ax.plot([], [], lw=1)
x_len = 400
xs = np.zeros([x_len,1])
ys = np.zeros([x_len,1])

#def init():
    #line.set_data(xs, ys)
    #return line,
    #ax1.clear()
    #ax1.plot(xs, ys)

ringing_data = genfromtxt('/Users/shuowanghe/github/IIB-Project/processed 22:11:19 Benet/flipped_ringingvid1.csv', delimiter=',')

def animate(i,ringing_data,xs,ys):
    xs = np.append(xs,ringing_data[i,0])
    xs = xs[-x_len:]
    ys = np.append(ys,ringing_data[i,1])
    ys = ys[-x_len:]
    ax1.clear()
    ax1.plot(xs, ys)
    #line.set_data(xs, ys)
    print(xs[-1],ys[-1])
    #return line,

anim = FuncAnimation(fig, animate, fargs = (ringing_data,xs,ys),interval=10)

plt.show()
#anim.save('test.gif', writer='PillowWriter')
