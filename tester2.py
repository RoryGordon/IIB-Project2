import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import genfromtxt

data = genfromtxt('/Users/shuowanghe/github/IIB-Project/processed 22:11:19 Benet/flipped_ringingvid1.csv', delimiter=',')

x = data[:,0]
y = data[:,1]

fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot([],[],lw=2)
ax.set_ylim(-2, np.max(y))
ax.set_xlim(0, np.max(30))
def animate(i):

    ax.set_xlim(x[i]-5, x[i]+1)
    line.set_xdata(x[:i])
    line.set_ydata(y[:i])

    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(x),
                              interval=3.6, blit=False)
ani.save('ringingvid1.gif', fps=50, extra_args=['-vcodec', 'libx264'])
