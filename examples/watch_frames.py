"""
This script listens on the ZMQ socket for the output of the live RPI code,
and updates a figure to show the latest reconstruction
"""
import zmq
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

context = zmq.Context()
sub = context.socket(zmq.SUB)
sub.connect("tcp://131.243.73.225:49207")
sub.setsockopt(zmq.SUBSCRIBE, b'')


message = sub.recv_pyobj()
fig = plt.figure()
im = plt.imshow(np.abs(message), animated=True)
plt.title('Magnitude of reconstruction')

def updatefig(i):
    message = sub.recv_pyobj()
    im.set_array(np.abs(message))
    return im,


ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)
plt.show()
