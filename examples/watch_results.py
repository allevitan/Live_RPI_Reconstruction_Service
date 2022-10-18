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
sub.connect("tcp://localhost:5556")
sub.setsockopt(zmq.SUBSCRIBE, b'')


message = sub.recv_pyobj()
fig = plt.figure()
if type(message) == dict and 'data' in message:
    im = plt.imshow(np.abs(message['data']), animated=True)
else:
    im = plt.imshow(np.abs(message), animated=True)
plt.title('Magnitude of reconstruction')

def updatefig(i):
    message = sub.recv_pyobj()
    if type(message) == dict:
        if 'data' in message and type(message['data']) != dict:
            obj = message['data']
        else:
            return im,
    else:
        obj = message
    xs, ys = np.mgrid[:obj.shape[-1],:obj.shape[-2]]
    xs = xs - np.mean(xs)
    ys = ys- np.mean(ys)
    mask = np.sqrt(xs**2+ys**2) < 25
    to_plot = mask * np.abs(obj)
    im.set_array(to_plot)
    im.set_clim([0, np.max(to_plot)])
    return im,


ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)
plt.show()
