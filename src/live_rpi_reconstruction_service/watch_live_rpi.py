"""
This script listens on the ZMQ socket for the output of the live RPI code,
and updates a figure to show the latest reconstruction
"""
import zmq
import time
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import sys

# This is apparently the best-practice way to load config files from within
# the package
import importlib.resources
import json


def main(argv = sys.argv):
    package_root = importlib.resources.files('live_rpi_reconstruction_service')
    # This loads the default configuration first. This file is managed by
    # git and should not be edited by a user
    config = json.loads(package_root.joinpath('defaults.json').read_text())\

    # And now, if the user has installed an optional config file, we allow it
    # to override what is in defaults.json
    config_file_path = package_root.joinpath('config.json')

    # not sure if this works with zipped packages
    if config_file_path.exists():
        config.update(json.loads(config_file_path.read_text()))

    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(config['rpi_frame_subscription_port'])
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
        mask = np.sqrt(xs**2+ys**2) < 70
        to_plot = mask * np.abs(obj)
        im.set_array(to_plot)
        im.set_clim([0, np.max(to_plot)])
        return im,


    ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
