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
    sub.connect(config['stitched_frame_subscription_port'])
    sub.setsockopt(zmq.SUBSCRIBE, b'')


    message = sub.recv_pyobj()
    fig = plt.figure()
    im = plt.imshow(np.abs(message), animated=True)
    plt.title('Magnitude of stitched reconstruction')
    
    def updatefig(i):
        message = sub.recv_pyobj()
        im.set_array(np.abs(message))
        return im,

    
    ani = animation.FuncAnimation(fig, updatefig, interval=0, blit=True)
    plt.show()

if __name__ == '__main__':
    sys.exit(main())
