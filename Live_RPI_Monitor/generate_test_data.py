"""
This script just runs through a particular dataset and publishes the patterns
onto a ZMQ socket at a specified delay
"""
import zmq
import time
import numpy as np
import CDTools
from scipy.io import loadmat, savemat
import torch as t

# Define the socket to publish patterns on
context = zmq.Context()
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5555")


calibration = loadmat('calibration.mat')
model = CDTools.models.RPI.from_calibration(calibration, [256,256])

dataset = '/home/abe/Old Dropbox (MIT)/Old Projects/RPI_Code_Optics_Express/data/Optical_Data_ptycho.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(dataset)

i = 0
while True:
    print('Sending packet', i)
    
    pattern = dataset.patterns[i%len(dataset)].numpy()

    pub.send_pyobj(pattern)
    i += 1
    time.sleep(0.2)
