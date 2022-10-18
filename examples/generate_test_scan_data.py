"""
This script just runs through a particular dataset and publishes the patterns
onto a ZMQ socket at a specified delay
"""
import zmq
import time
from scipy.io import loadmat
import cdtools

# Define the socket to publish patterns on
context = zmq.Context()
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:37013")


#dataset = '/home/abe/Old Dropbox (MIT)/Old Projects/RPI_Code_Optics_Express/data/Optical_Data_ptycho.cxi'
#dataset = '/home/abe/Dropbox (MIT)/Photon Scattering Group/Data/Experiments/Synchro/ALS/20220921_RZP_4_Test/NS_220924026_ccdframes_0_0.cxi'
dataset = '/home/abe/Dropbox (MIT)/Photon Scattering Group/Data/Experiments/Synchro/ALS/20220921_RZP_4_Test/NS_220922032_ccdframes_0_0.cxi'
mask = loadmat('/home/abe/Dropbox (MIT)/Photon Scattering Group/Data/Experiments/Synchro/ALS/20220921_RZP_4_Test/default_mask.mat')
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(dataset)
dataset.patterns = dataset.patterns * (1-mask['mask'])


i = 0
while True:
    print('Starting a scan')
    pub.send_pyobj({'event':'start'})
    try:
        for i in range(0,len(dataset),3):
            print('Sending packet', i)
            
            pattern = dataset.patterns[i%len(dataset)].numpy()
            position = dataset.translations[i%len(dataset)][:2].numpy()
            
            event = {'event': 'frame', # frame is for a synthesized frame, exposure is for a single exposure.
                     'data': pattern,
                     'position': position}
            pub.send_pyobj(event)
            i += 1
            time.sleep(0.5)
        print('Ending a scan')
        pub.send_pyobj({'event':'stop',
                        'abort': False})
    except KeyboardInterrupt as e:
        print('Aborting a scan')
        pub.send_pyobj({'event':'stop',
                        'abort':True})
        raise e
