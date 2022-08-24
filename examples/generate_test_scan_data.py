"""
This script just runs through a particular dataset and publishes the patterns
onto a ZMQ socket at a specified delay
"""
import zmq
import time
import cdtools

# Define the socket to publish patterns on
context = zmq.Context()
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5555")


dataset = '/home/abe/Old Dropbox (MIT)/Old Projects/RPI_Code_Optics_Express/data/Optical_Data_ptycho.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(dataset)


i = 0
while True:
    print('Starting a scan')
    pub.send_pyobj({'event':'start'})
    try:
        for i in range(0,len(dataset),5):
            print('Sending packet', i)
            
            pattern = dataset.patterns[i%len(dataset)].numpy()
            position = dataset.translations[i%len(dataset)][:2].numpy()
            
            event = {'event': 'exposure',
                     'data': pattern,
                     'position': position}
            pub.send_pyobj(event)
            i += 1
            time.sleep(0.3)
        print('Ending a scan')
        pub.send_pyobj({'event':'stop',
                        'abort': False})
    except KeyboardInterrupt as e:
        print('Aborting a scan')
        pub.send_pyobj({'event':'stop',
                        'abort':True})
        raise e
