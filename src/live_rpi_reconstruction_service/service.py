from multiprocessing import Process
import time
import zmq

def run_RPI_service(stopEvent, clearBufferEvent, updateEvent,
                    controlInfo, readoutInfo):

    # First we connect to the ZMQ services

    
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(controlInfo['patternsPort'])
    sub.setsockopt(zmq.SUBSCRIBE, b'')
    
    calib_sub = context.socket(zmq.SUB)
    calib_sub.connect(controlInfo['calibrationsPort'])
    calib_sub.setsockopt(zmq.SUBSCRIBE, b'')
    
    # Define the socket to publish results on
    pub = context.socket(zmq.PUB)
    pub.bind(controlInfo['resultsPort'])
    
    while readoutInfo['bufferSize'] < 20 and not stopEvent.is_set():
        time.sleep(0.1)
        #print(controlInfo['nIterations'])
        readoutInfo['bufferSize']+= 1
        updateEvent.set()


