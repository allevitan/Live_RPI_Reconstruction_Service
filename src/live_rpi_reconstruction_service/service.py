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


    pattern_buffer = []
    readoutInfo['processingTime'] = 0.0
    # The number of frames to run a trailing average of processing time on
    nFrames = 3.0
    dt = 0
    
    while not stopEvent.is_set():
        
        if clearBufferEvent.is_set():
            pattern_buffer = []
            clearBufferEvent.clear()
            
        while True:
            try:
                pattern_buffer.append(sub.recv_pyobj(flags=zmq.NOBLOCK))
            except zmq.ZMQError:
                break

        if not len(pattern_buffer):
            time.sleep(0.001)
            continue

        readoutInfo['processingTime'] = \
            ((readoutInfo['processingTime'] * (nFrames-1) + dt) / nFrames)
        readoutInfo['bufferSize'] = len(pattern_buffer)
        updateEvent.set()

        start_time = time.time()

        time.sleep(0.1)
        pattern_buffer.pop()

        dt = time.time() - start_time

