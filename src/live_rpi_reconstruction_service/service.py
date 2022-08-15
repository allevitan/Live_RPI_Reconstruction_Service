from multiprocessing import Process
import time
import zmq
import numpy as np
import torch as t
from live_rpi_reconstruction_service.conjugate_gradient_rpi import iterate_CG

"""
This service is designed to be called as the target of a multiprocessing
Process, and it handles the reading of the various ZMQ streams and runs the
reconstructions on all the GPUS.

The way this is written preserves the order that the data came in - so the
reconstructions are always ordered in the same way as the input data. This
means that if the selected GPUs have differing speeds, it will always be
limited by the slowest GPU.

"""

def run_RPI_service(stopEvent, clearBufferEvent,
                    updateEvent, calibrationAcquiredEvent,
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
    nFrames = 5.0

    GPUs = controlInfo['GPUs']

    # This will store information about the ongoing reconstructions
    recs = [None] * len(GPUs)
    
    
    new_calibration = False
    calibration = None

    
    while not stopEvent.is_set():

        if clearBufferEvent.is_set():
            pattern_buffer = []
            clearBufferEvent.clear()
            
        while True:
            try:
                pattern_buffer.append(sub.recv_pyobj(flags=zmq.NOBLOCK))
            except zmq.ZMQError:
                break

        while True:
            try:
                calibration = calib_sub.recv_pyobj(flags=zmq.NOBLOCK)
                new_calibration = True
            except zmq.ZMQError:
                break

        readoutInfo['bufferSize'] = len(pattern_buffer)
        updateEvent.set()

        if new_calibration:
            readoutInfo['hasMask'] = 'mask' in calibration
            readoutInfo['hasBackground'] = 'background' in calibration

            basis_norm = np.linalg.norm(calibration['basis'], axis=0)
            readoutInfo['FOV'] = calibration['probe'].shape[-1] * basis_norm[-1]
            readoutInfo['basis'] = calibration['basis']
            readoutInfo['nModes'] = calibration['probe'].shape[0]

            probes = [t.as_tensor(calibration['probe'],
                                  device=gpu) for gpu in GPUs]
            
            if 'mask' in calibration:
                masks = [t.as_tensor(calibration['mask'],
                                     device=gpu) for gpu in GPUs]
            else:
                masks = [None for gpu in GPUs]
                
            if 'background' in calibration:
                backgrounds = [t.as_tensor(calibration['background'],
                                           device=gpu)  for gpu in GPUs]
            else:
                backgrounds = [None for gpu in GPUs]

            calibrationAcquiredEvent.set()
            new_calibration = False


        # If there's no calibration, the code won't proceed past this point
        # also if the buffer is empty and all the recs
        if calibration is None or (len(pattern_buffer)==0 and not any(recs)):
            time.sleep(0.01)
            continue

        # TODO: check if this actually does the reconstructions fast.
        # I *believe* that pytorch is just queueing up the calculations,
        # so that even though this looks serial it is actually allowing the
        # calculations to run in parallel on all the GPUs. However, the proof
        # is in the pudding.
        for i, rec in enumerate(recs):
            if rec is None and len(pattern_buffer) != 0:
                shape = [1]+[controlInfo['pixelCount']]*2

                sqrtPattern = t.as_tensor(pattern_buffer.pop(),
                                          device=GPUs[i])
                if controlInfo['background'] and 'background' in calibration:
                    sqrtPattern =t.clamp(sqrtPattern - backgrounds[i], min=0)
                sqrtPattern = t.sqrt(sqrtPattern)
                # FFTshifting the pattern once actually saves a lot of time
                # compared  to fftshifting the wavefields at each iteration.
                sqrtPattern = t.fft.ifftshift(sqrtPattern, dim=(-1,-2))
                
                obj = t.ones(shape, dtype=probes[i].dtype, device=GPUs[i])
                obj.requires_grad = True

                mask = (masks[i]
                        if controlInfo['mask'] and 'mask' in calibration
                        else None)
                background = (backgrounds[i] if controlInfo['background']
                              else None)
                        
                recs[i]= {'probe': probes[i],
                          'sqrtPattern': sqrtPattern,
                          'obj': obj,
                          'mask': mask,
                          'background': background,
                          'iter': 0,
                          'nIterations': controlInfo['nIterations'],
                          'clearEvery': 10,
                          'startTime': time.time()}
            elif rec is None:
                continue
            else:
                # TODO: right now this only uses the top probe mode!
                iterate_CG(rec)

        # Once we queue up all the calculations, we check if any
        # are done
        for i, rec in enumerate(recs):
            if rec is not None and rec['iter'] >= rec['nIterations']:
                pub.send_pyobj(rec['obj'].detach().cpu().numpy()[0])
                dt = (time.time() - rec['startTime'])/len(GPUs)

                # Only true on the first frame:
                if readoutInfo['processingTime'] == 0: 
                    readoutInfo['processingTime'] = dt
                else:
                    readoutInfo['processingTime'] = \
                        ((readoutInfo['processingTime'] * (nFrames-1) + dt)
                         / nFrames)
                
                recs[i] = None
                rec = None
                # We also update whenever we finish a reconstruction
                readoutInfo['bufferSize'] = len(pattern_buffer)
                updateEvent.set()



        
        
