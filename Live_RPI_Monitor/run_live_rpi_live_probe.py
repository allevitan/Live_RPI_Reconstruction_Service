"""
This script runs a live RPI reconstruction on the data coming in from one
ZMQ socket, then publishes the results on a second socket.
"""
import zmq
import time
import torch as t
from scipy.io import loadmat, savemat
from conjugate_gradient_rpi import run_CG


# Define the socket to subscribe to
context = zmq.Context()
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:5555")
sub.setsockopt(zmq.SUBSCRIBE, b'')

calib_sub = context.socket(zmq.SUB)
calib_sub.connect("tcp://localhost:5557")
calib_sub.setsockopt(zmq.SUBSCRIBE, b'')

# Define the socket to publish results on
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5556")

# Device label to run the reconstructions on
dev = 'cuda:0'

have_calibration = False


#
# This works by always clearing the queue so it is always processing the latest
# calibration and the latest diffraction pattern, at the expense of dropping
# patterns if it can't keep up
#

verbose = True

message = None
while True:
    
    #  Implementing the logic for the two ways of reading from the socket
    calib_message=None
    while True:
        try:
            calib_message = calib_sub.recv_pyobj(flags=zmq.NOBLOCK)
            continue
        except zmq.ZMQError:
            break

    while True:
        try:
            message = sub.recv_pyobj(flags=zmq.NOBLOCK)
            continue
        except zmq.ZMQError:
            break

    #message = sub.recv_pyobj()
    
    # Keep waiting until we get our first calibration
    if not have_calibration and calib_message is None:
        print('Waiting for first calibration', end='\r')
        continue
    
    # If we have a new calibration for this image
    elif calib_message is not None:
        # load the information from the calibration
        probe = t.as_tensor(calib_message['probe'], dtype=t.complex64, device=dev)
        if 'mask' in calib_message:
            mask = t.as_tensor(calib_message['mask'], dtype=t.bool, device=dev)
        else:
            mask = None

        shape = [calib_message['resolution'],]*2
        iterations = calib_message['iterations']
        have_calibration=True

    if message is None:
        time.sleep(0.001)
        continue
    
    st = time.time()
    pattern = t.as_tensor(message, device=dev)
    init = t.ones(shape, dtype=t.complex64, device=dev).unsqueeze(0)

    # This runs a reconstruction using a conjugate gradient based algorithm
    # which is tuned for speed. One consequence of this is that it only
    # uses a single probe mode, as the iteration time scales linearly with
    # number of probe modes
    result = run_CG(iterations, init, probe, pattern.unsqueeze(0),
                    clear_every=10, mask=mask)
    
    result = result.cpu().detach().numpy()
    
    # And send on the results
    pub.send_pyobj(result[0])
    if verbose:
        print('Reconstruction in', time.time()-st)


