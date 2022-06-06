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

# Define the socket to publish results on
pub = context.socket(zmq.PUB)
pub.bind("tcp://*:5556")

# As made by make_example_calibration.py
calibration = loadmat('calibration.mat')
options = loadmat('options.mat')

# Device label to run the reconstructions on
dev = 'cuda:0'

# load the information from the calibration
probe = t.as_tensor(calibration['probe'][0], dtype=t.complex64, device=dev)
if 'mask' in calibration:
    mask = t.as_tensor(calibration['mask'], dtype=t.bool, device=dev)
else:
    mask = None


shape = [options['resolution'].ravel()[0],]*2
iterations = options['iterations'].ravel()[0]


# If fifo is set to True, the script will attempt to process every
# frame sent over the ZMQ socket, even if it backs up the buffer.
# If fifo is set to False, it will clear the ZMQ socket each time it
# runs a reconstruction, potentially dropping frames but never allowing
# a long queue to build up.
fifo = True

verbose = True

message = None
j = 0
while True:
    
    #  Implementing the logic for the two ways of reading from the socket
    if fifo:
        message = sub.recv_pyobj()
    else:
        try:
            message = sub.recv_pyobj(flags=zmq.NOBLOCK)
            j += 1
            continue
        except zmq.ZMQError:
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

    message = None

