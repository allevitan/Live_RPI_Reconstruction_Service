import torch as t
import cdtools
from cdtools.tools import interactions
import numpy as np
import zmq
import sys

# This is apparently the best-practice way to load config files from within
# the package
import importlib.resources
import json



def add_frame_to_object(obj, weights, basis, offset,
                        new_obj, new_weights, position):
    if obj is None:
        obj = np.abs(new_obj)
        weights = np.copy(new_weights)
        offset = np.copy(position)
        return obj, weights, offset

    t_basis = t.as_tensor(basis)
    t_pos = t.zeros([3], dtype=t_basis.dtype)

    t_pos[:2] = t.as_tensor(position-offset)
    pix_position = interactions.translations_to_pixel(t_basis, t_pos).numpy()

    for (i,j) in (0,1),(1,0):
        if pix_position[i] < 0:
            extra = int(np.ceil(np.abs(pix_position[i])))
            padding = [extra, extra]
            padding[j] = obj.shape[j]
            
            obj = np.concatenate([np.zeros(padding, dtype=obj.dtype),
                                  obj],
                                 axis=i)
            weights = np.concatenate([np.zeros(padding, dtype=weights.dtype),
                                weights],
                               axis=i)
            pix_offset = t.as_tensor(np.array([0,0]), dtype=t_basis.dtype)
            pix_offset[i] = -extra

            offset += interactions.pixel_to_translations(t_basis, pix_offset).numpy()[:2]
            pix_position[i] += extra

        elif (pix_position[i] + new_obj.shape[i]) > obj.shape[i]:

            extra = int(np.ceil((pix_position[i] + new_obj.shape[i])
                                -obj.shape[i]))
            padding = [extra, extra]
            padding[j] = obj.shape[j]
            
            obj = np.concatenate([obj,
                                  np.zeros(padding, dtype=obj.dtype)],
                                 axis=i)
            weights = np.concatenate([weights,
                                      np.zeros(padding, dtype=weights.dtype)],
                           axis=i)

    sl = np.s_[int(np.round(pix_position[0])):
               int(np.round(pix_position[0]))+new_obj.shape[0],
               int(np.round(pix_position[1])):
               int(np.round(pix_position[1]))+new_obj.shape[1]]
               
    obj[sl] = (obj[sl] * weights[sl] + np.abs(new_obj) * new_weights) \
        / (weights[sl] + new_weights)
    weights[sl] += new_weights
    
    return obj, weights, offset



def main(argv=sys.argv):

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
    pub = context.socket(zmq.PUB)
    pub.bind(config['stitched_frame_publish_port'])
    
    print('Listening on', config['rpi_frame_subscription_port'])
    print('Publishing on', config['stitched_frame_publish_port'])
    
    started = False
    while True:
        message = sub.recv_pyobj()
        if message['event'] == 'start':
            started = True
            print('Starting new synthesis')
            obj = None
            weights = None
            offset = None

        elif message['event'] == 'stop':
            print('Finished synthesis')
            started = False
            
        elif started and message['event'] == 'frame':
            print('Accepting a new frame')
            new_obj = message['data']
            new_weights = message['weights']
            position = message['position']
            print(position)
            # TODO: What if the basis changes?
            basis = message['basis']
            obj, weights, offset = add_frame_to_object(obj,
                                                       weights,
                                                       basis,
                                                       offset,
                                                       new_obj,
                                                       new_weights,
                                                       position)
            event = {'event':'frame',
                     'data':obj}
            pub.send_pyobj(event)

        else:
            print(message['event'])
            print('Rejected an event')

if __name__ == '__main__':
    sys.exit(main())
