"""Runs a sweep over a defocus region using conjugate gradient RPI for speed.

The way that I want this to work is:

* I run a command line script which identifies a probe calibration file,
   a data file and index within the file, and a range of defocuses (or a
   range of energies?), along with a pixel count and the # of iterations
* Then, the program runs CG-based reconstructions at all of the specified
   defocuses, producing a cube of data
* Then, I pop up a widget which lets me scroll through the reconstructed data
   and plop down a line, which will then produce a second plot which is a
   line cut through the cube showing me where the best focus is.

"""
import cdtools
from cdtools.tools import plotting as p
from cdtools.tools.propagators import generate_angular_spectrum_propagator as gasp
from cdtools.tools.propagators import near_field
from matplotlib import pyplot as plt
import torch as t
import numpy as np
from scipy import io
from live_rpi_reconstruction_service.conjugate_gradient_rpi import iterate_CG, downsample_to_shape


# First things first, I need to actually do the damn reconstructions.


def focus_scan(rec, spacing, wavelength, defocuses):
    """ Makes a stack of reconstructions at the defocus distances specified


    The rec object is the reconstruction dictionary used by the live RPI
    reconstruction service, and defocuses is a list or array of distances to
    defocus the beam by.
    """
    output = t.empty([len(defocuses),] + list(rec['obj'].shape),
                     dtype=rec['obj'].dtype, device=rec['obj'].device)
    for idx, defocus in enumerate(defocuses):
        asp = gasp(probe.shape[-2:], spacing, wavelength, defocus,
                   device=probe.device)
        
        print('Defocus',idx,'of',len(defocuses), '      ',end='\r')
        # shallow copy the reconstruction object and then clone the
        # probe and object, since we will be mutating them.
        this_rec = rec.copy()
        this_rec['probe'] = near_field(rec['probe'], asp)
        this_rec['obj'] = this_rec['obj'].detach().clone()
        this_rec['obj'].requires_grad=True
        # Now we defocus the probe

        while this_rec['iter'] < this_rec['nIterations']:
            iterate_CG(this_rec)
        output[idx] = this_rec['obj']

    print('Done calculating                                         ')
    return output


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Reconstruct a focus sweep to identify the best focus')

    parser.add_argument('data_file', type=str, help='the location of the .cxi file containing the data')
    parser.add_argument('calibration_file', type=str, help='the location of the .mat calibration file containing the probe')
    parser.add_argument('defocus_start', type=float, help='the minimum focal position to use')
    parser.add_argument('defocus_stop', type=float, help='The maximum focal position to use')
    parser.add_argument('steps', type=int, help='The number of steps to reconsttruct')
    parser.add_argument('pixels', type=int, help='The number of pixels across the object')
    parser.add_argument('iterations', type=int, help='The number of iterations to run')
    parser.add_argument('--cpu', action='store_true', help='Run everything on the cpu')
    
    args = parser.parse_args()
    
    if args.cpu:
        device = 'cpu'
    else:
        # TODO: offer more flexibility in the GPU choice
        device='cuda:0'
        
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(args.data_file)
    dataset.translations = dataset.translations[:len(dataset)]
    mask = io.loadmat('/home/abe/Dropbox (MIT)/Photon Scattering Group/Data/Experiments/Synchro/ALS/20220921_RZP_4_Test/default_mask.mat')
    dataset.patterns = dataset.patterns * (1-mask['mask'])


    calibration = io.loadmat(args.calibration_file)
    
    dataset.inspect()
    plt.show()

    frame = int(input('Which frame would you like to use? '))

    pattern = dataset.patterns[frame].to(device=device)
    
    probe = t.as_tensor(calibration['probe'], device=device)
    
    if 'oversampling' in calibration:
        oversampling = calibration['oversampling']
    else:
        oversampling=2
    if 'mask' in calibration:
        mask = calibration['mask']
    else:
        mask = t.ones_like(pattern)

    if 'background' in calibration:
        background = t.as_tensor(calibration['background'], device=device)
    else:
        background = t.zeros_like(pattern)

    sqrtPattern = t.sqrt(t.clamp(pattern - background, min=0))
    sqrtPattern = t.fft.ifftshift(sqrtPattern, dim=(-1,-2))

    obj = t.ones([1,args.pixels, args.pixels],
                  dtype=t.complex64, device=device,
                  requires_grad=True)
    
    rec= {'probe': probe,
          'oversampling': oversampling,
          'sqrtPattern': sqrtPattern,
          'obj': obj,
          'mask': mask,
          'iter': 0,
          'nIterations': args.iterations,
          'clearEvery': 10}

    wavelength = calibration['wavelength'].ravel()[0]
    spacing = [np.abs(calibration['basis'][0,1])]*2
    
    defocuses = t.linspace(args.defocus_start, args.defocus_stop, args.steps)
    defocuses = defocuses * 1e-6 # convert to m from um
    
    focus_stack = focus_scan(rec, spacing, wavelength, defocuses)
    
    p.plot_amplitude(focus_stack)
    p.plot_phase(focus_stack)
    plt.show()
