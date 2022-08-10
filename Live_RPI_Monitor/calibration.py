import h5py
import CDTools

#
# How should the calibration file be laid out?
#
# Calibration:
# 
# probe
# background
# detector mask
# probe energy
# probe basis


def make_calibration_from_ptycho(dataset, results):
    calibration = {}
    calibration['probe'] = results['probe']
    calibration['mask'] = dataset.mask.numpy()
    calibration['background'] = results['background']
    calibration['wavelength'] = dataset.wavelength
    calibration['basis'] = results['basis']
    return calibration
    
