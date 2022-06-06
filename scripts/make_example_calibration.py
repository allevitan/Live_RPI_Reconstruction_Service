import CDTools
import pickle
from calibration import make_calibration_from_ptycho
from scipy.io import loadmat, savemat

dataset = '/home/abe/Old Dropbox (MIT)/Old Projects/RPI_Code_Optics_Express/data/Optical_Data_ptycho.cxi'

results = '/home/abe/Old Dropbox (MIT)/Old Projects/RPI_Code_Optics_Express/data/Optical_ptycho.pickle'

with open(results, 'rb') as f:
    results = pickle.load(f)

dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(dataset)
calibration = make_calibration_from_ptycho(dataset, results)

savemat('calibration.mat', calibration)

options = {'resolution': 256, 'iterations':10}
savemat('options.mat', options)



