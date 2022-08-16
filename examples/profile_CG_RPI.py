import torch as t
from live_rpi_reconstruction_service.conjugate_gradient_rpi import iterate_CG
from scipy.io import loadmat
import time

# First we load some test data
calibration = loadmat('calibration.mat')
data = loadmat('test_data.mat')

# Now we set up a reconstruction dictionary, which is the object that the
# iterate_CG function works with. It has the current state of the reconstruction
# as well as the required calibration information

device = 'cuda:0' # change to change what device we're testing

pattern = t.as_tensor(data['pattern'],
                      dtype=t.float32, device=device)

# We pre-fftshift the patterns for speed of analysis
sqrtPattern = t.fft.ifftshift(t.sqrt(t.clamp(pattern, min=0)), dim=(-1,-2))

rec = {
    'obj': t.ones((1,256,256),
                  dtype=t.complex64, device=device, requires_grad=True),
    'probe': t.as_tensor(calibration['probe'],
                         dtype=t.complex64, device=device),
    'sqrtPattern': sqrtPattern,
    'clearEvery': 10,
    'iter': 0
}
# And we run a few iterations
iterations = 10

# we do a few iterations at the start to let pytorch do it's mysterious magic
for idx in range(3):
    iterate_CG(rec)


# and now we time the per-iteration speed
t.cuda.synchronize()
t0 = time.time()
for idx in range(iterations):
    iterate_CG(rec)

t.cuda.synchronize()
print('Total time per iteration:', (time.time() - t0)/iterations)


plot = False
if plot:
    obj = rec['obj'].detach().cpu()

    from matplotlib import pyplot as plt
    plt.imshow(t.abs(obj[0]))
    plt.colorbar()
    plt.figure()
    plt.imshow(t.abs(pattern.cpu()))
    plt.colorbar()
    plt.figure()
    plt.imshow(t.abs(rec['probe'][0].cpu()))
    plt.colorbar()
    plt.show()
