import torch as t
import time

def RPI_interaction(probe, obj):
    """Returns an exit wave from a high-res probe and a low-res obj

    In this interaction, the probe and object arrays are assumed to cover
    the same physical region of space, but with the probe array sampling that
    region of space more finely. Thus, to do the interaction, the object
    is first upsampled by padding it in Fourier space (equivalent to a sinc
    interpolation) before being multiplied with the probe. This function is
    called RPI_interaction because this interaction is central to the RPI
    method and is not commonly used elsewhere.

    This also works with object functions that have an extra first dimension
    for an incoherently mixing model.


    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for simulating the exit waves
    obj : torch.Tensor
        An M'xL' or NxM'xL' object function for simulating the exit waves

    Returns
    -------
    exit_wave : torch.Tensor
        An MxL tensor of the calculated exit waves
    """

    # The far-field propagator is just a 2D FFT but with an fftshift
    fftobj = t.fft.fftshift(t.fft.fft2(obj, norm='ortho'), dim=(-1,-2))

    # We calculate the padding that we need to do the upsampling
    # This is carefully set up to keep the zero-frequency pixel in the correct
    # location as the overall shape changes. Don't mess with this without
    # having thought about this carefully.
    pad2l = probe.shape[-2]//2 - obj.shape[-2]//2
    pad2r = probe.shape[-2] - obj.shape[-2] - pad2l
    pad1l = probe.shape[-1]//2 - obj.shape[-1]//2
    pad1r = probe.shape[-1] - obj.shape[-1] - pad1l
    
    fftobj = t.nn.functional.pad(fftobj, (pad1l, pad1r, pad2l, pad2r))
    
    # Again, just an inverse FFT but with an fftshift
    upsampled_obj = t.fft.ifft2(t.fft.ifftshift(fftobj, dim=(-1,-2)),
                               norm='ortho')

    return probe * upsampled_obj


def forward(obj, probe):
    """Simulates the wavefield at the detector plane from the probe and obj

    For speed reasons, this forward model does not implement fftshifts in
    the fft, so the output patterns have the zero frequency pixel in the
    corner. Honestly, it's surprising how long the fftshift takes.

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for simulating the exit waves
    obj : torch.Tensor
        An M'xL' or NxM'xL' object function for simulating the exit waves

    Returns
    -------
    wavefield : torch.Tensor
        An MxL or NxMxL tensor of the calculated detector-plane wavefields
    """
    ew = RPI_interaction(probe, obj)
    diff = t.fft.fft2(ew, norm='ortho')
    return diff


def iterate_CG(rec):
    """ This runs a single iteration of a CG reconstruction on the special
    reconstruction dictionary used by the live RPI reconstruction service
    """

    # TODO: Right now, this only works with the top probe mode
    # We start by zeroing the gradients
    rec['obj'].grad = None
    
    # This chunk runs the simulation and gets the gradients
    diff = forward(rec['obj'], rec['probe'][0])
    mag_diff = t.abs(diff)
    error_pattern = mag_diff - rec['sqrtPattern']
    if rec['mask'] is not None:
        error_pattern = error_pattern * rec['mask'].unsqueeze(0)

    error = t.sum(error_pattern**2)
    error.backward()
    grad = rec['obj'].grad.detach()

    # Here we calculate the CG step direction
    if rec['iter'] % rec['clearEvery'] == 0:
        rec['last_grad'] = None
        rec['last_step_dir'] = None
        step_dir = grad
    else:
        # This is Fletcher-Reeves
        grad_sum = t.sum(t.abs(grad)**2, dim=(-1,-2))
        last_grad_sum = t.sum(t.abs(rec['last_grad'])**2, dim=(-1,-2))
        beta = grad_sum/last_grad_sum
        
        # This is Polak-Ribiere - doesn't seem to work as well
        #numerator = t.sum(grad.conj() * (grad - last_grad)).real
        #numerator = t.clamp(numerator, min=0)
        #last_grad_sum = t.sum(t.abs(last_grad)**2, dim=(-1,-2))
        #beta = numerator/last_grad_sum
        
        step_dir = grad + beta[:,None,None] * rec['last_step_dir']
    
    rec['last_grad']=grad

    # This calculates an optimal step size, assuming that the step
    # remains small compared to the original object.
    grad_pat = forward(step_dir, rec['probe'][0])
    A = error_pattern.detach()
    B = t.real(diff.detach().conj()  * grad_pat) / (mag_diff.detach())
    if rec['mask'] is not None:
        B = B * rec['mask'].unsqueeze(0)
    
    alpha = -t.sum(A*B, dim=(-1,-2)) / t.sum(B**2, dim=(-1,-2))

    # Here we actually perform the update
    rec['last_step_dir'] = step_dir
    rec['obj'].data += alpha[:,None,None] * step_dir
    rec['iter'] += 1

def run_CG(n_iters, obj, probe, pat, mask=None, clear_every=10):
    """Runs a conjugate gradient based RPI algorithm

    This algorithm is tuned for speed, the main consequence of that being
    that it only accepts a single probe mode. Additionally, it doesn't allow
    for a background model (so, subtract a background before using)

    The algorithm uses the following approach:

    * The update direction is chosen using the Fletcher-Reeves formula 
        for beta.
    * The step size is chosen using an analytical formula to minimize the
        MSE between the simulated and measured magnitudes, assuming that
        the step is a small perturbation.

    Note that this expects to run in a batch form, with a stack of objects and
    a stack of patterns. All the tensors should be on the same device and
    have compatible dtypes.

    Parameters
    ----------
    n_iters : int
        The number of iterations to run
    obj : torch.Tensor
        An NxM'xL' initial guess of the object function
    probe : torch.Tensor
        An MxL probe function
    pat : torch.Tensor
        An NxMxL stack of patterns to reconstruct
    mask : torch.Tensor
        Optional, a boolean mask set to "True" for detecto pixels to be included
    clear_every : int
        Default is 10, reset the CG directions every <clear_every> iterations
    
    Returns
    -------
    obj : torch.Tensor
        An NxM'xL' tensor of the reconstructed objects
    """

    # FFTshifting the pattern once actually saves a lot of time compared
    # to fftshifting the wavefields at each iteration.
    pat = t.fft.ifftshift(pat, dim=(-1,-2))

    # We update this object internally as the iterative algorithm progresses
    temp_obj = t.empty_like(obj)
    temp_obj.data = obj
    temp_obj.requires_grad = True

    # Get the pattern's magnitudes once before starting the loop
    sqrt_pat = t.sqrt(pat)
    
    for i in range(n_iters):

        # We start by zeroing the gradients
        temp_obj.grad = None

        # This chunk runs the simulation and gets the gradients
        diff = forward(temp_obj, probe)
        mag_diff = t.abs(diff)
        error_pattern = mag_diff - sqrt_pat
        if mask is not None:
            error_pattern = error_pattern * mask.unsqueeze(0)
        error = t.sum(error_pattern**2)
        error.backward()
        grad = temp_obj.grad.detach()

        # Here we calculate the CG step direction
        if i % clear_every == 0:
            last_grad = None
            last_step_dir = None
            step_dir = grad
        else:
            # This is Fletcher-Reeves
            grad_sum = t.sum(t.abs(grad)**2, dim=(-1,-2))
            last_grad_sum = t.sum(t.abs(last_grad)**2, dim=(-1,-2))
            beta = grad_sum/last_grad_sum

            # This is Polak-Ribiere - doesn't seem to work as well
            #numerator = t.sum(grad.conj() * (grad - last_grad)).real
            #numerator = t.clamp(numerator, min=0)
            #last_grad_sum = t.sum(t.abs(last_grad)**2, dim=(-1,-2))
            #beta = numerator/last_grad_sum

            step_dir = grad + beta[:,None,None] * last_step_dir

        last_grad=grad

        
        # This calculates an optimal step size, assuming that the step
        # remains small compared to the original object.
        grad_pat = forward(step_dir, probe)
        A = error_pattern.detach()
        B = t.real(diff.detach().conj()  * grad_pat) / (mag_diff.detach())
        if mask is not None:
            B = B * mask.unsqueeze(0)
        alpha = -t.sum(A*B, dim=(-1,-2)) / t.sum(B**2, dim=(-1,-2))

        # Here we actually perform the update
        last_step_dir = step_dir
        temp_obj.data += alpha[:,None,None] * step_dir

    return temp_obj.data


def run_DM(n_iters, obj, probe, pat, mask=None, beta=1):
    """Runs a difference map based RPI algorithm

    This algorithm is tuned for speed, the main consequence of that being
    that it only accepts a single probe mode. Additionally, it doesn't allow
    for a background model (so, subtract a background before using)

    The algorithm is a straight implementation of the difference map
    algorithm, with the two projection operators being the measured
    Fourier magnitudes and a support constraint applied to the object
    in Fourier space. The object is returned in low resolution (after
    application of the Fourier space constraint).

    Note that this expects to run in a batch form, with a stack of objects and
    a stack of patterns. All the tensors should be on the same device and
    have compatible dtypes.

    Parameters
    ----------
    n_iters : int
        The number of iterations to run
    obj : torch.Tensor
        An NxM'xL' initial guess of the object function
    probe : torch.Tensor
        An MxL probe function
    pat : torch.Tensor
        An NxMxL stack of patterns to reconstruct
    mask : torch.Tensor
        Optional, a boolean mask set to "True" for detecto pixels to be included
    beta : float
        Default is 1, and doesn't currently allow beta other than 1
    
    Returns
    -------
    obj : torch.Tensor
        An NxM'xL' tensor of the reconstructed objects
    """

    padding_left = (np.array(obj.shape[-2:])+1)//2
    padding_right = np.array(obj.shape[-2:]) - np.array(obj.shape[-2:])//2

    # FFTshifting the pattern once actually saves a lot of time compared
    # to fftshifting the wavefields at each iteration.
    pat = t.fft.ifftshift(pat, dim=(-1,-2))
    sqrt_pat = t.sqrt(pat)
    
    inverse_probe = t.conj(probe) / (t.abs(probe)**2 + 0.00001 * t.max(t.abs(probe))**2)
    def project_Fourier_support(guess_obj):
        # Note: This will fail for 1-pixel objects, when padding-right
        # goes to 0. I think we'll live
        guess_obj_Fourier = t.fft.fft2(guess_obj, norm='ortho')
        guess_obj_Fourier[...,padding_left[0]:-padding_right[0],:] = 0
        guess_obj_Fourier[...,:,padding_left[1]:-padding_right[1]] = 0
        proj_obj = t.fft.ifft2(guess_obj_Fourier, norm='ortho')
        return proj_obj

    def project_magnitudes(guess_obj):
        ew = guess_obj * probe
        far_field = t.fft.fft2(ew, norm='ortho')
        far_field = t.exp(1j*t.angle(far_field)) * sqrt_pat
        ew = t.fft.ifft2(far_field, norm='ortho')
        return inverse_probe * ew
        
    # we need a temporary version of the obj in full resolution to
    # iterate over
    temp_obj = RPI_interaction(t.ones_like(probe),obj)    

    for i in range(n_iters):
        temp_obj = project_magnitudes(project_Fourier_support(temp_obj))
        term_2 = project_Fourier_support(temp_obj)
        term_1 = project_magnitudes(2*term_2 - temp_obj)
        temp_obj += term_1 - term_2

    guess_obj_Fourier = t.fft.fft2(temp_obj, norm='ortho')
    rolled = t.roll(guess_obj_Fourier, tuple(padding_right), dims=(-2,-1))
    low_res_Fourier = rolled[...,:obj.shape[-2],:obj.shape[-1]]
    unrolled = t.roll(low_res_Fourier, tuple(-padding_right), dims=(-2,-1))
    return t.fft.ifft2(unrolled, norm='ortho')

    
if __name__ == '__main__':

    # Here we simulate some data using the probe defined in the calibration
    # file, and we then check to see how long the reconstruction takes.
    
    from scipy.io import loadmat, savemat
    import numpy as np
    import time
    from matplotlib import pyplot as plt
    
    calibration = loadmat('calibration.mat')
    options = loadmat('options.mat')
    
    probe = t.as_tensor(calibration['probe'][0], dtype=t.complex64)
    
    eps = 0.1
    
    n_objs = 100
    shape = [256,256]
    obj_slice = np.s_[...,500-shape[0]//2:500+shape[0]//2,
                      500-shape[0]//2:500+shape[0]//2]
    
    uniform = np.ones([n_objs]+shape,dtype=np.complex64)
    
    test_objs = uniform*1.0 + eps / np.sqrt(2) * \
        (np.random.randn(n_objs, *shape) + 1j * np.random.randn(n_objs, *shape))
    
    test_objs = t.as_tensor(test_objs, dtype=t.complex64)
    uniform = t.as_tensor(uniform, dtype=t.complex64)
    
    true_pats = t.fft.fftshift(t.abs(forward(test_objs.detach(), probe))**2,
                               dim=(-1,-2))
    
    dev = 'cuda:0'
    uniform = uniform.to(device=dev)
    probe = probe.to(device=dev)
    true_pats = true_pats.to(device=dev)
    
    n_iters = 10

    # Test it with a mask
    # mask = t.ones_like(probe, dtype=t.bool)
    # mask[0:500,0:500] = False

    # Test it without a mask
    mask = None

    start_time = time.time()    
    rec_objs = run_CG(n_iters, uniform, probe, true_pats,
                      mask=mask, clear_every=10)
    #rec_objs = run_DM(n_iters, uniform, probe, true_pats,
    #                  mask=mask)
    t.cuda.synchronize()
    print(n_iters, 'iterations run on', n_objs, 'objects in',
          (time.time() - start_time), 'seconds')
    print((time.time() - start_time)/(n_iters*n_objs),
          'seconds per iteration per object')
    
    rec_objs = rec_objs.cpu()
    
    gammas = t.angle(t.sum(rec_objs.conj()*test_objs,
                           dim=(-1,-2)))
    rec_objs = t.exp(1j*gammas)[:,None,None]*rec_objs

    plt.figure()
    plt.imshow(true_pats[0].detach().cpu().numpy())
    plt.colorbar()
    plt.title('Simulated pattern')
    plt.figure()
    plt.imshow(np.abs(rec_objs[0].numpy()))
    plt.colorbar()
    plt.title('Reconstructed object')
    plt.figure()
    plt.imshow(np.abs((rec_objs[0]-test_objs[0]).numpy()))
    plt.colorbar()
    plt.title('Pixel by pixel error')
    plt.show()
