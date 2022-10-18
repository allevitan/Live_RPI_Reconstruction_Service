import torch as t

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


def downsample_to_shape(wavefield, shape):
    # The far-field propagator is just a 2D FFT but with an fftshift
    fftobj = t.fft.fftshift(t.fft.fft2(wavefield, norm='ortho'), dim=(-1,-2))
    pad2l =  wavefield.shape[-2]//2 - shape[-2]//2
    pad2r = wavefield.shape[-2] - shape[-2] - pad2l
    pad1l = wavefield.shape[-1]//2 - shape[-1]//2
    pad1r = wavefield.shape[-1] - shape[-1] - pad1l

    fftobj = fftobj[..., pad2l:-pad2r,pad1l:-pad1r]

    downsampled_obj = t.fft.ifft2(t.fft.ifftshift(fftobj, dim=(-1,-2)),
                                  norm='ortho')
    return downsampled_obj

    
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


#@t.jit.script
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
    if 'mask' in rec and rec['mask'] is not None:
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
    # Now we calculate an optimal step size, assuming that the step
    # remains small compared to the original object.
    grad_pat = forward(step_dir, rec['probe'][0])
    A = error_pattern.detach()
    B = t.real(diff.detach().conj()  * grad_pat) / (mag_diff.detach())
    if 'mask' in rec and rec['mask'] is not None:
        B = B * rec['mask'].unsqueeze(0)
    
    alpha = -t.sum(A*B, dim=(-1,-2)) / t.sum(B**2, dim=(-1,-2))

    # Here we actually perform the update
    rec['last_step_dir'] = step_dir
    rec['obj'].data += alpha[:,None,None] * step_dir
    rec['iter'] += 1

