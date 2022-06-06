import zmq
import time
import torch as t
import numpy as np
import CDTools
from CDTools.tools import propagators as prop
from CDTools.tools import interactions
from CDTools.tools import plotting as p
from scipy.io import loadmat, savemat
from matplotlib import pyplot as plt
from scipy.sparse import linalg as spla


calibration = loadmat('calibration.mat')
options = loadmat('options.mat')

model = CDTools.models.RPI.from_calibration(calibration, [256,256])


probe = model.probe.detach()[0]

eps = 0.1

shape = [128,128]
#shape = [256,256]
#shape = [512,512]
obj_slice = np.s_[500-shape[0]//2:500+shape[0]//2,
                  500-shape[0]//2:500+shape[0]//2]

uniform = np.ones(shape,dtype=np.complex64) 
obj = uniform*1.0 + eps / np.sqrt(2) * \
    (np.random.randn(*shape) + 1j * np.random.randn(*shape))

obj = t.as_tensor(obj, dtype=t.complex64)
obj.requires_grad=True
uniform = t.as_tensor(uniform, dtype=t.complex64)

low_res_probe = prop.inverse_far_field(prop.far_field(t.abs(probe)**2)[obj_slice])
low_res_probe = t.abs(low_res_probe) / t.max(t.abs(low_res_probe))
#p.plot_amplitude(low_res_probe)
#plt.show()

def forward(obj, probe):
    ew = interactions.RPI_interaction(probe, obj)
    diff = prop.far_field(ew)
    return diff



weiner_probe_inverse = probe.conj() / (np.abs(probe)**2 + 0.00)#1)
#probe_mag = np.abs(probe)**2


def backward(wavefield, probe_inverse, obj_slice, e=0.00):
    nf = prop.inverse_far_field(wavefield)
    hr_obj = nf * probe_inverse
    obj = prop.inverse_far_field(prop.far_field(hr_obj)[obj_slice])
    return obj


def calculate_best_step(obj, grad, probe, pat):
    obj_pat = forward(obj, probe)
    grad_pat = forward(grad, probe)

    A = t.abs(obj_pat) - t.sqrt(pat)
    B = t.real(obj_pat.conj()  * grad_pat) / (t.abs(obj_pat))

    alpha = -t.sum(A*B) / t.sum(B**2)

    return alpha

def iterate(obj, probe, pat):
    temp_obj = t.empty_like(obj)
    temp_obj.data = obj
    temp_obj.requires_grad = True

    sqrt_pat = t.sqrt(pat)
    diff = forward(temp_obj, probe)
    mag_diff = t.abs(diff)

    error = t.sum((sqrt_pat - mag_diff)**2)
    error.backward()

    
    #alpha = calculate_best_step(temp_obj.data, temp_obj.grad, probe, pat)
    grad_pat = forward(temp_obj.grad, probe)
    A = mag_diff - sqrt_pat
    B = t.real(diff.conj()  * grad_pat) / (mag_diff)

    alpha = -t.sum(A*B) / t.sum(B**2)

    return (temp_obj.data + alpha * temp_obj.grad).detach()


def iterate_least_squares(obj, probe, pat):

    def vectorify(im):
        return np.hstack([im.real.cpu().numpy().ravel(),
                          im.imag.cpu().numpy().ravel()])

    def devectorify(vec):
        real = t.as_tensor(vec[:len(vec)//2].reshape(shape), device=obj.device)
        imag = t.as_tensor(vec[len(vec)//2:].reshape(shape), device=obj.device)
        return real + 1j * imag
    
    def calc_difference(delta_obj):
        return (forward(obj, probe).conj() * forward(delta_obj, probe)).real
    
    def calc_difference_transpose(delta_pattern):
        wavefield = forward(obj, probe) * delta_pattern
        exit_wave = prop.inverse_far_field(wavefield)
        hr_obj = probe.conj() * exit_wave
        return prop.inverse_far_field(prop.far_field(hr_obj)[obj_slice])
    
    def matvec(vec):
        
        return calc_difference(devectorify(vec)).cpu().numpy().ravel()
    
    def rmatvec(difference):
        reshaped_difference = t.as_tensor(difference.reshape(probe.shape),
                                          device=obj.device)
        return vectorify(calc_difference_transpose(reshaped_difference))
    full_size = 1000*1000
    obj_size = shape[0]*shape[1]
    
    op = spla.LinearOperator(shape=(full_size, obj_size*2),
                             matvec = matvec, rmatvec=rmatvec)

    sqrt_pat = t.sqrt(pat)
    diff = forward(obj, probe)
    mag_diff = t.abs(diff)

    dpattern = (-mag_diff+sqrt_pat).cpu().numpy()

    grad = devectorify(spla.lsqr(op,dpattern.ravel())[0])

    step = calculate_best_step(obj, grad, probe, pat)
    print(step)    
    return obj + step * grad

    

def iterate_least_squares_restricted(obj, probe, pat):
    smaller_size = 1000*1000//16
    args = np.argsort(pat.cpu().numpy().ravel())[-smaller_size:]
    

    def vectorify(im):
        return np.hstack([im.real.cpu().numpy().ravel(),
                          im.imag.cpu().numpy().ravel()])

    def devectorify(vec):
        real = t.as_tensor(vec[:len(vec)//2].reshape(shape), device=obj.device)
        imag = t.as_tensor(vec[len(vec)//2:].reshape(shape), device=obj.device)
        return real + 1j * imag
    
    def calc_difference(delta_obj):
        return (forward(obj, probe).conj() * forward(delta_obj, probe)).real
    
    def calc_difference_transpose(delta_pattern):
        wavefield = forward(obj, probe) * delta_pattern
        exit_wave = prop.inverse_far_field(wavefield)
        hr_obj = probe.conj() * exit_wave
        return prop.inverse_far_field(prop.far_field(hr_obj)[obj_slice])
    
    def matvec(vec):
        diff = calc_difference(devectorify(vec)).cpu().numpy().ravel()
        return diff[args]
    
    def rmatvec(difference):
        diff = np.zeros(probe.shape)
        diff.ravel()[args] = difference
        reshaped_difference = t.as_tensor(diff,
                                          device=obj.device)
        return vectorify(calc_difference_transpose(reshaped_difference))
    
    full_size = 1000*1000
    obj_size = shape[0]*shape[1]
    
    op = spla.LinearOperator(shape=(smaller_size, obj_size*2),
                             matvec = matvec, rmatvec=rmatvec)

    sqrt_pat = t.sqrt(pat)
    diff = forward(obj, probe)
    mag_diff = t.abs(diff)

    dpattern = (-mag_diff+sqrt_pat).cpu().numpy().ravel()[args]

    grad = devectorify(spla.lsqr(op,dpattern)[0])

    step = calculate_best_step(obj, grad, probe, pat)
    print(step)
    
    return obj + step * grad
    



def iterate_shitty_least_squares(obj, probe, pat):
    def operator(delta_obj):
        return (forward(obj, probe).conj() * forward(delta_obj, probe)).real

    def transpose_operator(delta_pattern):
        wavefield = forward(obj, probe) * delta_pattern
        exit_wave = prop.inverse_far_field(wavefield)
        hr_obj = probe.conj() * exit_wave
        return prop.inverse_far_field(prop.far_field(hr_obj)[obj_slice])

    test = t.zeros_like(obj)
    test[100,100] = 1
    estimate_of_mat = t.abs(transpose_operator(operator(t.ones_like(obj))))
    
    sqrt_pat = t.sqrt(pat)
    diff = forward(obj, probe)
    mag_diff = t.abs(diff)

    estimate = transpose_operator(-mag_diff+sqrt_pat)

    estimate_of_mat_inv = estimate_of_mat / (estimate_of_mat**2 + 0.01*t.max(estimate_of_mat**2))

    grad = estimate * estimate_of_mat_inv
    
    step = calculate_best_step(obj, grad, probe, pat)
    #p.plot_real(estimate_of_mat)
    #p.plot_imag(estimate_of_mat)
    #plt.show()
    return obj + step * grad

def run_CG(n_iters, obj, probe, pat, clear_every=10):
    temp_obj = t.empty_like(obj)
    temp_obj.data = obj
    temp_obj.requires_grad = True

    objs = []
    sqrt_pat = t.sqrt(pat)
    for i in range(n_iters):
        temp_obj.grad = None
        if i % clear_every == 0:
            last_grad_sum = None
            last_step_dir = None

        diff = forward(temp_obj, probe)
        mag_diff = t.abs(diff)
        
        error = t.sum((sqrt_pat - mag_diff)**2)
        error.backward()
        
        grad = temp_obj.grad 
        grad_sum = t.sum(t.abs(grad)**2)
        if last_step_dir is None:
            step_dir = grad
        else:
            beta = grad_sum/last_grad_sum

            step_dir = grad + beta * last_step_dir
        last_grad_sum = grad_sum
        
        grad_pat = forward(step_dir, probe) 
        A = mag_diff - sqrt_pat
        B = t.real(diff.conj()  * grad_pat) / (mag_diff)
        
        alpha = -t.sum(A*B) / t.sum(B**2)
        print(alpha)
        print(type(alpha))
        last_step_dir = step_dir
        temp_obj.data = temp_obj.data + alpha * step_dir
        objs.append(temp_obj.data)
    return objs


# has problems with LBFGS not knowing the actual step that was taken
def run_LBFGS(n_iter, obj, probe, pat):
    temp_obj_real = t.empty_like(obj, dtype=t.float32)
    temp_obj_imag = t.empty_like(obj, dtype=t.float32)    
    temp_obj_real.data = obj.real
    temp_obj_imag.data = obj.imag

    temp_obj_real.requires_grad = True
    temp_obj_imag.requires_grad = True

    
    opt = t.optim.LBFGS([temp_obj_real, temp_obj_imag],history_size=1, max_iter=1,
                        line_search_fn=None)
    objs = [obj.clone()]
    sqrt_pat = t.sqrt(pat)
    for i in range(n_iter):
        temp_obj = temp_obj_real + 1j * temp_obj_imag
        def closure():
            opt.zero_grad()
            diff = forward(temp_obj, probe)
            mag_diff = t.abs(diff)
            
            error = t.sum((sqrt_pat - mag_diff)**2)
            error.backward()
            return error

        
        old_obj = temp_obj.data.clone()
        diff = forward(old_obj, probe)
        mag_diff = t.abs(diff)
        opt.step(closure)
                
        temp_obj = temp_obj_real + 1j * temp_obj_imag

        objs.append(temp_obj.clone())
        continue
        
        grad = temp_obj.data - old_obj
        #print(grad)
        
    
        #alpha = calculate_best_step(temp_obj.data, temp_obj.grad, probe, pat)
        grad_pat = forward(grad, probe)
        A = mag_diff - sqrt_pat
        B = t.real(diff.conj()  * grad_pat) / (mag_diff)

        alpha = -t.sum(A*B) / t.sum(B**2)
        temp_obj.data = old_obj + alpha * grad
        objs.append(temp_obj.clone())
    return objs
    

true_pat = t.abs(forward(obj.detach(), probe))**2

dev = 'cuda:0'
uniform = uniform.to(device=dev)
probe = probe.to(device=dev)
true_pat = true_pat.to(device=dev)


st = time.time()
#objs = run_LBFGS(100, uniform, probe, true_pat)
objs = run_CG(5, uniform, probe, true_pat)

#objs = [uniform]

#for i in range(50):
#    print(i)
#    #objs.append(iterate(objs[-1], probe, true_pat))
#    objs.append(iterate_shitty_least_squares(objs[-1], probe, true_pat))
#    #objs.append(iterate_least_squares(objs[-1], probe, true_pat))
#    objs.append(iterate_least_squares_restricted(objs[-1], probe, true_pat))
print((time.time() - st)/100)
    
objs = t.stack(objs).to(device='cpu')
gamma = t.angle(t.mean(objs[-1][50:-50].conj()*obj[50:-50]))
objs = t.exp(1j*gamma)*objs


p.plot_amplitude(objs)
p.plot_amplitude(objs-obj)
plt.show()
