import numpy as np
import theano
import theano.tensor as T

def ferromagnet_free_energy(r, u, H):
    # variables
    m = T.dscalar('m')

    # free energy and its gradient
    F = 0.5*r*m*m+0.25*u*m*m*m*m-H*m
    grad_m = T.grad(F, m)
    free_energy = theano.function(inputs=[m], outputs=F)
    free_energy_grad = theano.function(inputs=[m], outputs=grad_m)

    # minimization
    # nsteps = T.iscalar('nsteps')
    # init_m = T.dscalar('init_m')
    # m_seq = T.dscalars('m_seq')
    # grad_results, grad_updates = theano.scan(fn=lambda vars: vars - 0.1*grad_m,
    #                                          outputs_info=[{'initial': init_m}],
    #                                          sequences=m_seq,
    #                                          n_steps=nsteps)
    # final_m = grad_results[-1]
    # cal_m = theano.function(inputs=[init_m, nsteps], outputs=final_m, updates=grad_updates)

    # return free_energy, [m], free_energy_grad, cal_m
    return free_energy, [m], free_energy_grad

def ferromagnet_grad_descent(free_energy_grad, init_m, learning_rate=0.1, tol=1e-16, max_iter=10000):
    update = lambda mval: mval - learning_rate*free_energy_grad(mval)

    diff = 1e+16
    step = 0
    current_mval = init_m
    while diff > tol and step < max_iter:
        previous_mval = current_mval
        current_mval = update(current_mval)
        diff = np.abs(current_mval-previous_mval)
        step += 1

    return current_mval
