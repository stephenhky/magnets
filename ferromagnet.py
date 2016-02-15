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
    nsteps = T.iscalar('nsteps')
    init_m = T.dscalar('init_m')
    m_seq = T.dscalar('m_seq')
    grad_results, grad_updates = theano.scan(fn=lambda vars: vars-0.1*grad_m,
                                             outputs_info=T.ones_like(m_seq),
                                             sequences=m_seq,
                                             n_steps=nsteps)
    final_m = grad_results[-1]
    cal_m = theano.function(inputs=[init_m, nsteps], outputs=final_m, updates=grad_updates)
    # cal_m = []

    return free_energy, [m], free_energy_grad, cal_m


