import numpy as np
import theano
import theano.tensor as T

def ferromagnet_free_energy(r, u, H):
    m = T.iscalar('m')
    params = [m]

    F = 0.5*r*m*m+0.25*u*m*m*m*m-H*m
    free_energy_gradients = [T.grad(F, m)]

    free_energy = theano.function(inputs=[m], outputs=F)

    return free_energy, params, free_energy_gradients

def minimize_free_energy(free_energy, params, gparams):
    grad_results, grad_updates = theano.scan(fn=lambda param, gparam: param-0.1*gparam,
                                             sequences=params,
                                             non_sequences=gparams)
    cal_newparam = theano.function(inputs=params, outputs=grad_results)

