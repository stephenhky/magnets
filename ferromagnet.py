import numpy as np
import theano
import theano.tensor as T

def ferromagnet_free_energy():
    m = T.iscalar('m')
    r = T.iscalar('r')
    u = T.iscalar('u')
    H = T.iscalar('H')
    variables = [m]
    params = [r, u, H]

    F = 0.5*r*m*m+0.25*u*m*m*m*m-H*m
    free_energy_gradients = [T.grad(F, m)]

    free_energy = theano.function(inputs=[r, u, H, m], outputs=F)

    return free_energy, variables, free_energy_gradients, params

def minimize_free_energy(variables, grad_variables):
    nsteps = T.iscalar('nsteps')
    init_variables = T.iscalar('init_variables')
    grad_results, grad_updates = theano.scan(fn=lambda vars, gvars: vars-0.1*gvars,
                                             outputs_info=[{'initial': init_variables}],
                                             sequences=variables,
                                             non_sequences=grad_variables,
                                             n_steps=nsteps)
    cal_newparam = theano.function(inputs=[variables, nsteps], outputs=grad_results, updates=grad_updates)
    return cal_newparam
