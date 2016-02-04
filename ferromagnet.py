import numpy as np
import theano
import theano.tensor as T

def free_energy(r, u, H):
    m = T.dscalar('m')
    F = 0.5*r*m*m+0.25*u*m*m*m*m-H*m
    free_energy = theano.function(inputs=[m], outputs=F)
    return free_energy

