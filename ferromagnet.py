import numpy as np
import theano
import theano.tensor as T

def ferromagnet_free_energy(r, u, H):
    # variables
    m = T.dscalar('m')
    nsteps = T.iscalar('nsteps')

    # free energy and its gradient
    fe = lambda m: 0.5*r*m*m+0.25*u*m*m*m*m-H*m
    grad_m = T.grad(fe(m), m)
    free_energy = theano.function(inputs=[m], outputs=fe(m))
    free_energy_grad = theano.function(inputs=[m], outputs=grad_m)

    # gradient descent
    def updatefcn(m1, gm1):
        m2 = m1-0.1*gm1
        gm2 = T.grad(fe(m2), m2)
        return m2, gm2

    grad_results, grad_updates = theano.scan(fn=updatefcn,
                                             outputs_info=[m, grad_m],
                                             sequences=[],
                                             n_steps=nsteps)
    final_m = grad_results[0][-1]
    cal_m = theano.function(inputs=[m, nsteps], outputs=final_m, updates=grad_updates)

    return free_energy, [m], free_energy_grad, cal_m

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
