import time
import numpy as np
from random import randint

from utils import print_end_message, print_start_message, print_progress

def ista(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    # Parameter setup
    lmbd = params['lambda']
    alpha = 1 / params['Lips']
    
    X = params['x0']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    for k in range(1, params['maxit'] + 1):
        # Perform proximal gradient step
        X = proxg(X - alpha * gradf(X), alpha * lmbd)
        # Record convergence
        run_details['conv'][k] = fx(X) + lmbd * gx(X)
        
        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(X), gx(X))


    run_details['X_final'] = X
    print_end_message(method_name, time.time() - tic)
    return run_details

def fista(fx, gx, gradf, proxg, params):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    
    print_start_message(method_name)

    tic = time.time()

    # Parameter setup
    num_it = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['Lips']
    
    X = params['x0']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    if params['restart_fista']:
        X_prev = params['x0']
        theta_prev = 1
        theta = 1
    else:
        Y = params['x0']
        t = 1

    for k in range(1, num_it + 1):

        if params['restart_fista']:
            Y = X + theta * (1 / theta_prev - 1) * (X - X_prev)
            X_next = proxg(Y - alpha * gradf(Y), alpha * lmbd)
            
            if gradient_scheme_restart_condition(X, X_next, Y):
                theta_prev = 1
                theta = 1
                Y = X
                X_next = proxg(Y - alpha * gradf(Y), alpha * lmbd)

            theta_next = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
            # Update parameters
            X_prev = X
            X = X_next

            theta_prev = theta
            theta = theta_next

        else:
            X_next = proxg(Y - alpha * gradf(Y), alpha * lmbd)
            t_next = (1 + np.sqrt(4 * t ** 2 + 1)) / 2
            Y = X_next + (X_next - X) * (t - 1) / t_next

            # Update variables
            X = X_next
            t = t_next

        # record convergence
        run_details['conv'][k] = fx(X) + lmbd * gx(X)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(X), gx(X))

    run_details['X_final'] = X

    print_end_message(method_name, time.time() - tic)
    return run_details

def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    return np.trace((Y_k - X_k_next).T @ (X_k_next - X_k)) > 0

def prox_sg(fx, gx, stocgradfx, proxg, params):
    method_name = 'PROX-SG'
    print_start_message(method_name)

    tic = time.time()

    # Parameter setup
    num_it = params['maxit']
    lmbd = params['lambda']

    X = params['x0']
    gamma = params['stoch_rate_regime']
    
    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    # Auxiliary variables to define ergodic iterate
    num = gamma(0) * X
    den = gamma(0)

    for k in range(1, num_it + 1):
        X = proxg(X - gamma(k - 1) * stocgradfx(X, params['minib_size']), gamma(k - 1) * lmbd)
        num += gamma(k) * X
        den += gamma(k)

        X_erg = num / den
        run_details['conv'][k] = fx(X_erg) + lmbd * gx(X_erg)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(X_erg), gx(X_erg))

    run_details['X_final'] = X_erg
    
    print_end_message(method_name, time.time() - tic)
    return run_details

