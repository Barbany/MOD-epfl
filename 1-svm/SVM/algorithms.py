import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve


def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Gradient Descent')

    # Initialize x and alpha.
    x = parameter['x0']
    alpha = 1 / parameter['Lips']
    
    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter):
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    print(68 * '*')
    print('Gradient Descent  with strong convexity')

    # Initialize x and alpha.
    x = parameter['x0']
    alpha = 2 / (parameter['Lips'] + parameter['strcnvx'])

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                 strcnvx	- strong convexity parameter
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient')

    # Initialize x, y and t.
    x = parameter['x0']
    y = parameter['x0']
    t = 1
    
    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - gradf(y) / parameter['Lips']
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        t = t_next

    return x, info


# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx	- strong convexity parameter
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with strong convexity')

    # Initialize x and y.
    x = parameter['x0']
    y = parameter['x0']

    # Smoothness and strong convexity
    L = parameter['Lips']
    mu = parameter['strcnvx']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - gradf(y) / L
        y = x_next + (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)) * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info


# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent with line search')

    # Initialize x and L.
    x = parameter['x0']
    L = parameter['Lips']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d = - gradf(x)
        
        # Compute local smoothness
        L = 0.5 * L
        i = 0
        while fx(x + d / (2**i * L)) > fx(x) - (d.T @ d) / (2**(i + 1) * L):
            i += 1
                
        L = 2**i * L

        # Perform actual Gradient Descent step
        x_next = x + d / L

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info


# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
        """
    print(68 * '*')
    print('Accelerated Gradient with line search')

    # Initialize x, y and t.
    x = parameter['x0']
    y = parameter['x0']
    t = 1

    # Initialize L to Lipschitz constant of the gradients
    L = parameter['Lips']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d = - gradf(y)
        
        # Compute local smoothness
        L *= 0.5
        i = 0
        while fx(y + d / (2**i * L)) > fx(y) - (d.T @ d) / (2**(i + 1) * L):
            i = i + 1
        
        L *= 2**i

        # Perform actual Accelerated Gradient Descent step
        x_next = y + d / L
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * 2**(i-1) * t**2))
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with restart')

    # Initialize x, y, t and find the initial function value (fval).
    x = parameter['x0']
    y = parameter['x0']
    t = 1

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - gradf(y) / parameter['Lips']
        
        # Check if we restart
        if fx(x) < fx(x_next):
            y = x
            x_next = x - gradf(y) / parameter['Lips']
            t = 1

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        y = x_next + (t - 1) / t_next * (x_next - x)
        
        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with line search + restart')

    # Initialize x, y, t and find the initial function value (fval).
    x = parameter['x0']
    y = parameter['x0']
    t = 1
    
    # Initialize L to Lipschitz constant of the gradients
    L = parameter['Lips']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # This variable is true if we restarted in the last iteration (reuse alpha from LS)

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        # TODO: Review
        # Compute local smoothness
        d = - gradf(y)
        L *= 0.5
        i = 0
        while fx(y + d / (2**i * L)) > fx(y) - (d.T @ d) / (2**(i + 1) * L):
            i = i + 1
        
        L *= 2**i

        # Perform actual Accelerated Gradient Descent step
        x_next = y + d / L
            
        # Check if we restart
        if fx(x) < fx(x_next):
            x_next = x - gradf(x) / L
            t = 1
            
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * 2**(i-1) * t**2))
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        fval = fx(x)

    return x, info


def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Adaptive Gradient method')

    # Initialize x, alpha, delta (and any other)
    x = parameter['x0']
    alpha = 1
    delta = 1e-5
    q = 0

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        d = gradf(x)
        q = q + d.T @ d
        x_next = x - alpha / (np.sqrt(q) + delta) * d

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """

    print(68 * '*')
    print('ADAM')

    # Initialize x, beta1, beta2, alpha, epsilon (and any other)
    x = parameter['x0']
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.1
    epsilon = 1e-8
    m = np.zeros(x.shape[0])
    v = np.zeros(x.shape[0])

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        g = gradf(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        x_next = x - alpha * (m / (1 - beta1)) / (np.sqrt(v / (1 - beta2)) + epsilon)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SGD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Stochastic Gradient Descent')

    # Initialize x and alpha.
    x = parameter['x0']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        # Pick i u.a.r
        i = np.random.randint(parameter['no0functions']) 
        x_next = x - gradf(x, i) / (iter + 1)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SAG(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Stochastic Gradient Descent with averaging')

    # Define dimension n
    n = parameter['no0functions']

    # Initialize x, alpha, v and sum of v.
    x = parameter['x0']
    alpha = 1 / (16 * parameter['Lmax'])
    v = np.zeros((n, len(x)))
    avg_v = np.zeros(len(x))

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        # Pick i u.a.r
        i = np.random.randint(n)
        x_next = x - avg_v * alpha
        # Update v
        avg_v -= v[i] / n
        v[i] = gradf(x, i)
        avg_v += v[i] / n

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Stochastic Gradient Descent with variance reduction')

    # Initialize x, q and gamma.
    x = parameter['x0']
    q = int(1000 * parameter['Lmax'])
    gamma = 0.01 / parameter['Lmax']

    maxit = parameter['maxit']
    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = np.zeros(x.shape[0])
        x_tilde = x
        v_k = gradf(x)
        for _ in range(q):
            # Pick i u.a.r.
            i = np.random.randint(parameter['no0functions'])
            x_tilde = x_tilde - gamma * (gradfsto(x_tilde, i) - gradfsto(x, i) + v_k)
            x_next += x_tilde / q

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info
