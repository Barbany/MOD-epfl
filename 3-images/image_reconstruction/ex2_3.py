import sys

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

from utils import apply_random_mask, load_image, print_start_message, print_end_message, print_progress
from operators import RepresentationOperator, p_omega, p_omega_t, l1_prox


def ISTA(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    # Parameter setup
    alpha = 1 / params['Lips']

    info = []

    x = params['x0']

    for k in range(params['maxit']):
        # Perform proximal gradient step
        x = proxg(x - alpha * gradf(x), alpha)

        # Record convergence
        F_x = fx(x) + gx(x)
        info.append(F_x)

        # Early stopping
        if stop({'F_x': F_x}, params):
            print('Early stop')
            return x, info

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], F_x, fx(x), gx(x))
    return x, info


def FISTA(fx, gx, gradf, proxg, params):
    if params['restart_criterion']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'

    print_start_message(method_name)

    # Parameter setup
    num_it = params['maxit']
    alpha = 1 / params['Lips']

    info = []
    err_0 = None

    x = params['x0']
    y = params['x0']

    if params['restart_criterion']:
        x_prev = params['x0']
        theta_prev = 1
        theta = 1
    else:
        t = 1

    for k in range(num_it):

        if params['restart_criterion']:
            y_next = x + theta * (1 / theta_prev - 1) * (x - x_prev)
            x_next = proxg(y_next - alpha * gradf(y_next), alpha)

            if gradient_scheme_restart_condition(x, x_next, y_next):
                theta = 1
                y_next = x
                x_next = proxg(y_next - alpha * gradf(y_next), alpha)

            theta_next = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
            # Update parameters
            x_prev = x
            x = x_next

            theta_prev = theta
            theta = theta_next

        else:
            x_next = proxg(y - alpha * gradf(y), alpha)
            t_next = (1 + np.sqrt(4 * t ** 2 + 1)) / 2
            y_next = x_next + (x_next - x) * (t - 1) / t_next

            # Update variables
            x = x_next
            t = t_next

        if err_0 is None and ~np.all(np.isclose(y_next, y)):
            err_0 = np.linalg.norm(y_next - y)

        # Record convergence
        F_x = fx(x) + gx(x)
        info.append(F_x)

        # Early stopping
        if stop({'F_x': F_x, 'y': y, 'y_next': y_next, 'err_0': err_0}, params):
            return x, info

        # Common update
        y = y_next

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], F_x, fx(x), gx(x))

    return x, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    return ((y_k - x_k_next).T @ (x_k_next - x_k)) > 0


def stop(data, params):
    if params['stopping_criterion'] == 'rerr':
        if data['err_0'] is None:
            return False
        else:
            return np.linalg.norm(data['y_next'] - data['y']) / data['err_0'] < params['tol']
    elif params['stopping_criterion'] == 'F*':
        return np.abs(data['F_x'] - params['F*']) / params['F*'] < params['tol']
    else:
        raise NotImplemented


def plot_convergence(results, fs_star, ylabel):
    colors = {'ISTA': 'red', 'FISTA': 'blue', 'FISTA-RESTART': 'green'}

    for key in ['ISTA', 'FISTA', 'FISTA-RESTART']:
        if key in results:
            num_iterations = len(results[key])
            k = np.array(range(0, num_iterations))
            plt.plot(k, abs(results[key] - fs_star) / fs_star,
                     color=colors[key], lw=2, label=key)

    plt.legend()
    plt.xlabel('#iterations')
    plt.ylabel(ylabel)
    # plt.ylim(1e-8, 1e6)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 5000,
        'tol': 10e-15,
        'Lips': 1,
        'lambda': 0.01,
        'F*': None,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': True,
        'stopping_criterion': 'rerr',
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    PATH = 'data/gandalf.jpg'
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices

    # Wavelet operator
    r = RepresentationOperator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices)  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m']))  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image.reshape(params['N'], 1), indices)

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: params['lambda'] * np.linalg.norm(x, ord=1)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    # Part (b): Compute F*
    x, info = FISTA(fx, gx, gradf, proxg, params)
    params['F*'] = info[-1]

    # Part (c): Plot relative error to F*
    params['stopping_criterion'] = 'F*'
    params['maxit'] = 2000

    _, info_ista = ISTA(fx, gx, gradf, proxg, params)
    _, info_fista_restart = FISTA(fx, gx, gradf, proxg, params)

    params['restart_criterion'] = False
    _, info_fista = FISTA(fx, gx, gradf, proxg, params)

    results = {'ISTA': info_ista, 'FISTA': info_fista, 'FISTA-RESTART': info_fista_restart}
    plot_convergence(results, params['F*'], r'$ |F(\mathbf{x}^k) - F^\star|  /  F^\star$')

    # Part (d): Plot relative error to F^natural
    x_nat = image.reshape(params['N'], 1)
    F_natural = fx(x_nat) + gx(x_nat)
    plot_convergence(results, F_natural, r'$ |F(\mathbf{x}^k) - F^\natural|  /  F^\natural$')

    print('The value of F* is ' + str(params['F*']))
    print('The value of F^nat is ' + str(F_natural))
