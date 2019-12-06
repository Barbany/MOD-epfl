import sys

sys.path.append('../')

import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

from utils import apply_random_mask, psnr, load_image, print_start_message, print_end_message
from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox


def FISTA(fx, gx, gradf, proxg, params):
    method_name = 'FISTA with lambda = ' + str(params['lambda'])
    print_start_message(method_name)

    tic = time.time()

    # Parameter setup
    lmbd = params['lambda']
    alpha = 1 / params['Lips']

    x = params['x0']
    y = params['x0']
    t = 1
    info = []

    for k in range(1, params['maxit'] + 1):
        # Perform accelerated proximal gradient step
        x_next = proxg(y - alpha * gradf(y), alpha)
        t_next = (1 + np.sqrt(4 * t ** 2 + 1)) / 2
        y = x_next + (x_next - x) * (t - 1) / t_next

        # Update parameters
        x = x_next
        t = t_next

        # Record convergence
        F_x = fx(x) + lmbd * gx(x)
        info.append(F_x)

    print_end_message(method_name, time.time() - tic)
    return x, info


def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = RepresentationOperator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices)  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m']))  # W. P_Omega^T

    # Generate measurements
    b = p_omega(image.reshape(params['N'], 1), indices)

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: np.linalg.norm(x, ord=1)
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: p_omega(x, indices)  # P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m'])  # P_Omega^T

    # Generate measurements
    b = forward_operator(image.reshape(params['N'], 1))

    fx = lambda x: 0.5 * np.linalg.norm(b - forward_operator(x)) ** 2
    gx = lambda x: TV_norm(x)
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(params['N'], 1)

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), info


def plot_best_worst(image, im_us, recons, psnr, reg):
    _, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(recons[np.argmax(psnr)], cmap='gray')
    ax[2].set_title('Largest PSNR ' + reg)
    ax[3].imshow(recons[np.argmin(psnr)], cmap="gray")
    ax[3].set_title('Lowest PSNR ' + reg)
    plt.show()

if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    PATH = 'data/me.jpg'
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    # Choose optimization parameters

    ###########################
    # Perform parameter sweep #
    ###########################
    num_points = 50
    lambdas = np.logspace(-5, 1, num=num_points)
    psnr_l1 = np.zeros(num_points)
    psnr_tv = np.zeros(num_points)

    reconstruction_l1 = []
    reconstruction_tv = []

    for i, lambda_ in enumerate(lambdas):
        params['lambda'] = lambda_

        reconstruction_l1.append(reconstruct_l1(image, indices, FISTA, params)[0])

        psnr_l1[i] = psnr(image, reconstruction_l1[-1])

        reconstruction_tv.append(reconstruct_TV(image, indices, FISTA, params)[0])

        psnr_tv[i] = psnr(image, reconstruction_tv[-1])

    colors = {'l1': 'red', 'tv': 'blue'}

    # Plot PSNR against lambda
    plt.plot(lambdas, psnr_l1, color=colors['l1'], lw=2, label='L1')
    plt.plot(lambdas, psnr_tv, color=colors['tv'], lw=2, label='TV')
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel('PSNR')
    plt.xscale('log')
    plt.grid()
    plt.show()

    # Plot best and worst images
    plot_best_worst(image, im_us, reconstruction_l1, psnr_l1, 'L1')
    plot_best_worst(image, im_us, reconstruction_tv, psnr_tv, 'TV')
