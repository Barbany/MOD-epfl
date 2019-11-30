import sys

sys.path.append('../')

import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from utils import apply_random_mask, psnr, load_image
from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from unrolled_network import ResNetDC, train_epoch


def FISTA(gradf, proxg, params):
    # Parameter setup
    lmbd = params['lambda']
    alpha = 1 / params['Lips']
    
    x = params['x0']
    x_prev = params['x0']
    theta_prev = 1
    theta = 1
    
    tic = time.time()

    for k in range(params['maxit']):
        y = x + theta * (1 / theta_prev - 1) * (x - x_prev)
        x_next = proxg(y - alpha * gradf(y), alpha * lmbd)
            
        if gradient_scheme_restart_condition(x, x_next, y):
            theta = 1
            y_next = x
            x_next = proxg(y_next - alpha * gradf(y_next), alpha * lmbd)

        theta_next = (np.sqrt(theta ** 4 + 4 * theta ** 2) - theta ** 2) / 2
        
        # Update parameters
        x_prev = x
        x = x_next
        theta_prev = theta
        theta = theta_next

    return x, time.time() - tic


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    return ((y_k - x_k_next).T @ (x_k_next - x_k)) > 0


def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = RepresentationOperator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: p_omega(r.WT(x), indices)                  # P_Omega.W^T
    adjoint_operator = lambda x: r.W(p_omega_t(x, indices, params['m']))    # W. P_Omega^T

    # Generate measurements
    b = p_omega(image.reshape(params['N'], 1), indices)

    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: adjoint_operator(forward_operator(x) - b)

    x, time = optimizer(gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m'])), time


def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: p_omega(x, indices)                # P_Omega
    adjoint_operator = lambda x: p_omega_t(x, indices, params['m']) # P_Omega^T

    # Generate measurements
    b = forward_operator(image.reshape(params['N'], 1))

    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m'])),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1))
    gradf = lambda x: adjoint_operator(forward_operator(x) - b).reshape(params['N'], 1)
    
    x, time = optimizer(gradf, proxg, params)
    return x.reshape((params['m'], params['m'])), time


def reconstruct_nn(image, mask, params):
    image_torch = torch.tensor(image).view(1, 1, params['shape'][0], params['shape'][1]).float()

    class Dataset(torch.utils.data.Dataset):
        """Characterizes a dataset for PyTorch"""
        def __init__(self):
            """Initialization"""

        def __len__(self):
            """Denotes the total number of samples"""
            return 1

        def __getitem__(self, index):
            """Generates one sample of data"""
            if index == 0:
                return image_torch, None
            else:
                raise IndexError

    model = ResNetDC(2, unroll_depth=5)
    optimizer = torch.optim.Adam(model.parameters())
    data_loader = Dataset()

    args = {'rate': params['rate'], 'device': 'cpu', 'report_interval': 1, 'num_epochs': 4}

    t_start = time.time()
    for epoch in range(params['maxit']):
        train_epoch(args, epoch, model, data_loader, optimizer)
    time_nn = time.time() - t_start

    with torch.no_grad():
        mask_torch = torch.tensor(mask).view(1, 1, params['shape'][0], params['shape'][1]).float()
        im_us_torch = image_torch * mask_torch

        reconstruction = model(im_us_torch, mask_torch)
        reconstruction = reconstruction[0, 0, :, :].cpu().numpy()
    return reconstruction, time_nn


def plot_performance(image, image_us, reconstruction, time, method):
    psnr_rec = psnr(image, reconstruction)
    ssim_rec = ssim(image, reconstruction, data_range=np.max(image))

    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction, cmap="gray")
    ax[2].set_title(method + ' - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_rec, ssim_rec, time))
    im = ax[3].imshow(abs(image - reconstruction), cmap="inferno", vmax=.05)
    # Plot the colorbar
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical');

    [axi.set_axis_off() for axi in ax.flatten()]

    plt.show()


if __name__ == "__main__":
    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 500,
        'Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'shape': shape,
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

    # Section (a): Comparison with 500 iterations
    im_l1, time_l1 = reconstruct_l1(image, indices, FISTA, params)
    plot_performance(image, im_us, im_l1, time_l1, 'L1')
    im_tv, time_tv = reconstruct_TV(image, indices, FISTA, params)
    plot_performance(image, im_us, im_tv, time_tv, 'TV')

    # Section (b): Comparison with 5 iterations
    params['maxit'] = 5
    im_l1, time_l1 = reconstruct_l1(image, indices, FISTA, params)
    plot_performance(image, im_us, im_l1, time_l1, '5 iterations L1')
    im_tv, time_tv = reconstruct_TV(image, indices, FISTA, params)
    plot_performance(image, im_us, im_tv, time_tv, '5 iterations TV')
    im_nn, time_nn = reconstruct_nn(image, mask, params)
    plot_performance(image, im_us, im_nn, time_nn, '5 iterations Neural Network')
