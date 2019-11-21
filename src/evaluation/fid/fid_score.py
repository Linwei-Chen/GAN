import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
# from scipy.misc import imread
from imageio import imread
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

from evaluation.fid.inception import InceptionV3
from data.image_folder import make_dataset as make_imglist

def get_activations(files, model, batch_size=1, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)  # 少于1维则转换为1维，高于则保留
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():  # .all 均为true则返回true，否则返回false
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps  # np.eye 创建一个1填充的对角矩阵
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)  # n*2048 todo:这里需要debug跑一边看属性
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)  # 协方差矩阵
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    files = make_imglist(path)
    m, s = calculate_activation_statistics(files, model, batch_size,
                                           dims, cuda)

    return m, s


def calculate_fid_given_paths(real_path, fake_path, batch_size, gpus, dims):
    for p in (real_path, fake_path):
        assert os.path.exists(p), 'Invalid path: %s' % p

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if gpus:
        model = model.to('cuda')

    m1, s1 = _compute_statistics_of_path(real_path, model, batch_size,
                                         dims, gpus)
    m2, s2 = _compute_statistics_of_path(fake_path, model, batch_size,
                                         dims, gpus)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def fid_score(real_path, fake_path, gpu='-1'):
    # parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--real_path', type=str, help=('Path to the real images'))
    # parser.add_argument('--fake_path', type=str, help=('Path to the generated images'))
    # parser.add_argument('--batch-size', type=int, default=1, help='Batch size to use')
    # parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    #                     help=('Dimensionality of Inception features to use. By default, uses pool3 features'))
    # parser.add_argument('--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
    # args = parser.parse_args()

    if gpu == '-1':
        gpu = ''
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    batch_size = 50
    dims = 2048

    fid_value = calculate_fid_given_paths(real_path, fake_path, batch_size, gpu != '', dims)
    return fid_value
