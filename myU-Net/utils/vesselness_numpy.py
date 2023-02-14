import numpy as np
from itertools import combinations_with_replacement
from scipy.ndimage.filters import gaussian_filter
import torch


def MCC(pred,target):
    tp = pred*target
    tn = (1-pred)*(1-target)
    fp = pred*(1-target)
    fn = (1-pred)*target
    N = (tp*tn) - (fp*fn)
    D = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return N/np.sqrt(D)

def rescale(data):
    rescaled = ((data - data.min()) / (data.max() - data.min()))
    return rescaled

def rescale_pred(data,channels):
    rescaled = ((data - data.min()) / (data.max() - data.min())) * channels
    return np.rint(rescaled)

def norm1(a,b):
    c = np.mean(abs(a - b))
    return c

def norm2(a,b):
    c = np.mean((a - b)**2)
    return c

def myabs(a):
    return torch.sqrt(torch.square(a)+1)

def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc'):
        """Compute the Hessian matrix.
        In 2D, the Hessian matrix is defined as::
            H = [Hrr Hrc]
                [Hrc Hcc]
        which is computed by convolving the image with the second derivatives
        of the Gaussian kernel in the respective r- and c-directions.
        The implementation here also supports n-dimensional data.
        Parameters
        ----------
        image : ndarray
            Input image.
        sigma : float
            Standard deviation used for the Gaussian kernel, which is used as
            weighting function for the auto-correlation matrix.
        mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.
        cval : float, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.
        order : {'rc', 'xy'}, optional
            This parameter allows for the use of reverse or forward order of
            the image axes in gradient computation. 'rc' indicates the use of
            the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
            usage of the last axis initially (Hxx, Hxy, Hyy)
        Returns
        -------
        H_elems : list of ndarray
            Upper-diagonal elements of the hessian matrix for each pixel in the
            input image. In 2D, this will be a three element list containing [Hrr,
            Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.
        Examples
        --------
        >>> from skimage.feature import hessian_matrix
        >>> square = np.zeros((5, 5))
        >>> square[2, 2] = 4
        >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc')
        >>> Hrc
        array([[ 0.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0., -1.,  0.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 0., -1.,  0.,  1.,  0.],
               [ 0.,  0.,  0.,  0.,  0.]])
        """
        H_elems = np.zeros(np.append(image.ndim*2,image.shape))
        for i in range(1,6):
                gaussian_filtered = gaussian_filter(image*255.0, sigma=i, mode=mode, cval=cval)

                gradients = np.gradient(gaussian_filtered)
                axes = range(image.ndim)

                if order == 'rc':
                        axes = reversed(axes)
                for idx, (ax0, ax1) in enumerate(combinations_with_replacement(axes, 2)):
                        H_elems[idx] += np.gradient(gradients[ax0], axis=ax1)

        return H_elems

def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
        l1 = (M00 + M11) / 2 + np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
        l2 = (M00 + M11) / 2 - np.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
        return l1, l2

def _symmetric_image(S_elems):
        image = S_elems[0]
        symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim))
        for idx, (row, col) in enumerate(combinations_with_replacement(range(image.ndim), 2)):
            symmetric_image[..., row, col] = S_elems[idx]
            symmetric_image[..., col, row] = S_elems[idx]
        return symmetric_image

def _symmetric_compute_eigenvaluesh(S_elems):
        """Compute eigenvalues from the upperdiagonal entries of a symmetric matrix
        Parameters
        ----------
        S_elems : list of ndarray
            The upper-diagonal elements of the matrix, as returned by
            `hessian_matrix` or `structure_tensor`.
        Returns
        -------
        eigs : ndarray
            The eigenvalues of the matrix, in decreasing order. The eigenvalues are
            the leading dimension. That is, ``eigs[i, j, k]`` contains the
            ith-largest eigenvalue at position (j, k).
        """

        if len(S_elems) == 3:  # Use fast Cython code for 2D
            eigs = np.stack(_image_orthogonal_matrix22_eigvals(*S_elems))
        else:
            matrices = _symmetric_image(S_elems)
            # eigvalsh returns eigenvalues in increasing order. We want decreasing
            eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
            leading_axes = tuple(range(eigs.ndim - 1))
            eigs = np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)
        return eigs

def _symmetric_compute_eigenvalues(S_elems):
        """Compute eigenvalues from the upperdiagonal entries of a symmetric matrix
        Parameters
        ----------
        S_elems : list of ndarray
            The upper-diagonal elements of the matrix, as returned by
            `hessian_matrix` or `structure_tensor`.
        Returns
        -------
        eigs : ndarray
        """

        if len(S_elems) == 3:  # Use fast Cython code for 2D
            eigs = np.stack(_image_orthogonal_matrix22_eigvals(*S_elems))
        else:
            matrices = _symmetric_image(S_elems)
            eigs = np.linalg.eigvals(matrices)
            leading_axes = tuple(range(eigs.ndim - 1))
            eigs = np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)
        return eigs

def _sortbyabs(array, axis=0):
    """
    Sort array along a given axis by absolute values.
    Parameters
    ----------
    array : (N, ..., M) ndarray
        Array with input image data.
    axis : int
        Axis along which to sort.
    Returns
    -------
    array : (N, ..., M) ndarray
        Array sorted along a given axis by absolute values.
    Notes
    -----
    Modified from: http://stackoverflow.com/a/11253931/4067734
    """

    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(array).argsort(axis)

    # Return abs sorted array
    return array[tuple(index)]

def vesselnessh(seg,sigma=1):
    hm = hessian_matrix(seg, sigma)
    hm = [(sigma ** 2) * e for e in hm]
    hessian_eigenvalues = _symmetric_compute_eigenvaluesh(hm)
    hessian_eigenvalues = _sortbyabs(hessian_eigenvalues, axis=0)
    hessian_eigenvalues = np.rollaxis(hessian_eigenvalues,0,4)

    return hessian_eigenvalues

def vesselness(seg,sigma=1):
    hm = hessian_matrix(seg, sigma)
    hessian_eigenvalues = _symmetric_compute_eigenvalues(hm)
    hessian_eigenvalues = np.rollaxis(hessian_eigenvalues,0,4)

    return hessian_eigenvalues

