from __future__ import division

import torch
import torchvision.transforms.functional as F
from utils.losses import L1, L2
import time
from utils.patches import gaussian_map_hessian
import sys

ee = sys.float_info.epsilon
import numpy as np


class eig_real_symmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        evalue, evec = torch.linalg.eig(A)

        evecs = torch.view_as_real(evec)[..., 0]
        evals = torch.view_as_real(evalue)[..., 0]
        evalsnew = evals.clone()
        evecsnew = evecs.clone()

        if torch.isnan(evals).any():
            print(evals[evals != evals])

        xx = torch.argmax(torch.abs(evecs[:, 0, :]), dim=1, keepdim=True)
        xxx = xx.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)
        yy = torch.argmax(torch.abs(evecs[:, 1, :]), dim=1, keepdim=True)
        yyy = yy.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)
        zz = torch.argmax(torch.abs(evecs[:, 2, :]), dim=1, keepdim=True)
        zzz = zz.repeat(1, evecs.size()[-1]).view(evecs.size()[0], evecs.size()[-1], 1)

        evalsnew[:, 0:1] = torch.gather(evals, 1, xx)
        evalsnew[:, 1:2] = torch.gather(evals, 1, yy)
        evalsnew[:, 2:3] = torch.gather(evals, 1, zz)
        evecsnew[:, :, 0:1] = torch.gather(evecs, 2, xxx)
        evecsnew[:, :, 1:2] = torch.gather(evecs, 2, yyy)
        evecsnew[:, :, 2:3] = torch.gather(evecs, 2, zzz)
        ctx.save_for_backward(evecsnew)

        return evalsnew

    @staticmethod
    def backward(ctx, grad_evals):

        # grad_evals: (...,na)
        na = grad_evals.shape[-1]
        nbatch = grad_evals.shape[0]
        dLde = grad_evals.view(-1, na, 1)  # (nbatch, na)
        evecs, = ctx.saved_tensors
        U = evecs.view(-1, na, na)  # (nbatch,na,na)
        UT = U.transpose(-2, -1)  # (nbatch,na,na)
        econtrib = None

        # Control
        U1 = torch.bmm(UT[:, 0:1, :], U[:, :, 0:1])
        U2 = torch.bmm(UT[:, 1:2, :], U[:, :, 1:2])
        U3 = torch.bmm(UT[:, 2:3, :], U[:, :, 2:3])

        if not torch.allclose((torch.mean(U1) + torch.mean(U2) + torch.mean(U3)),
                              torch.tensor([3], dtype=torch.float).to("cuda"), atol=1e-5, rtol=1e-3):
            print(torch.mean(U1), torch.mean(U2), torch.mean(U3))
            # raise ValueError("W not normalized - Gradient forced at 0")
            print("W not normalized - Gradient forced at 0")
            econtrib = torch.zeros((nbatch, na, na))
        else:
            UU1 = (torch.bmm(U[:, :, 0:1], UT[:, 0:1, :])).view(-1, 1, na * na)
            UU2 = (torch.bmm(U[:, :, 1:2], UT[:, 1:2, :])).view(-1, 1, na * na)
            UU3 = (torch.bmm(U[:, :, 2:3], UT[:, 2:3, :])).view(-1, 1, na * na)
            UU = torch.cat((UU1, UU2, UU3), 1)

            '''
            UU = torch.empty(nbatch, na, na*na).to(grad_evals.dtype).to(grad_evals.device)
            # calculate the contribution from grad_evals
            for i in range(nbatch):
                for j in range(3):
                    UU[i,j] = torch.kron(UT[i,j],UT[i,j])
            '''
            UUT = UU.transpose(-2, -1)  # (nbatch,na,na)
            econtrib = torch.bmm(UUT, dLde)
            econtrib = econtrib.view(nbatch, na, na)
        return econtrib


def hessian_matrix_sigma_old(image, sigmas=5):
    H = torch.zeros(image.size() + (3, 3), dtype=torch.float32, requires_grad=True).to(torch.device("cuda"))
    '''
    PyTorch or more precisely autograd is not very good in handling in-place operations, 
    especially on those tensors with the requires_grad flag set to True.
    Generally you should avoid in-place operations where it is possible, in some cases it can work, 
    but you should always avoid in-place operations on tensors where you set requires_grad to True.
    Unfortunately there are not many pytorch functions to help out on this problem. 
    So you would have to use a helper tensor to avoid the in-place operation.
    '''
    H = H + 0  # new tensor, out-of-place operation (get around the problem, it is like "magic!")

    gaussian_importance_map = torch.as_tensor(gaussian_map_hessian((image.size()[1], image.size()[2], image.size()[3])),
                                              dtype=torch.float16).to("cuda").detach()
    gaussian_filtered = (image * 255.0) * gaussian_importance_map

    for i in range(1, sigmas + 1, sigmas // 5):
        sigma = i
        gaussian_filtered = F.gaussian_blur(image * 255.0, 2 * int(4 * sigma + 0.5) + 1, sigma)
        H[..., 0, 0] = H[..., 0, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=1)[0]
        H[..., 0, 1] = H[..., 0, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=2)[0]
        H[..., 0, 2] = H[..., 0, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=3)[0]
        H[..., 1, 0] = H[..., 1, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=1)[0]
        H[..., 1, 1] = H[..., 1, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=2)[0]
        H[..., 1, 2] = H[..., 1, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=3)[0]
        H[..., 2, 0] = H[..., 2, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=1)[0]
        H[..., 2, 1] = H[..., 2, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=2)[0]
        H[..., 2, 2] = H[..., 2, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=3)[0]

    return H, gaussian_filtered


def hessian_matrix_gt_sigma_old(image, sigmas=5):
    H = torch.zeros(image.size() + (3, 3), dtype=torch.float32, requires_grad=True).to(torch.device("cuda"))
    '''
    PyTorch or more precisely autograd is not very good in handling in-place operations, 
    especially on those tensors with the requires_grad flag set to True.
    Generally you should avoid in-place operations where it is possible, in some cases it can work, 
    but you should always avoid in-place operations on tensors where you set requires_grad to True.
    Unfortunately there are not many pytorch functions to help out on this problem. 
    So you would have to use a helper tensor to avoid the in-place operation.
    '''
    H = H + 0  # new tensor, out-of-place operation (get around the problem, it is like "magic!")

    for i in range(1, sigmas + 1, sigmas // 5):
        sigma = i
        gaussian_filtered = F.gaussian_blur(image * 255.0, 2 * int(4 * sigma + 0.5) + 1, sigma)
        H[..., 0, 0] = H[..., 0, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=1)[0]
        H[..., 0, 1] = H[..., 0, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=2)[0]
        H[..., 0, 2] = H[..., 0, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=3)[0]
        H[..., 1, 0] = H[..., 1, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=1)[0]
        H[..., 1, 1] = H[..., 1, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=2)[0]
        H[..., 1, 2] = H[..., 1, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=3)[0]
        H[..., 2, 0] = H[..., 2, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=1)[0]
        H[..., 2, 1] = H[..., 2, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=2)[0]
        H[..., 2, 2] = H[..., 2, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=3)[0]

    return H, gaussian_filtered


def hessian_matrix_gt_sigma(image, sigmas=5):
    H = torch.zeros(image.size() + (3, 3), dtype=torch.float32, requires_grad=True).to(torch.device("cuda"))
    '''
    PyTorch or more precisely autograd is not very good in handling in-place operations, 
    especially on those tensors with the requires_grad flag set to True.
    Generally you should avoid in-place operations where it is possible, in some cases it can work, 
    but you should always avoid in-place operations on tensors where you set requires_grad to True.
    Unfortunately there are not many pytorch functions to help out on this problem. 
    So you would have to use a helper tensor to avoid the in-place operation.
    '''
    H = H + 0  # new tensor, out-of-place operation (get around the problem, it is like "magic!")
    sigma = 1
    gaussian_filtered = F.gaussian_blur(image * 255.0, 2 * int(4 * sigma + 0.5) + 1, sigma)
    for i in range(1 + sigmas // 5, sigmas + 1, sigmas // 5):
        sigma = i
        gaussian_filtered = gaussian_filtered + F.gaussian_blur(image * 255.0, 2 * int(4 * sigma + 0.5) + 1, sigma)

    gaussian_filtered = gaussian_filtered / gaussian_filtered.max() * 255.0
    H[..., 0, 0] = H[..., 0, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=1)[0]
    H[..., 0, 1] = H[..., 0, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=2)[0]
    H[..., 0, 2] = H[..., 0, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=1)[0], dim=3)[0]
    H[..., 1, 0] = H[..., 1, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=1)[0]
    H[..., 1, 1] = H[..., 1, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=2)[0]
    H[..., 1, 2] = H[..., 1, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=2)[0], dim=3)[0]
    H[..., 2, 0] = H[..., 2, 0] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=1)[0]
    H[..., 2, 1] = H[..., 2, 1] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=2)[0]
    H[..., 2, 2] = H[..., 2, 2] + torch.gradient(torch.gradient(gaussian_filtered, dim=3)[0], dim=3)[0]

    return H, gaussian_filtered


def eigen_hessian_matrix(image, gt, sigma=25):
    H, gaussian = hessian_matrix_gt_sigma(image, sigma)

    eigenvalues = eig_real_symmetric.apply(H[gt == 1])

    return eigenvalues


def eigen_hessian_matrix_nogt(image, wt, sigma=25):
    H, gaussian = hessian_matrix_sigma_old(image, sigma)

    eigenvalues = eig_real_symmetric.apply(H[wt == 1].view(-1, H.shape[-2], H.shape[-1]))

    return eigenvalues


def vesselness_gt(pred, gt, eigenvtrue, sigma=25):
    eigenv = eigen_hessian_matrix(pred, gt, sigma)
    # loss = L1(eigenv, eigenvtrue)
    loss = L2(eigenv, eigenvtrue)

    if torch.isnan(loss):
        print(eigenv)
        print(eigenvtrue)

    return loss


def vesselness_nogt(pred, wt, eigenvtrue, sigma=25):
    eigenv = eigen_hessian_matrix_nogt(pred, wt, sigma)
    # loss = L1(eigenv, eigenvtrue)
    loss = L2(eigenv, eigenvtrue)
    if torch.isnan(loss):
        print(eigenv)
        print(eigenvtrue)

    return loss


def vesselness_true(pred, gt, sigma=25):
    eigenv = eigen_hessian_matrix(pred, gt, sigma)

    return eigenv


def vesselness_true_nogt(pred, wt, sigma=25):
    eigenv = eigen_hessian_matrix_nogt(pred, wt, sigma)

    return eigenv


def frangi(eigenvalues, alpha=0.1, beta=0.1, gamma=2):
    Ra = torch.abs(eigenvalues[..., 1]) / (torch.abs(eigenvalues[..., 2]) + 1e-6)
    Rb = torch.abs(eigenvalues[..., 0]) / (
                torch.sqrt(torch.abs(eigenvalues[..., 1]) * (torch.abs(eigenvalues[..., 2])) + 1e-6) + 1e-6)
    S = torch.sqrt((eigenvalues[..., 0] ** 2) + (eigenvalues[..., 1] ** 2) + (eigenvalues[..., 2] ** 2) + 1e-6)
    F = (1 - torch.exp(-(Ra ** 2) / (2 * (alpha ** 2)))) * torch.exp(-(Rb ** 2) / (2 * (beta ** 2))) * (
                1 - torch.exp(-(S ** 2) / (2 * (gamma ** 2))))
    F = torch.where((eigenvalues[..., 1] < 0) & (eigenvalues[..., 2] < 0), F, F * 0.)
    F = F / (torch.max(F) + 1e-6)

    return F


def jerman(eigenvalues, x, tau=0.5):
    eigenvalues = -eigenvalues
    l3max = tau * (eigenvalues[torch.argmax(x)][2])
    eigenvalues[..., 2] = torch.where((eigenvalues[..., 2] <= l3max) & (eigenvalues[..., 2] > 0),
                                      (eigenvalues[..., 2] / 1e6) + l3max, eigenvalues[..., 2])
    eigenvalues[..., 2] = torch.where((eigenvalues[..., 2] < 0), eigenvalues[..., 2] * 0., eigenvalues[..., 2])
    J = (eigenvalues[..., 1] ** 2) * (eigenvalues[..., 2] - eigenvalues[..., 1]) * (
                (3 / (eigenvalues[..., 1] + eigenvalues[..., 2] + 1e-6)) ** 3)
    J = torch.where((eigenvalues[..., 1] <= 0) | (eigenvalues[..., 2] <= 0), J * 0, J)
    J = torch.where(((eigenvalues[..., 2] / 2) > 0) & ((eigenvalues[..., 2] / 2) <= eigenvalues[..., 1]),
                    J / (J + 1e-6), J)
    J = J / (torch.max(J) + 1e-6)

    return J


def vesselness_frangi_ssvmd(pred, gt, eigenvtrue, sigma=25):
    eigenv = eigen_hessian_matrix(pred, gt, sigma)

    _, indices = torch.abs(eigenvtrue).sort(dim=1, stable=True)
    eigenvtrue = torch.take_along_dim(eigenvtrue, indices, dim=1)
    _, indices = torch.abs(eigenv).sort(dim=1, stable=True)
    eigenv = torch.take_along_dim(eigenv, indices, dim=1)

    loss = L2(frangi(eigenvtrue), frangi(eigenv))

    return loss


def vesselness_jerman_ssvmd(pred, gt, eigenvtrue, sigma=25):
    eigenv = eigen_hessian_matrix(pred, gt, sigma)

    _, indices = torch.abs(eigenvtrue).sort(dim=1, stable=True)
    eigenvtrue = torch.take_along_dim(eigenvtrue, indices, dim=1)
    _, indices = torch.abs(eigenv).sort(dim=1, stable=True)
    eigenv = torch.take_along_dim(eigenv, indices, dim=1)

    loss = L2(jerman(eigenvtrue, gt[gt == 1]), jerman(eigenv, pred[gt == 1]))

    return loss


def vesselness_self_frangi(pred, gt, sigma=25):
    # alpha=0.1,beta=0.5,gamma=5

    eigenv = eigen_hessian_matrix(pred, gt, sigma)

    _, indices = torch.abs(eigenv).sort(dim=1, stable=True)

    eigenvalues = torch.take_along_dim(eigenv, indices, dim=1)
    F_loss = torch.mean(1 - frangi(eigenvalues))

    return F_loss


