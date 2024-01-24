from __future__ import division  # use // for integer division
from __future__ import absolute_import  # use from . import
from __future__ import print_function  # print function
from __future__ import unicode_literals  # all the strings are unicode

__author__ = 'Youhei Akimoto'

import argparse
import json
import os
import time
from math import sqrt, exp, log, ceil, floor

import numpy as np
import pandas as pd
from numpy.linalg import norm, eigh
from numpy.random import randn

from rvgomea.run_algorithm import PROBLEM_CODES


class VkdCma(object):
    """O(N*k^2 + k^3) Time/Space Variant of CMA-ES with C = D * (I + V * V^T) * D

    References
    ----------
    [1] Youhei Akimoto and Nikolaus Hansen.
    Online Model Selection for Restricted Covariance Matrix Adaptation.
    In Proc. of PPSN 2016, pp. 3--13 (2016)
    [2] Youhei Akimoto and Nikolaus Hansen.
    Projection-Based Restricted Covariance Matrix Adaptation for High
    Dimension. In Proc. of GECCO 2016, pp. 197--204 (2016)
    """

    def __init__(self, func, xmean0, sigma0, **kwargs):

        # ES Parameters
        self.N = len(xmean0)
        self.lam = kwargs.get('lam', int(4 + floor(3 * log(self.N))))
        wtemp = np.array([
            np.log(float(self.lam + 1) / 2.0) - np.log(1 + i)
            for i in range(self.lam // 2)
        ])
        self.w = kwargs.get('w', wtemp / np.sum(wtemp))
        self.sqrtw = np.sqrt(self.w)
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.mu = self.w.shape[0]
        self.neval = 0

        # Arguments
        self.func = func
        self.xmean = np.array(xmean0)
        if isinstance(sigma0, np.ndarray):
            self.sigma = np.exp(np.log(sigma0).mean())
            self.D = sigma0 / self.sigma
        else:
            self.sigma = sigma0
            self.D = np.ones(self.N)

        # VkD Static Parameters
        self.k = kwargs.get('k_init', 0)  # alternatively, self.w.shape[0]
        self.kmin = kwargs.get('kmin', 0)
        self.kmax = kwargs.get('kmax', self.N - 1)
        assert (0 <= self.kmin <= self.kmax < self.N)
        self.k_inc_cond = kwargs.get('k_inc_cond', 30.0)
        self.k_dec_cond = kwargs.get('k_dec_cond', self.k_inc_cond)
        self.k_adapt_factor = kwargs.get('k_adapt_factor', 1.414)
        self.factor_sigma_slope = kwargs.get('factor_sigma_slope', 0.1)
        self.factor_diag_slope = kwargs.get(
            'factor_diag_slope', 1.0)  # 0.3 in PPSN (due to cc change)
        self.opt_conv = 0.5 * min(1, self.lam / self.N)
        self.accepted_slowdown = max(1., self.k_inc_cond / 10.)
        self.k_adapt_decay = 1.0 / self.N
        self.k_adapt_wait = 2.0 / self.k_adapt_decay - 1

        # VkD Dynamic Parameters
        self.k_active = 0
        self.last_log_sigma = np.log(self.sigma)
        self.last_log_d = 2.0 * np.log(self.D)
        self.last_log_cond_corr = np.zeros(self.N)
        self.ema_log_sigma = ExponentialMovingAverage(
            decay=self.opt_conv / self.accepted_slowdown, dim=1)
        self.ema_log_d = ExponentialMovingAverage(
            decay=self.k_adapt_decay, dim=self.N)
        self.ema_log_s = ExponentialMovingAverage(
            decay=self.k_adapt_decay, dim=self.N)
        self.itr_after_k_inc = 0

        # CMA Learning Rates
        self.cm = kwargs.get('cm', 1.0)
        (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)

        # TPA Parameters
        self.cs = kwargs.get('cs', 0.3)
        self.ds = kwargs.get('ds', 4 - 3 / self.N)  # or sqrt(N)
        self.flg_injection = False
        self.ps = 0

        # Initialize Dynamic Parameters
        self.V = np.zeros((self.k, self.N))
        self.S = np.zeros(self.N)
        self.pc = np.zeros(self.N)
        self.dx = np.zeros(self.N)
        self.U = np.zeros((self.N, self.k + self.mu + 1))
        self.arx = np.zeros((self.lam, self.N))
        self.arf = np.zeros(self.lam)

        # Stopping Condition
        self.fhist_len = 20 + self.N // self.lam
        self.tolf_checker = TolfChecker(self.fhist_len)
        self.ftarget = kwargs.get('ftarget', 1e-8)
        self.maxeval = kwargs.get('maxeval', 5e3 * self.N * self.lam)
        self.tolf = kwargs.get('tolf', abs(self.ftarget) / 1e5)
        self.tolfrel = kwargs.get('tolfrel', 1e-12)
        self.minstd = kwargs.get('minstd', 1e-12)
        self.minstdrel = kwargs.get('minstdrel', 1e-12)
        self.maxconds = kwargs.get('maxconds', 1e12)
        self.maxcondd = kwargs.get('maxcondd', 1e6)

    def run(self):

        itr = 0
        satisfied = False
        while not satisfied:
            itr += 1
            self._onestep()
            satisfied, condition = self._check()
            if itr % 20 == 0:
                print(itr, self.neval, self.arf.min(), self.sigma)
            if satisfied:
                print(condition)
        return self.xmean

    def _onestep(self):

        # ======================================================================
        # VkD-CMA (GECCO 2016)

        k = self.k
        ka = self.k_active

        # Sampling
        if True:
            # Sampling with two normal vectors
            # Available only if S >= 0
            arzd = randn(self.lam, self.N)
            arzv = randn(self.lam, ka)
            ary = (arzd + np.dot(arzv * np.sqrt(self.S[:ka]), self.V[:ka])
                   ) * self.D
        else:
            # Sampling with one normal vectors
            # Available even if S < 0 as long as V are orthogonal to each other
            arz = randn(self.lam, self.N)
            ary = arz + np.dot(
                np.dot(arz, self.V[:ka].T) *
                (np.sqrt(1.0 + self.S[:ka]) - 1.0), self.V[:ka])
            ary *= self.D

        # Injection
        if self.flg_injection:
            mnorm = self._mahalanobis_square_norm(self.dx)
            if mnorm > 0:
                dy = (norm(randn(self.N)) / sqrt(mnorm)) * self.dx
                ary[0] = dy
                ary[1] = -dy
        self.arx = self.xmean + self.sigma * ary

        # Evaluation
        self.arf = np.zeros(self.lam)
        for i in range(self.lam):
            self.arf[i] = self.func(self.arx[i])
            self.neval += 1
        idx = np.argsort(self.arf)
        sary = ary[idx[:self.mu]]

        # Update xmean
        self.dx = np.dot(self.w, sary)
        self.dz = self._inv_sqrt_ivv(self.dx / self.D)  # For flg_dev_kadapt
        self.xmean += (self.cm * self.sigma) * self.dx

        # TPA (PPSN 2014 version)
        if self.flg_injection:
            alpha_act = np.where(idx == 1)[0][0] - np.where(idx == 0)[0][0]
            alpha_act /= float(self.lam - 1)
            self.ps += self.cs * (alpha_act - self.ps)
            self.sigma *= exp(self.ps / self.ds)
            hsig = self.ps < 0.5
        else:
            self.flg_injection = True
            hsig = True

        # Cumulation
        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (2 - self.cc)
                                                        * self.mueff) * self.dx

        # Update V, S and D
        # Cov = D(alpha**2 * I + UU^t)D
        if self.cmu == 0.0:
            rankU = ka + 1
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, rankU - 1] = sqrt(self.cone) * (self.pc / self.D)
        elif self.cone == 0.0:
            rankU = ka + self.mu
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU] = sqrt(self.cmu) * self.sqrtw * (sary /
                                                                 self.D).T
        else:
            rankU = ka + self.mu + 1
            alpha = sqrt(
                abs(1 - self.cmu - self.cone + self.cone * (1 - hsig) * self.cc
                    * (2 - self.cc)))
            self.U[:, :ka] = (self.V[:ka].T * (np.sqrt(self.S[:ka]) * alpha))
            self.U[:, ka:rankU - 1] = sqrt(self.cmu) * self.sqrtw * (sary /
                                                                     self.D).T
            self.U[:, rankU - 1] = sqrt(self.cone) * (self.pc / self.D)

        if self.N > rankU:
            # O(Nk^2 + k^3)
            DD, R = eigh(np.dot(self.U[:, :rankU].T, self.U[:, :rankU]))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma
            beta = max(beta, 1e-12)

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            interm = np.sqrt(DD[idxeig[:ka]])
            interm = np.maximum(interm, 1e-12)
            self.V[:ka] = (np.dot(self.U[:, :rankU], R[:, idxeig[:ka]]) / interm).T
        else:
            # O(N^3 + N^2(k+mu+1))
            # If this is the case, the standard CMA is preferred
            DD, L = eigh(np.dot(self.U[:, :rankU], self.U[:, :rankU].T))
            idxeig = np.argsort(DD)[::-1]
            gamma = 0 if rankU <= k else DD[idxeig[k:]].sum() / (self.N - k)
            beta = alpha * alpha + gamma
            beta = max(beta, 1e-12)

            self.k_active = ka = min(np.sum(DD >= 0), k)
            self.S[:ka] = (DD[idxeig[:ka]] - gamma) / beta
            self.V[:ka] = L[:, idxeig[:ka]].T

        self.D *= np.sqrt(
            (alpha * alpha + np.sum(
                self.U[:, :rankU] * self.U[:, :rankU], axis=1)) /
            (1.0 + np.dot(self.S[:ka], self.V[:ka] * self.V[:ka])))

        # Covariance Normalization by Its Determinant
        gmean_eig = np.exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

        # ======================================================================
        # k-Adaptation (PPSN 2016)
        self.itr_after_k_inc += 1

        # Exponential Moving Average
        self.ema_log_sigma.update(log(self.sigma) - self.last_log_sigma)
        self.lnsigma_change = self.ema_log_sigma.M / (self.opt_conv /
                                                      self.accepted_slowdown)
        self.last_log_sigma = log(self.sigma)
        self.ema_log_d.update(2. * np.log(self.D) + np.log(1 + np.dot(
            self.S[:self.k], self.V[:self.k] ** 2)) - self.last_log_d)
        self.lndiag_change = self.ema_log_d.M / (self.cmu + self.cone)
        self.last_log_d = 2. * np.log(
            self.D) + np.log(1 + np.dot(self.S[:self.k], self.V[:self.k] ** 2))
        self.ema_log_s.update(np.log(1 + self.S) - self.last_log_cond_corr)
        self.lnlambda_change = self.ema_log_s.M / (self.cmu + self.cone)
        self.last_log_cond_corr = np.log(1 + self.S)

        # Check for adaptation condition
        flg_k_increase = self.itr_after_k_inc > self.k_adapt_wait
        flg_k_increase *= self.k < self.kmax
        flg_k_increase *= np.all((1 + self.S[:self.k]) > self.k_inc_cond)
        flg_k_increase *= (
                np.abs(self.lnsigma_change) < self.factor_sigma_slope)
        flg_k_increase *= np.all(
            np.abs(self.lndiag_change) < self.factor_diag_slope)

        flg_k_decrease = (self.k > self.kmin) * (
                1 + self.S[:self.k] < self.k_dec_cond)
        flg_k_decrease *= (self.lnlambda_change[:self.k] < 0.)

        if (self.itr_after_k_inc > self.k_adapt_wait) and flg_k_increase:
            # ----- Increasing k -----
            self.k_active = k
            self.k = newk = min(
                max(int(ceil(self.k * self.k_adapt_factor)), self.k + 1),
                self.kmax)
            self.V = np.vstack((self.V, np.zeros((newk - k, self.N))))
            self.U = np.empty((self.N, newk + self.mu + 1))
            # update constants
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)
            self.itr_after_k_inc = 0

        elif self.itr_after_k_inc > k * self.k_adapt_wait and np.any(
                flg_k_decrease):
            # ----- Decreasing k -----
            flg_keep = np.logical_not(flg_k_decrease)
            new_k = max(np.count_nonzero(flg_keep), self.kmin)
            self.V = self.V[flg_keep]
            self.S[:new_k] = (self.S[:flg_keep.shape[0]])[flg_keep]
            self.S[new_k:] = 0
            self.k = self.k_active = new_k
            # update constants
            (self.cone, self.cmu, self.cc) = self._get_learning_rate(self.k)
        # ==============================================================================

        # Covariance Normalization by Its Determinant
        gmean_eig = exp(self._get_log_determinant_of_cov() / self.N / 2.0)
        self.D /= gmean_eig
        self.pc /= gmean_eig

    def _mahalanobis_square_norm(self, dx):
        """Square norm of dx w.r.t. C = D*(I + V*S*V^t)*D

        Parameters
        ----------
        dx : numpy.ndarray (1D)

        Returns
        -------
        square of the Mahalanobis distance dx^t * (D*(I + V*S*V^t)*D)^{-1} * dx
        """
        D = self.D
        V = self.V[:self.k_active]
        S = self.S[:self.k_active]

        dy = dx / D
        vdy = np.dot(V, dy)
        return np.sum(dy * dy) - np.sum((vdy * vdy) * (S / (S + 1.0)))

    def _inv_sqrt_ivv(self, vec):
        """Return (I + V*V^t)^{-1/2} x"""
        if self.k_active == 0:
            return vec
        else:
            return vec + np.dot(
                np.dot(self.V[:self.k_active], vec) *
                (1.0 / np.sqrt(1.0 + self.S[:self.k_active]) - 1.0
                 ), self.V[:self.k_active])

    def _get_learning_rate(self, k):
        """Return the learning rate cone, cmu, cc depending on k

        Parameters
        ----------
        k : int
            the number of vectors for covariance matrix

        Returns
        -------
        cone, cmu, cc : float in [0, 1]. Learning rates for rank-one, rank-mu,
         and the cumulation factor for rank-one.
        """
        nelem = self.N * (k + 1)
        cone = 2.0 / (nelem + self.N + 2 * (k + 2) + self.mueff)  # PPSN 2016
        # cone = 2.0 / (nelem + 2 * (k + 2) + self.mueff)  # GECCO 2016
        # cc = (4 + self.mueff / self.N) / (
        #     (self.N + 2 * (k + 1)) / 3 + 4 + 2 * self.mueff / self.N)

        # New Cc and C1: Best Cc depends on C1, not directory on K.
        # Observations on Cigar (N = 3, 10, 30, 100, 300, 1000) by Rank-1 VkD.
        cc = sqrt(cone)
        cmu = min(1 - cone, 2.0 * (self.mueff - 2 + 1.0 / self.mueff) /
                  (nelem + 4 * (k + 2) + self.mueff))
        return cone, cmu, cc

    def _get_log_determinant_of_cov(self):
        return 2.0 * np.sum(np.log(self.D)) + np.sum(
            np.log(1.0 + self.S[:self.k_active]))

    def _check(self):
        is_satisfied = False
        condition = ''
        self.tolf_checker.update(self.arf)
        std = self.sigma * exp(self._get_log_determinant_of_cov() / self.N /
                               2.0)

        if self.arf.min() <= self.ftarget:
            is_satisfied = True
            condition = 'ftarget'
        if not is_satisfied and self.neval >= self.maxeval:
            is_satisfied = True
            condition = 'maxeval'
        if not is_satisfied and self.tolf_checker.check_relative(self.tolfrel):
            is_satisfied = True
            condition = 'tolfrel'
        if not is_satisfied and self.tolf_checker.check_absolute(self.tolf):
            is_satisfied = True
            condition = 'tolf'
        if not is_satisfied and self.tolf_checker.check_flatarea():
            is_satisfied = True
            condition = 'flatarea'
        if not is_satisfied and std < self.minstd:
            is_satisfied = True
            condition = 'minstd'
        if not is_satisfied and std < self.minstd * np.median(
                np.abs(self.xmean)):
            is_satisfied = True
            condition = 'minstdrel'
        if not is_satisfied and np.any(
                1 + self.S[:self.k_active] > self.maxconds):
            is_satisfied = True
            condition = 'maxconds'
        if not is_satisfied and self.D.max() / self.D.min() > self.maxcondd:
            is_satisfied = True
            condition = 'maxcondd'
        return is_satisfied, condition


class ExponentialMovingAverage(object):
    """Exponential Moving Average, Variance, and SNR (Signal-to-Noise Ratio)

    See http://www-uxsup.csx.cam.ac.uk/~fanf2/hermes/doc/antiforgery/stats.pdf
    """

    def __init__(self, decay, dim, flg_init_with_data=False):
        """

        The latest N steps occupy approximately 86% of the information when
        decay = 2 / (N - 1).
        """
        self.decay = decay
        self.M = np.zeros(dim)  # Mean Estimate
        self.S = np.zeros(dim)  # Variance Estimate
        self.flg_init = -flg_init_with_data

    def update(self, datum):
        a = self.decay if self.flg_init else 1.
        self.S += a * ((1 - a) * (datum - self.M) ** 2 - self.S)
        self.M += a * (datum - self.M)


class TolfChecker(object):
    def __init__(self, size=20):
        """
        Parameters
        ----------
        size : int
            number of points for which the value is restored
        """
        self._min_hist = np.empty(size) * np.nan
        self._l_quartile_hist = np.empty(size) * np.nan
        # self._median_hist = np.empty(size) * np.nan
        self._u_quartile_hist = np.empty(size) * np.nan
        # self._max_hist = np.empty(size) * np.nan
        # self._pop_hist = np.empty(size) * np.nan
        self._next_position = 0

    def update(self, arf):
        self._min_hist[self._next_position] = np.nanmin(arf)
        self._l_quartile_hist[self._next_position] = np.nanpercentile(arf, 25)
        # self._median_hist[self._next_position] = np.nanmedian(arf)
        self._u_quartile_hist[self._next_position] = np.nanpercentile(arf, 75)
        # self._max_hist[self._next_position] = np.nanmax(arf)
        self._next_position = (
                                      self._next_position + 1) % self._min_hist.shape[0]

    def check(self, tolfun=1e-9):
        # alias to check_absolute
        return self.check_relative(tolfun)

    def check_relative(self, tolfun=1e-9):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        return iqr < tolfun * np.abs(np.nanmedian(self._min_hist))

    def check_absolute(self, tolfun=1e-9):
        iqr = np.nanmedian(self._u_quartile_hist - self._l_quartile_hist)
        return iqr < tolfun

    def check_flatarea(self):
        return np.nanmedian(self._l_quartile_hist - self._min_hist) == 0


def run_vkd_cma(problem_name: str, dimensionality: int, population_size: int, show_output: bool):
    assert problem_name in PROBLEM_CODES.keys(), "Unknown problem"
    problem_code = PROBLEM_CODES[problem_name]

    rotation_angle_1 = 45
    rotation_angle_2 = 45
    conditioning_number_1 = 6
    conditioning_number_2 = 6
    overlap_size = 0
    block_size = 5

    if problem_code > 100000:
        pid = problem_code
        rotation_angle_2 = (pid % 10) * 5
        pid = int(pid // 10)
        rotation_angle_1 = (pid % 10) * 5
        pid = int(pid // 10)
        conditioning_number_2 = pid % 10
        pid = int(pid // 10)
        conditioning_number_1 = pid % 10
        pid = int(pid // 10)
        overlap_size = pid % 10
        pid = int(pid // 10)
        block_size = pid

    def initialize_rotation_matrix(num_vars, rotation_angle):
        th = rotation_angle * np.pi / 180.0
        sinth = np.sin(th)
        costh = np.cos(th)

        rot_mat = np.identity(num_vars)
        mat = np.identity(num_vars)
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                mat[i][i] = costh
                mat[i][j] = -sinth
                mat[j][i] = sinth
                mat[j][j] = costh
                rot_mat = np.matmul(mat, rot_mat)
                mat[i][i] = 1.0
                mat[i][j] = 0.0
                mat[j][i] = 0.0
                mat[j][j] = 1.0
        return rot_mat

    def sphere(x):
        return np.sum(np.power(x, 2))

    def rosenbrock(x):
        sum = 0
        for i in range(len(x) - 1):
            sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        return sum

    def ellipsoid(x, cond_nr):
        res = 0
        n = len(x)
        for i in range(n):
            res += pow(10, cond_nr * (i / (n - 1))) * x[i] * x[i]
        return res
        # return np.dot( np.logspace(0, cond_nr, num=len(x), base=10, endpoint=True), x * x)

    def soreb(x, block_size, overlap_size, cond_nr, rotation_angle):
        res = 0
        i = 0
        global rotation_matrices
        if (block_size, rotation_angle) not in rotation_matrices.keys():
            rotation_matrices[(block_size, rotation_angle)] = initialize_rotation_matrix(block_size, rotation_angle)
        rot_mat = rotation_matrices[(block_size, rotation_angle)]
        while i + block_size <= len(x):
            r = np.matmul(rot_mat, x[i:i + block_size])
            res += ellipsoid(r, cond_nr)
            i += block_size - overlap_size
        return res

    def ellipsoid_grid(x):
        w = round(sqrt(len(x)))
        assert (w * w == len(x))
        res = 0.0
        for i in range(w):
            for j in range(w):
                y = [x[i * w + j]]
                if j + 1 < w:
                    y.append(x[i * w + j + 1])
                if i + 1 < w:
                    y.append(x[(i + 1) * w + j])
                if j > 0:
                    y.append(x[i * w + j - 1])
                if i > 0:
                    y.append(x[(i - 1) * w + j])
                res += soreb(y, len(y), 0, 6, -45)

        return res

    def osoreb(x):
        res = 0.0

        # Large blocks
        res += soreb(x, 5, 0, 6, 45)

        # Small blocks
        for large_block in range(1, dimensionality // 5):
            res += soreb(x[large_block * 5 - 1:large_block * 5 + 1], 2, 0, 6, 45)

        return res

    def osorebBig(x):
        res = 0.0

        # Large blocks
        res += soreb(x, 5, 0, 6, 45)

        # Small blocks
        for large_block in range(1, dimensionality // 5):
            res += soreb(x[large_block * 5 - 1:large_block * 5 + 1], 2, 0, 1, 5)

        return res

    def osorebSmall(x):
        res = 0.0

        # Large blocks
        res += soreb(x, 5, 0, 1, 5)

        # Small blocks
        for large_block in range(1, dimensionality // 5):
            res += soreb(x[large_block * 5 - 1:large_block * 5 + 1], 2, 0, 6, 45)

        return res

    def soreb_disjoint(x):
        res = 0.0

        for dual_block_index in range(dimensionality // 9):
            block_1 = dual_block_index * 9
            block_2 = dual_block_index * 9 + 4

            res += soreb(x[block_1:block_1 + 5], 5, 0, 6, 45)
            res += soreb(x[block_2:block_2 + 5], 5, 0, 1, 5)

        return res

    def reb_variant(x):
        # Homogeneous cases
        if rotation_angle_1 == rotation_angle_2 and conditioning_number_1 == conditioning_number_2:
            return soreb(x, block_size, overlap_size, conditioning_number_1, rotation_angle_1)

        res = 0.0

        stride = block_size - overlap_size
        block_index = 0
        for i in range(0, dimensionality, stride):
            c = conditioning_number_1 if block_index % 2 == 0 else conditioning_number_2
            r = rotation_angle_1 if block_index % 2 == 0 else rotation_angle_2
            res += soreb(x[i:i + block_size], block_size, 0, c, r)
            block_index += 1

        return res

    if problem_code > 100000:
        fobj = reb_variant
        assert dimensionality >= block_size  # num_variables > block_size
        assert overlap_size < block_size  # overlap < block_size
        assert (dimensionality - block_size) % (block_size - overlap_size) == 0  # no 'incomplete' blocks
    elif problem_name == "sphere":
        fobj = sphere
    elif problem_name == "rosenbrock":
        assert dimensionality >= 2
        fobj = rosenbrock
    elif problem_name == "reb5-disjoint-pairs":
        assert dimensionality % 9 == 0
        fobj = soreb_disjoint
    elif problem_name == "osoreb":
        assert dimensionality >= 10
        assert dimensionality % 5 == 0
        fobj = osoreb
    elif problem_name == "osoreb-big-strong":
        assert dimensionality >= 10
        assert dimensionality % 5 == 0
        fobj = osorebBig
    elif problem_name == "osoreb-small-strong":
        assert dimensionality >= 10
        assert dimensionality % 5 == 0
        fobj = osorebSmall
    elif problem_name == "reb-grid":
        assert abs((sqrt(dimensionality) ** 2) - dimensionality) < 1e-6
        fobj = ellipsoid_grid
    else:
        raise Exception("Unknown problem")

    xmean0 = -110. + 10. * randn(dimensionality)
    sigma0 = 10.

    x1 = np.ones(dimensionality)
    for i in range(dimensionality):
        x1[i] = i

    # Optional Parameters
    esoption = dict()
    esoption['lam'] = int(4 + 3 * log(dimensionality))
    if population_size > 0:
        esoption['lam'] = population_size
    esoption['ds'] = 4 - 3 / dimensionality  # sqrt(N) in PPSN
    # Termination Condition
    tcoption = dict()
    tcoption['ftarget'] = 1e-10
    tcoption['maxeval'] = 1e7  # int(5e3 * N * esoption['lam'])
    tcoption['tolf'] = 1e-16
    tcoption['tolfrel'] = 1e-16
    tcoption['minstd'] = 1e-12
    tcoption['minstdrel'] = 1e-12
    tcoption['maxconds'] = 1e12
    tcoption['maxcondd'] = 1e6
    # k-adaptation
    kaoption = dict()
    kaoption['kmin'] = 0
    kaoption['kmax'] = dimensionality - 1
    kaoption['k_init'] = kaoption['kmin']
    kaoption['k_inc_cond'] = 30.0
    kaoption['k_dec_cond'] = kaoption['k_inc_cond']
    kaoption['k_adapt_factor'] = 1.414
    kaoption['factor_sigma_slope'] = 0.1
    kaoption['factor_diag_slope'] = 1.0  # 0.3 in PPSN

    opts = dict()
    opts.update(esoption)
    opts.update(tcoption)
    opts.update(kaoption)

    tlimit = 3 * 60 * 60
    maxeval = 1e7
    tot_eval = 0
    t = time.time()
    n_restart = 25
    itr = 0
    r = 0
    success = False
    last_obj = -1
    while r < n_restart:
        tcoption['maxeval'] = maxeval - tot_eval
        vkd = VkdCma(fobj, xmean0, sigma0, **opts)
        satisfied = False
        condition = None
        while not satisfied:
            itr += 1
            vkd._onestep()
            satisfied, condition = vkd._check()
            if time.time() - t > tlimit:
                break

            if show_output and itr % 20 == 1:
                print(itr, vkd.neval, vkd.arf.min(), vkd.sigma, vkd.k)

        tot_eval += vkd.neval

        if show_output:
            print("CONDITION", condition, itr, vkd.neval, vkd.arf.min(), vkd.sigma, vkd.k, tot_eval)

        if condition == 'ftarget':
            success = True
            last_obj = vkd.arf.min()
            break
        if time.time() - t > tlimit:
            last_obj = vkd.arf.min()
            break
        r += 1

        if show_output:
            print("Restarting.")

    if not success:
        tot_eval = 1e7

    return itr, tot_eval, time.time() - t, last_obj


def main():
    parser = argparse.ArgumentParser(prog='Run VkD-CMA')

    parser.add_argument('-i', '--in-directory', type=str, default=".")
    parser.add_argument('-p', '--problem', type=str, required=True)
    parser.add_argument('-d', '--dimensionality', type=int, required=True)
    parser.add_argument('-s', '--population-size', type=int, required=True)
    parser.add_argument('-o', '--show-output', action="store_true")
    parser.add_argument('-r', '--random-seed', type=int, default=-1)

    args = parser.parse_args()

    if args.random_seed != -1:
        np.random.seed(args.random_seed)

    generations, evaluations, seconds, best_objective = run_vkd_cma(
        args.problem.lower(), args.dimensionality, args.population_size, args.show_output
    )

    directory = args.in_directory
    os.system(f"mkdir -p {directory}")

    # Save run config for future reference
    config = {
        "random_seed": args.random_seed,
        "problem": args.problem.lower(),
        "dimensionality": args.dimensionality,
        "population_size": args.population_size,
        "linkage_model": "vkd-cma",
    }
    with open(os.path.join(directory, "run_config.json"), "w") as f:
        f.write(json.dumps(config, indent=4))

    # Convert statistics to CSV
    statistics = pd.DataFrame([{
        "generations": generations,
        "evaluations": evaluations,
        "seconds": seconds,
        "best_objective": best_objective,
    }])
    assert len(statistics) > 0, f"No generations were executed with RunConfig {config}"

    statistics.to_csv(os.path.join(directory, "statistics.csv"))


rotation_matrices = {}
if __name__ == '__main__':
    main()
