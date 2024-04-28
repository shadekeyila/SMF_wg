# -*- coding:utf-8 -*-

import numpy as np
from numba import jit

@jit(nopython=True)
def shape_change(t, p0, tshift, xscale, aug_ts, trans_ts):

    t = t * xscale + (1 - xscale) * p0 + tshift * xscale
    for i, t_ in enumerate(t):
        if t_ <= 0:
            t[i] = 0
        if t_ >= 365:
            t[i] = 365
        trans_ts[i] = aug_ts[int(t[i] * 100)]
    return

@jit(nopython=True)
def pcc(x, y, length):
    xmm, ymm = x - np.mean(x), y - np.mean(y)
    tp1, tp2, tp3 = 0.0, 0.0, 0.0
    for i in range(length):
        tp1 += xmm[i] * ymm[i]
        tp2 += xmm[i] * xmm[i]
        tp3 += ymm[i] * ymm[i]
    tp2, tp3 = np.sqrt(tp2), np.sqrt(tp3)
    return tp1 / (tp2 + 1e-6) / (tp3 + 1e-6)


class SMFS:
    P0 = 0
    ref_vt = None
    T = None
    w = 45
    dtol = 1e-1
    max_IT = 10
    R = 0.8
    xscale_rang = (0.7, 1.3,)
    tshift_rang = (-30, 30,)
    tshift_IT_range = 4
    xscale_IT_range = 0.2
    tshift_search_b = 1
    xscale_search_b = 0.05
    day_int = 8
    ref_vt_a = None
    T_a = np.arange(0, 365, 0.01)
    temp_trans = np.zeros((46,))

    def __init__(self, ref_vt, p0, t):
        self.ref_vt = ref_vt
        self.T = t
        self.day_int = int(self.T[1] - self.T[0])
        self.ref_vt_a = np.interp(self.T_a, t, ref_vt)
        self.P0 = p0
        self.length = len(ref_vt)


    def shape_change_faster(self, tshift, xscale):
        shape_change(self.T, self.P0, tshift, xscale, self.ref_vt_a, self.temp_trans)
        return self.temp_trans

    def difference(self, tshift, xscale, tar_ts):
        shape_change(self.T, self.P0, tshift, xscale, self.ref_vt_a, self.temp_trans)

        t0, length = int((self.P0 - tshift - self.w) / self.day_int), int(self.w / self.day_int) * 2 + 1

        if t0 < 0:
            t0 = 0
        if length >= self.length - t0 - 1:
            length = self.length - t0 - 1

        x, y = self.temp_trans[t0: t0 + length], tar_ts[t0: t0 + length]

        r = pcc(x, y, length)
        return 1 - r

    def TSHIFT_SEARCH(self, tshift, xscale, tar_ts):
        if tshift > 1e4:
            search_locs = np.arange(self.tshift_rang[0], self.tshift_rang[1])
            xscale = 1

        else:
            search_locs = np.arange(np.max((tshift - self.tshift_IT_range, self.tshift_rang[0],)),
                                    np.min((tshift + self.tshift_IT_range + 1, self.tshift_rang[1],)))
        r_arr = np.zeros(len(search_locs))
        for i, v in enumerate(search_locs):
            r_arr[i] = self.difference(v, xscale, tar_ts)
        return search_locs[np.argmin(r_arr)]

    def XSCALE_SEARCH(self, tshift, xscale, tar_ts):

        if xscale > 1e4:
            search_locs = np.arange(self.xscale_rang[0], self.xscale_rang[1], self.xscale_search_b)
        else:
            search_locs = np.arange(np.max((xscale - self.xscale_IT_range, self.xscale_rang[0],)),
                                    np.min((xscale + self.xscale_IT_range + 1, self.xscale_rang[1],)),
                                    self.xscale_search_b)
        loss_arr = np.zeros(len(search_locs))
        for i, v in enumerate(search_locs):
            loss_arr[i] = self.difference(tshift, v, tar_ts)
        i_min = np.argmin(loss_arr)
        return search_locs[i_min], 1 - loss_arr[i_min]

    def runit(self, tar_ts):
        tshift_previous = np.inf
        tshift, xscale = np.inf, np.inf
        iter, r = 0, - np.inf
        while iter < self.max_IT:
            tshift = self.TSHIFT_SEARCH(tshift, xscale, tar_ts)
            if abs(tshift_previous - tshift) < self.dtol:
                break
            tshift_previous = tshift
            xscale, r = self.XSCALE_SEARCH(tshift, xscale, tar_ts)
            iter += 1
        if r < self.R:
            return 0
        else:
            return self.P0 - tshift



