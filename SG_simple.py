import numpy as np
from scipy.signal import savgol_coeffs
from scipy.ndimage import convolve1d


COEFFS_LONG_TREND = savgol_coeffs(17, 2)
COEFFS_SHORT_TREND = savgol_coeffs(17, 6)


def simple_sg_filter(curve_0, max_iteration=10):
    curve_tr = convolve1d(curve_0, COEFFS_LONG_TREND, mode="wrap")
    d = curve_tr - curve_0
    dmax = np.max(np.abs(d))
    dmax = max(dmax, 1e-10)
    w_func = np.frompyfunc(lambda d_i: min((1, 1 - d_i/dmax)), 1, 1)
    W = w_func(d)
    curve_k = np.copy(curve_tr)
    #print("数据类型：", curve_k.dtype)
    f_arr = np.zeros(max_iteration)
    curve_previous = None
    for i in range(max_iteration):
        curve_k = np.maximum(curve_k, curve_0)
        curve_k = convolve1d(curve_k, COEFFS_SHORT_TREND, mode="wrap")
        f_arr[i] = np.sum(np.abs(curve_k - curve_0) * W)
        if i >= 1 and f_arr[i] > f_arr[i - 1]:
            return curve_previous
        curve_previous = curve_k
    return curve_previous
