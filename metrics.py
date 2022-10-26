import numpy as np

def rel_err(xcorr, xtrue):
    xcorr = xcorr.flatten()
    xtrue = xtrue.flatten()

    return np.linalg.norm(xcorr - xtrue) / np.linalg.norm(xtrue)

def PSNR(original, compressed):
    original = original.flatten()
    compressed = compressed.flatten()

    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr