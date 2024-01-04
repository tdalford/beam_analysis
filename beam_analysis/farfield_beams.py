#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from . import utils
from . import plot_beams as pb


def fit_beam_size_2d_gaussian(x_arcmin, y_arcmin, beam_ff, beam_power_limit=-6,
                              data_limits=[[-250, 250], [-250, 250]],
                              segment_data=True, debug=False, guess_sigma=25,
                              **debug_multi_cuts_kwargs):
    """Make sure we choose the right dimensions of our fitting square...

    (i.e. always go out to 12 db out so.)

    Ensure that the beam is centered. Or make it an option to center before
    fitting!
    """
    # try to constrain with ~6db of center (tunable)
    # and also make sure we constrain within roughly ~25 arcmin at max
    if segment_data:
        beam_area = (utils.to_db(beam_ff) >= beam_power_limit)
        x_limits = ((x_arcmin >= data_limits[0][0]) & (
            x_arcmin <= data_limits[0][1]))
        y_limits = ((y_arcmin >= data_limits[1][0]) & (
            y_arcmin <= data_limits[1][1]))
        # cut out data outside of this main beam
        total_limits = np.where((x_limits & y_limits) & beam_area)
        x_size = np.max(total_limits[0]) - np.min(total_limits[0])
        y_size = np.max(total_limits[1]) - np.min(total_limits[1])
        pix_size = np.max((x_size, y_size))  # maybe take the max?
        # make this defined by the circumscribed circle
        pix_size = int(pix_size / np.sqrt(2))

        # use the center index of our square to fit
        max_ind = [int(np.floor(x_arcmin.shape[0] // 2)),
                int(np.floor(x_arcmin.shape[1] // 2))]

        start_x, stop_x = max_ind[0] - pix_size // 2, max_ind[
            0] + pix_size // 2
        start_y, stop_y = max_ind[1] - pix_size // 2, max_ind[
            1] + pix_size // 2

        fit_x = x_arcmin[start_x: stop_x, start_y: stop_y]
        fit_y = y_arcmin[start_x: stop_x, start_y: stop_y]
        fit_data = beam_ff[start_x: stop_x, start_y: stop_y]

    else:
        fit_x = x_arcmin
        fit_y = y_arcmin
        fit_data = beam_ff


    def gauss_2d(xy_data, amp, x0, y0, sigma_x, sigma_y, theta):
        x, y = xy_data
        a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2)/(
            2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(
            4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(
            2*sigma_y**2)
        g = amp*np.exp(- (a*((x-x0)**2) + 2*b*(x-x0)*(
            y-y0) + c*((y-y0)**2)))
        # g = utils.to_db(g, square=False)
        return g.ravel()

    guess = (1, 0, 0, guess_sigma, guess_sigma, 0)
    bounds = (0, [np.inf, np.inf, np.inf, np.inf, np.inf, np.pi / 4])
    popt, pcov = curve_fit(gauss_2d, (fit_x, fit_y),
                           # utils.to_db(fit_data).ravel(),
                           (fit_data ** 2).ravel(),
                           p0=guess,
                           bounds=bounds)

    if debug:
        # print('popt: %s' % popt)
        # print('angle = ', popt[-1] * 180 / np.pi)
        data_square = utils.multi_cuts(fit_x, fit_y, fit_data,
                                       **debug_multi_cuts_kwargs)
        pb.plot_data_cuts(data_square, log=False)
        plt.show()
        xlen = len(fit_x)
        data_fitted = gauss_2d((fit_x, fit_y), *popt).reshape(xlen, xlen)
        fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(fit_x, fit_y, utils.to_db(fit_data), shading='auto')
        plt.colorbar(label='[dB]')
        cs = plt.contour(fit_x, fit_y, utils.to_db(data_fitted, square=False),
                         cmap='viridis', vmin=np.min(utils.to_db(fit_data)),
                         vmax=0, levels=[-30, -20, -15, -10, -3],)
        plt.clabel(cs)
        fit_square = utils.multi_cuts(fit_x, fit_y, np.sqrt(
            data_fitted), **debug_multi_cuts_kwargs)
        plt.subplot(1, 2, 2)
        plt.plot(data_square['radii'], data_square['means'],
                 label=f'data: {data_square["rad_fwhm"]:.2f}')
        plt.plot(fit_square['radii'], fit_square['means'],
                 label=f'fit: {fit_square["rad_fwhm"]:.2f}')
        # plt.xlim(0, fit_square['rad_fwhm'] * 1.3)
        plt.ylim(1e-4, 1.05)
        # plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()

    sigma_x, sigma_y = popt[[3, 4]]
    data = {'x_fwhm': utils.sigma_to_fwhm(sigma_x),
            'y_fwhm': utils.sigma_to_fwhm(sigma_y),
            'average_fwhm': utils.sigma_to_fwhm((sigma_x + sigma_y) / 2),
            'center': (popt[1], popt[2]),
            'angle':  popt[-1] * 180 / np.pi,
            # 'popt': popt,
            # 'pcov': pcov,
            }
    return data


def forward_gain(az, el, beam):
    '''Compute the forward gain of a beam from the solid angle integral.

    az and el are given in ARCMIN

    return value is in nano-steradians  (nsr)
    '''

    # convert from arcmin to rad
    az = utils.arcmin2rad(az)
    el = utils.arcmin2rad(el)
    # cos(theta) factor to multiply for the solid angle integral
    integrand = beam * np.cos(el)
    # perform the solid angle integral
    integral = np.trapz(np.trapz(integrand, el, axis=0), az, axis=0)
    return integral * 1e9
