#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import interp1d


def get_max_ind(amp):
    return (np.where(amp == np.max(amp)))


def get_arr_inds_closest_to_point(X, Y, point):
    # Calculate the Euclidean distance between each element and the target
    # point
    distances = np.sqrt((X - point[0])**2 + (Y - point[1])**2)
    # Find the index of the element with the smallest distance
    index = np.unravel_index(np.argmin(distances), distances.shape)
    return index


def multi_cuts(x, y, amp, est_center=True,
               center_guess=(0, 0), center_est_rad=5, center_est_func="est",
               max_r='auto', rad_bin_size='auto', keep_len='auto'):
    '''Assumed amplitude is NOT squared'''
    # Make sure the arrays are inputted as we want.
    if np.sum(np.diff(x[0]) != 0):
        # print("swapping x and y for the cuts...")
        y, x = x, y
        amp = amp.T
    if np.sum(np.diff(y[:, 0]) != 0):
        raise Exception("arrays are not in the correct format!")
    power = (amp ** 2)
    index = get_max_ind(amp)
    max_point = (x[index], y[index])
    if est_center:
        if center_est_func == "est":
            center_estimate = estimate_center(x, y, amp, max_point, [[
                -center_est_rad, center_est_rad], [-center_est_rad,
                                                   center_est_rad]])
            index = get_arr_inds_closest_to_point(
                x, y, center_estimate)
        elif center_est_func == "max":
            index = (index[0][0], index[1][0])
            center_estimate = max_point
        else:
            raise Exception
    else:
        center_estimate = center_guess
        index = get_arr_inds_closest_to_point(
            x, y, center_estimate)

    # change the center estimate to just the pixel location
    square_center = (x[index], y[index])
    if keep_len == 'auto':
        keep_len = int(np.floor(2 * np.min([
            abs(len(x) - abs(index[0])),
            abs(len(y) - abs(index[1])),
            abs(0 - abs(index[0])),
            abs(0 - abs(index[1])),
        ])))
    # now isolate a square around this center
    power_square = power[index[0] - keep_len // 2: index[
        0] + keep_len // 2 + 1, index[1] - keep_len // 2: index[
            1] + keep_len // 2 + 1]
    x_square = x[index[0] - keep_len // 2: index[0] + keep_len // 2 + 1, index[
        1] - keep_len // 2: index[1] + keep_len // 2 + 1]
    y_square = y[index[0] - keep_len // 2: index[0] + keep_len // 2 + 1, index[
        1] - keep_len // 2: index[1] + keep_len // 2 + 1]
    if max_r == 'auto':
        # just slightly less than x_max * sqrt(2)
        max_r = np.max((x_square - center_estimate[0])) * 1.4
    # make sure the center of out square is the center!
    # Now take X, Y, 45, 135 cuts
    cut_x = power_square[:, keep_len // 2]
    cut_x_vals = x_square[:, keep_len // 2]
    cut_x_fwhm = get_fwhm(
        cut_x_vals, cut_x, interpolate=True)[2]
    cut_y = power_square[keep_len // 2, :]
    cut_y_vals = y_square[keep_len // 2, :]
    cut_y_fwhm = get_fwhm(
        cut_y_vals, cut_y, interpolate=True)[2]
    # Take a 45 degree cut
    cut_45 = np.diag(power_square, k=0)
    cut_45_x = np.diag(x_square, k=0)
    cut_45_y = np.diag(y_square, k=0)
    cut_45_r = np.sqrt((cut_45_x - square_center[0]) ** 2 + (
        cut_45_y - square_center[1]) ** 2)
    cut_45_r[:keep_len // 2] *= -1
    cut_45_fwhm = get_fwhm(
        cut_45_r, cut_45, interpolate=True)[2]
    # Take a 135 degree cut
    cut_135 = np.diag(np.fliplr(power_square), k=0)
    cut_135_x = np.diag(np.fliplr(x_square), k=0)
    cut_135_y = np.diag(np.fliplr(y_square), k=0)
    cut_135_r = np.sqrt((cut_135_x - square_center[0]) ** 2 + (
        cut_135_y - square_center[1]) ** 2)
    cut_135_r[:keep_len // 2] *= -1
    cut_135_fwhm = get_fwhm(
        cut_135_r, cut_135, interpolate=True)[2]
    # take a radial cut
    if rad_bin_size == 'auto':
        ydiff = np.diff(y[0])[0]
        rad_bin_size = ydiff / 1.4
    radii, means = rad_avg(
        x_square - center_estimate[0], y_square - center_estimate[1],
        np.sqrt(power_square), 0, max_r, rad_bin_size, shift_center=False)
    rad_fwhm = get_fwhm_radial_bins(
        radii, means, interpolate=True)
    data = {'center': center_estimate, 'power': power_square, 'x': x_square,
            'y': y_square, 'cut_x': cut_x, 'cut_y': cut_y, 'cut_45': cut_45,
            'cut_135': cut_135, 'cut_x_vals': cut_x_vals,
            'cut_y_vals': cut_y_vals,
            'cut_45_vals': (cut_45_x, cut_45_y, cut_45_r),
            'cut_135_vals': (cut_135_x, cut_135_y, cut_135_r),
            'x_fwhm': cut_x_fwhm, 'y_fwhm': cut_y_fwhm,
            '45_fwhm': cut_45_fwhm, '135_fwhm': cut_135_fwhm,
            'radii': radii, 'means': means, 'rad_fwhm': rad_fwhm,
            'index': index, 'square_center': square_center,
            'amp': np.sqrt(power_square)}
    return data


def get_fwhm(x, y, val_to_use=None, interpolate=False,
             restrict_range=None):
    if restrict_range is not None:
        keep_inds = np.where((x >= restrict_range[0]) & (
            x <= restrict_range[1]))
        x = x[keep_inds]
        y = y[keep_inds]
    if not val_to_use:
        val_to_use = np.max(y) * .5

    if interpolate:
        x_diff = x[1] - x[0]
        interp_func = interp1d(x, y)
        x_interp = np.arange(np.min(x), np.max(x) - x_diff, x_diff / 10)
        # x_interp = np.linspace(np.min(x), np.max(x), 1000)
        y_interp = interp_func(x_interp)
        x, y = (x_interp, y_interp)
    d = y - val_to_use
    inds = np.where(d > 0)[0]
    fwhm = x[inds[-1]] - x[inds[0]]
    return x, inds, fwhm


def get_fwhm_radial_bins(r, y, val_to_use=None, interpolate=False):
    if not val_to_use:
        val_to_use = np.max(y) * .5

    if interpolate:
        r_diff = r[1] - r[0]
        interp_func = interp1d(r, y)
        r_interp = np.arange(np.min(r), np.max(r) - r_diff, r_diff / 100)
        y_interp = interp_func(r_interp)
        r, y = (r_interp, y_interp)
    d = y - val_to_use
    inds = np.where(d > 0)[0]
    fwhm = 2 * (r[inds[-1]])
    return fwhm


def rad_avg(x_arr, y_arr, beam, rmin, rmax, inc, shift_center,
            use_center=None, data_limits=[[-25, 25], [-25, 25]]):
    """Compute radial average of measured beam over range [0,deg] of radii.

    the beam should be in AMPLITUDE not in POWER (don't square-- that is
    done in here.)

    x_arr,y_arr,rmin,rmax,inc [arcmin]

    """
    # we might have to find the centroid first..
    if use_center is None:
        shift_x = x_arr[np.where(beam == np.max(beam))]
        shift_y = y_arr[np.where(beam == np.max(beam))]
        # print(shift_x, shift_y)
        # plt.pcolormesh(x_arr - shift_x, y_arr - shift_y, to_db(abs(beam)),
        #                shading='auto', vmin=-40, vmax=0)
        # plt.plot([0], [0], '.', ms=15, color='black')
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        # plt.colorbar()
        # plt.show()

        # TODO: remove this magic number
        cen = centroid(x_arr - shift_x, y_arr - shift_y, abs(beam) ** 2,
                       data_limits=data_limits)
        shift_x += cen[0]
        shift_y += cen[1]
        # cen = centroid(x_arr - shift_x, y_arr - shift_y, abs(beam) ** 2,
        #                data_limits=[[-25, 25], [-25, 25]])
        # print(f'final centroid = {cen}')

        # plt.subplots(figsize=(6, 2))
        # plt.plot([0], [0], '.', ms=15, color='black')
        # plt.plot(cen[0], cen[0], '.', ms=15, color='black')
        # # plt.contourf(x_arr - shift_x, y_arr - shift_y, to_db(abs(beam)))
        # plt.pcolormesh(x_arr - shift_x, y_arr - shift_y, to_db(abs(beam)),
        #                shading='auto', vmin=-40, vmax=0)
        # plt.colorbar()
        # # plt.xlim(-100, 100)
        # # plt.ylim(-100, 100)
        # plt.show()
    else:
        shift_x, shift_y = use_center

    if shift_center:
        r_i = np.sqrt((x_arr-shift_x) ** 2 + ((y_arr-shift_y) ** 2))
    else:
        r_i = np.sqrt((x_arr) ** 2 + ((y_arr) ** 2))

    # calculate the mean
    # use the power (i.e. square the amplitude)
    def rad_bins(r_b):
        return (abs(beam) ** 2)[(r_i >= r_b - (inc)) & (r_i < r_b + (
            inc))].mean()

    # Adjust old code from grace: remove magic number to make this scalable
    # r_phi = np.linspace(rmin, rmax, num=200)
    r_phi = np.arange(rmin, rmax, inc)

    mean = np.vectorize(rad_bins)(r_phi)

    max_mean = np.max(mean[np.where(np.isnan(mean) == False)])

    # don't divide by maximum yet
    return r_phi, mean# / max_mean


def centroid(x_arr, y_arr, data, data_limits=None):
    if data_limits is not None:
        x_limits = ((x_arr > data_limits[0][0]) & (
            x_arr < data_limits[0][1]))
        y_limits = ((y_arr > data_limits[1][0]) & (
            y_arr < data_limits[1][1]))
        total_limits = (x_limits & y_limits)
        weights = np.zeros(data.shape)
        weights[total_limits] = data[total_limits]

    x_center = np.average(x_arr, weights=weights)
    y_center = np.average(y_arr, weights=weights)
    return (x_center, y_center)


def estimate_center(x2d, y2d, beam2d, center_guess, size_limits):
    """
    Reduce beam to size_limits around center_guess and find the center.

    Estimates via centroid.
    """
    centroid_estimate = centroid(x2d - center_guess[0], y2d - center_guess[1],
                                 beam2d, size_limits)
    # add the guess back to our centroid to put in proper coordinates
    return (centroid_estimate[0] + center_guess[0],
            centroid_estimate[1] + center_guess[1])


def roll_beam(beam, x_roll, y_roll):
    beam_roll = np.roll(np.roll(beam, y_roll, axis=0), x_roll, axis=1)
    return beam_roll


def restrict_beam(x2d, y2d, beam2d, beam_center, beam_size,
                  return_beam=True, noise_floor=1e-6):
    """restrict to within (x, y) within dist beam_size of beam_center.

    if return_beam is set to True, return the main beam region.
    Otherwise return the region outside of the beam!
    """
    new_beam = np.copy(beam2d)
    if return_beam:
        bad_inds = np.where(np.sqrt((x2d - beam_center[0]) ** 2 + (
            y2d - beam_center[1]) ** 2) > beam_size)
    else:
        bad_inds = np.where(np.sqrt((x2d - beam_center[0]) ** 2 + (
            y2d - beam_center[1]) ** 2) <= beam_size)
    new_beam[bad_inds] = np.max(beam2d) * noise_floor
    return new_beam


def to_db(amp, norm_factor=None, square=True):
    """Convert data in linear scale to squared db scale.

    Arguments:

    data - (array) data to convert
    norm_factor - (float) factor which we divide the data by before converting.
                  if None, just normalize by max(abs(data)).

    (Take abs value, square, go to log space.)
    """
    if square:
        factor = 20
    else:
        factor = 10
    if norm_factor is None:
        return factor * np.log10(abs(amp) / np.max(abs(amp)))
    return factor * np.log10(abs(amp) / norm_factor)


def db_to_percent(db_level):
    """Convert data in db to percent."""
    return (10 ** (db_level / 10)) * 100


def percent_to_db(percent_level):
    """Convert data in percent to db."""
    return 10 * np.log10(percent_level / 100)


def get_complex_beam(beam, phase):
    """phase in degrees"""
    return abs(beam) * np.exp(complex(0, 1) * np.deg2rad(phase))


def complex_to_amp_phase(complex_beam):
    amp = np.abs(complex_beam)
    phase = np.arctan2(np.imag(complex_beam), np.real(complex_beam))
    return amp, phase


def normalize(arr):
    return arr / np.max(arr)


def ellipticity(sigma_max, sigma_min):
    return (sigma_max - sigma_min) / (sigma_max + sigma_min)


def zero_pad(x_in, y_in, beam_in, pts):
    x_int = abs(x_in[0, 0] - x_in[0, 1])
    y_int = abs(y_in[0, 0] - y_in[1, 0])
    beam_out = np.pad(beam_in, pts, mode="constant")
    x_new = np.array(np.arange(len(beam_out)))
    y_new = np.array(np.arange(len(beam_out)))
    x_new = x_new - np.mean(x_new)
    y_new = y_new - np.mean(y_new)
    x_new = x_new * x_int
    y_new = y_new * y_int
    x_new, y_new = np.meshgrid(x_new, y_new)
    return x_new, y_new, beam_out


def coords_spat_to_ang(x, y, freq):
    """
    Coordinate transformation from spatial to angular.

    x and y are in units of [m]
    """
    ff_ghz = freq * 1e9  # frequency in [Hz]
    # Get spatial coordinates
    lam = (3 * 10 ** 8) / ff_ghz  # wavelength in [m]

    # Resolution in aperture plane [m]
    delta_x = abs(np.max(x) - np.min(x)) / (len(x) - 1)  # increment in x
    delta_y = abs(np.max(y) - np.min(y)) / (len(y) - 1)  # increment in y

    x_len = len(x)
    y_len = len(y)

    alpha = lam / delta_x
    beta = lam / delta_y

    # Conversion for spatial to angular
    delta_th = alpha / x_len
    delta_ph = beta / y_len

    x_ang = np.linspace(-int((len(x) / 2)),
                        int((len(x) / 2)), int((len(x)))) * delta_th
    y_ang = np.linspace(-int((len(x) / 2)),
                        int((len(x) / 2)), int((len(x)))) * delta_ph
    x_ang, y_ang = np.meshgrid(x_ang, y_ang)
    return x_ang, y_ang


def coords_ang_to_spat(theta_x, theta_y, freq):
    """
    Coordinate transformation from angular to spatial.

    theta_x and theta_y are in units of [rad]
    Returns units of [m]
    """
    ff_ghz = freq * 1e9
    # Get spatial coordinates
    lam = (3 * 10 ** 8) / ff_ghz
    delta_th = abs(np.max(theta_x) - np.min(theta_x)) / (
        len(theta_x) - 1
    )  # increment in azimuthal angle
    delta_th = abs(np.max(theta_y) - np.min(theta_y)) / (
        len(theta_y) - 1
    )  # increment in azimuthal angle

    x_len = len(theta_x)
    y_len = len(theta_y)

    alpha = lam / delta_th  # increment in x
    beta = lam / delta_th

    delta_x = alpha / x_len  # spatial coordinates conversion
    delta_y = beta / y_len

    x_spat = np.linspace(-int((len(theta_x) / 2)), int((len(
        theta_x) / 2)), int((len(theta_x)))) * delta_x
    y_spat = np.linspace(-int((len(theta_x) / 2)), int((len(
        theta_x) / 2)), int((len(theta_x)))) * delta_y
    x_spat, y_spat = np.meshgrid(x_spat, y_spat)
    return x_spat, y_spat


def sigma_to_fwhm(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def freq_to_wavelength(freq):  # in GHz
    c = 3e8
    return c / (freq * 1e9)  # returns in m


def arcmin2rad(x):
    return (x / 60 / (180 / np.pi))


def rad2arcmin(x):
    return x * 60 * (180 / np.pi)

def noise_floor(x, y, beam, center, beam_rad, set_level=1e-6, debug=False):
    # set the main beam part to zero with given radius
    beam = np.copy(beam)
    beam = abs(beam) / np.max(abs(beam))
    beam = restrict_beam(x, y, beam, center, beam_rad,
                         noise_floor=set_level, return_beam=False)
    beam[beam == set_level] = np.nan

    if debug:
        import matplotlib.pyplot as plt
        from . import plot_beams as pb
        plt.plot(beam)
        plt.axhline(np.nanmedian(beam), )
        plt.axhline(np.nanmedian(beam), color="black",
                    label=f'noise floor = {np.nanmedian(beam):.1e}')
        # print(f'{np.nanstd(beam):.2e}')
        plt.yscale('log')
        plt.legend()
        plt.ylim(set_level, 1)
        plt.show()
        pb.plot_beam(x, y, abs(beam), norm_factor=1)
        plt.show()
    # take the median now
    return np.nanmedian(beam)

def corner_noise_floor(x, y, beam, center, beam_rad, set_level=1e-6,
                       debug=False, corner_width=5):
    # set the main beam part to zero with given radius
    beam = np.copy(beam)
    beam = abs(beam) / np.max(abs(beam))
    # beam = restrict_beam(x, y, beam, center, beam_rad,
    #                      noise_floor=set_level, return_beam=False)
    # beam[beam == set_level] = 0
    bottom_left = ((abs(x - np.min(x)) <= corner_width) & (
        abs(y - np.min(y)) <= corner_width))
    bottom_right = ((abs(x - np.max(x)) <= corner_width) & (
        abs(y - np.min(y)) <= corner_width))
    top_left = ((abs(x - np.min(x)) <= corner_width) & (
        abs(y - np.max(y)) <= corner_width))
    top_right = ((abs(x - np.max(x)) <= corner_width) & (
        abs(y - np.max(y)) <= corner_width))
    corners = np.where((bottom_left | bottom_right | top_left | top_right))
    return np.nanmedian(beam[corners])

    # beam[corners] = np.nan
    import matplotlib.pyplot as plt
    plt.pcolormesh(x, y, 10 * np.log10(beam), vmin=-30)
    plt.colorbar()
    plt.show()

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(beam)
        plt.axhline(np.nanmedian(beam), )
        plt.axhline(np.nanmedian(beam), color="black",
                    label=f'noise floor = {np.nanmedian(beam):.1e}')
        # print(f'{np.nanstd(beam):.2e}')
        plt.yscale('log')
        plt.legend()
        plt.ylim(set_level, 1)
        plt.show()
    # take the median now
    return np.nanmedian(beam)
