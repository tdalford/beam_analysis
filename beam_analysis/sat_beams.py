#!/usr/bin/env python3
import numpy as np
import scipy
from . import utils

PLOT_DIR = '/Users/talford/software/CMB/simons/holography/sat_holography_data/data/'
FWHM_ESTIMATE = 25 / (60 * (180 / np.pi))  # in radians


def load_data(freq):
    beam_co = np.load(PLOT_DIR + '%d_copolar/' %
                      freq + 'amp_%d_copolar.npy' % freq)
    beam_cross = np.load(
        PLOT_DIR + '%d_crosspolar/' % freq + 'amp_%d_crosspolar.npy' % freq)
    phase_co = np.load(PLOT_DIR + '%d_copolar/' %
                       freq + 'phase_%d_copolar.npy' % freq)
    phase_cross = np.load(
        PLOT_DIR + '%d_crosspolar/' % freq + 'phase_%d_crosspolar.npy' % freq)
    return beam_co, beam_cross, phase_co, phase_cross


def convert_to_farfield(x2d, y2d, beam2d, phase2d, pix_num, frequency):
    """x2d and y2d are in units of [cm]"""
    # zero pad the beam
    x_new, y_new, beam_new = utils.zero_pad(x2d, y2d, beam2d, pix_num)
    _, _, phase_new = utils.zero_pad(x2d, y2d, phase2d, pix_num)

    beam_complex = abs(beam_new) * np.exp(complex(0, 1)
                                          * np.deg2rad(phase_new))
    beam_temp = np.fft.fftshift(beam_complex)
    beam_temp = np.fft.fft2(beam_temp)
    beam_ff = np.fft.fftshift(beam_temp)
    x_ang, y_ang = utils.coords_spat_to_ang(
        x_new / 1e2, y_new / 1e2, frequency)
    return x_ang, y_ang, beam_ff


def cut_farfield(x_ang, y_ang, beam_ff, fwhm_cut_num):
    max_ind = np.unravel_index(np.argmax(np.abs(beam_ff), axis=None),
                               beam_ff.shape)
    max_pos = (x_ang[max_ind], y_ang[max_ind])
    dist = np.sqrt((x_ang - max_pos[0]) ** 2 + (y_ang - max_pos[1]) ** 2)
    new_beam_ff = np.copy(beam_ff)
    new_beam_ff[np.where(dist >= fwhm_cut_num * FWHM_ESTIMATE)] = 0.0001
    return new_beam_ff


def fft_filter_beam(x2d, y2d, beam2d, phase2d, fwhm_cut_num, frequency):
    # don't zero pad
    x_ang, y_ang, beam_ff = convert_to_farfield(x2d, y2d, beam2d, phase2d, 0,
                                                frequency)
    beam_ff_cut = cut_farfield(x_ang, y_ang, beam_ff, fwhm_cut_num)
    return convert_to_nearfield(beam_ff_cut)


def convert_to_nearfield(beam_ff):
    beam_temp = np.fft.fftshift(beam_ff)
    beam_temp = np.fft.ifft2(beam_temp)
    beam_nearfield = np.fft.fftshift(beam_temp)
    return beam_nearfield


def add_phase_offset(complex_beam, phase_offset):
    return complex_beam * np.exp(complex(0, 1) * phase_offset)


def rotate_beam(phi, beam_co, beam_cr):
    phi_rad = phi * np.pi / 180
    # beamco = (np.cos(phi_rad) * beam_co - (np.sin(phi_rad) * beam_cr))
    beamcr = (np.sin(phi_rad) * beam_co + (np.cos(phi_rad) * beam_cr))
    return beamcr


def total_power(e_data):
    return np.sum(np.abs(e_data) ** 2)


def get_total_power(x2d, y2d, theta, beam_co, beam_cross,
                    bool_restrict_beam=True):
    rot_beam = rotate_beam(theta, beam_co, beam_cross)
    # restricted_beam = restrict_beam(x2d, y2d, rot_beam, (-15, 0), 25)
    if bool_restrict_beam:
        center_guess = (-15, 0)
        beam_center = utils.estimate_center(x2d, y2d, abs(
            beam_co), center_guess, [[-25, 25], [-25, 25]])
        # print(beam_center)
        restricted_beam = utils.restrict_beam(x2d, y2d, rot_beam, beam_center,
                                              25)
        # plt.pcolormesh(x2d, y2d, restricted_beam, shading='auto')
        # plt.plot([beam_center[0]], [beam_center[1]], '.', ms=15, color='black')
        # plt.show()
        return total_power(restricted_beam)
    return total_power(rot_beam)


def optimize_minimum_angle(x2d, y2d, beam_co, beam_cross):
    def power(theta):
        return get_total_power(x2d, y2d, theta, beam_co, beam_cross)
    return scipy.optimize.minimize(power, 0).x[0]


def optimize_maximum_angle(x2d, y2d, beam_co, beam_cross):
    def power(theta):
        return -1 * get_total_power(x2d, y2d, theta, beam_co, beam_cross)
    return scipy.optimize.minimize(power, 0).x[0]


def find_min_rotation_angle(x2d, y2d, beam_co, beam_cross):
    # thetas = np.linspace(0, 180, n_theta)
    min_angle = optimize_minimum_angle(x2d, y2d, beam_co, beam_cross)
    max_angle = optimize_maximum_angle(x2d, y2d, beam_co, beam_cross)

    min_power = get_total_power(x2d, y2d, min_angle, beam_co, beam_cross)
    max_power = get_total_power(x2d, y2d, max_angle, beam_co, beam_cross)
    # print(min_power, max_power, min_power / max_power)
    crosspol = min_power / max_power
    return min_angle, max_angle, crosspol


def get_final_co_cross(x2d, y2d, freq, phase_offset=0, add_fft_filter=True,
                       fft_filter_num_beamwidths=5):
    beam_co, beam_cross, phase_co, phase_cross = load_data(freq)
    beam_co = utils.get_complex_beam(beam_co, phase_co)
    beam_cross = utils.get_complex_beam(beam_cross, phase_cross)

    # fft filter the beams.
    if add_fft_filter:
        beam_co_filter = fft_filter_beam(
            x2d, y2d, beam_co, phase_co, fft_filter_num_beamwidths, freq)
        beam_cross_filter = fft_filter_beam(
            x2d, y2d, beam_cross, phase_cross, fft_filter_num_beamwidths, freq)
    else:
        beam_co_filter = beam_co
        beam_cross_filter = beam_cross

    # add the phase offset.
    if phase_offset != 0:
        beam_co = add_phase_offset(beam_co, phase_offset)
        beam_co_filter = add_phase_offset(beam_co_filter, phase_offset)

    min_angle, max_angle, crosspol = find_min_rotation_angle(
        x2d, y2d, beam_co_filter, beam_cross_filter)

    # rotate the beams by the min and max angles to get final co and crosspolar
    # beams!
    rotated_copol = rotate_beam(max_angle, beam_co, beam_cross)
    rotated_crosspol = rotate_beam(min_angle, beam_co, beam_cross)
    return rotated_copol, rotated_crosspol, min_angle, max_angle, crosspol


def get_final_beams(x2d, y2d, freq, phase_offset=0, add_fft_filter=True,
                    fft_filter_num_beamwidths=5, zero_pad_pix=100,
                    x_gradient=0):
    """x2d and y2d are in  units of [cm]"""
    rotated_copol, rotated_crosspol, min_angle, _, crosspol = \
        get_final_co_cross(x2d, y2d, freq, phase_offset=phase_offset,
                           add_fft_filter=add_fft_filter,
                           fft_filter_num_beamwidths=fft_filter_num_beamwidths)
    # subtract a phase gradient from the rotated copol!
    if x_gradient != 0:
        amp, phase = utils.complex_to_amp_phase(rotated_copol)
        phase *= 180 / np.pi
        phase -= x2d * x_gradient
        rotated_copol = utils.get_complex_beam(amp, phase)

    # zero pad the beam and FFT
    x_new, y_new, beam_complex = utils.zero_pad(x2d, y2d, rotated_copol,
                                                zero_pad_pix)
    beam_temp = np.fft.fftshift(beam_complex)
    beam_temp = np.fft.fft2(beam_temp)
    beam_ff = np.fft.fftshift(beam_temp)
    x_ang, y_ang = utils.coords_spat_to_ang(x_new / 1e2, y_new / 1e2, freq)

    return {'crosspol': crosspol, 'min_angle': min_angle,
            'copol_beam': rotated_copol, 'crosspool_beam': rotated_crosspol,
            'x_ang': x_ang, 'y_ang': y_ang, 'beam_ff': beam_ff}
