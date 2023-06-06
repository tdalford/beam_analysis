#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import utils
import plot_beams


def beam_convolve_forward(x, y, beam, apert1, apert2, plot=False):
    #     Define g
    if apert1 < apert2:
        disc = np.cos(np.pi*y/apert2)
        disc = np.where(abs(y) <= apert2/2, disc, 0)
        disc = np.where(abs(x) <= apert1/2, disc, 0)

    else:
        disc = np.cos(np.pi*x/apert1)
        disc = np.where(abs(x) <= apert1/2, disc, 0)
        disc = np.where(abs(y) <= apert2/2, disc, 0)

    if plot:
        plt.pcolormesh(np.real(x), np.real(y), abs(disc))
        plt.colorbar()
        plt.axis([-apert1 * 2, apert2 * 2, -apert2 * 2, apert2 * 2])
        plt.show()

    tmp = np.fft.fftshift(disc)
    tmp = np.fft.fft2(tmp)
    # Matrix G
    disc_fft = np.fft.fftshift(tmp)

    tmp = np.fft.fftshift(beam)
    tmp = np.fft.fft2(tmp)
    beam_fft = np.fft.fftshift(tmp)

    beam_conv = beam_fft*disc_fft

    tmp = np.fft.ifftshift(beam_conv)
    tmp = np.fft.ifft2(tmp)
    beam_final = np.fft.ifftshift(tmp)
    return beam_final


def disc_convolve(x, y, beam_complex, aperture_size):
    disc = np.ones(x.shape)
    disc = np.where(np.sqrt(y ** 2 + x ** 2) <= (aperture_size / 2), disc, 0)
    disc = np.where(disc == 0, disc, 1)

    tmp = np.fft.fftshift(disc)
    tmp = np.fft.fft2(tmp)
    # Matrix G
    disc_fft = np.fft.fftshift(tmp)

    tmp = np.fft.fftshift(beam_complex ** 2)
    tmp = np.fft.fft2(tmp)
    beam_fft = np.fft.fftshift(tmp)

    beam_conv = beam_fft * disc_fft

    tmp = np.fft.ifftshift(beam_conv)
    tmp = np.fft.ifft2(tmp)
    beam_final = np.fft.ifftshift(tmp)
    return abs(beam_final)


def get_centered_square(array, square_size):
    rows, cols = array.shape
    start_row = (rows - square_size) // 2
    start_col = (cols - square_size) // 2
    return start_row, start_row + square_size, start_col, \
        start_col + square_size


def pad_data(
        large_data_x, large_data_y, large_data_amp, large_data_phase,
        small_data_x, small_data_y, small_data_amp, small_data_phase):
    padded_amp = np.copy(large_data_amp)
    # padded_amp = padded_amp * 0 + \
    #     np.random.normal(0, 8e-3, size=padded_amp.shape)
    padded_phase = np.copy(large_data_phase)
    start_row, end_row, start_col, end_col = get_centered_square(
        padded_amp, small_data_x.shape[0])
    padded_amp[start_row:end_row, start_col:end_col] = small_data_amp
    padded_phase[start_row:end_row, start_col:end_col] = small_data_phase
    return large_data_x, large_data_y, padded_amp, padded_phase


def center_around_beam(x, y, beam, phase, beam_center, square_length=None):
    '''Center the beam around beam_center with a square of square_length.

    If square_length is not specified then use the largest possible square.'''
    # center the beam.
    # to do this we have to make sure that the decimals in our center match up
    # to the decimals in our data resolution!
    # e.g. if our resolution is .2GHz then our center has to be to at most
    # 0.2GHz tolerance.
    mask = np.where((abs(x - beam_center[0]) < square_length) & (
        abs(y - beam_center[1]) < square_length))
    x_centered = x[mask]
    xlen = int(np.sqrt(x_centered.shape))
    # subtract the center so that our x and y coordinates are centered around
    # zero.
    # Now we can subtract the actual center value without rounding!
    return x_centered.reshape(xlen, xlen) - beam_center[0], \
        y[mask].reshape((xlen, xlen)) - beam_center[1], \
        beam[mask].reshape((xlen, xlen)), phase[mask].reshape((xlen, xlen))


def find_center_and_take_stamp(
        x, y, beam, phase, square_length=None, center_guess=(0, 0),
        size_limits=[[-25, 25], [-25, 25]], estimate_center=True):
    '''Estimate the beam center and take return a smaller stamped beam.'''
    if estimate_center:
        center_estimate = estimate_center(
            x, y, beam, center_guess, size_limits)
    else:
        center_estimate = center_guess
    return center_around_beam(x, y, beam, phase, center_estimate,
                              square_length=square_length)


def center_and_save_beam(x, y, beam, phase, fname, plot=False,
                         **center_and_take_stamp_kwargs):
    '''Center the beam and save the final arrays to fname.'''
    x_centered, y_centered, beam_centered, phase_centered = \
        find_center_and_take_stamp(
            x, y, beam, phase, **center_and_take_stamp_kwargs)
    # convert phase from radians to degrees
    if np.max(phase) > 2 * np.pi:
        print("converting from degrees to radians...")
        phase_centered = np.deg2rad(phase_centered)
    if plot:
        plot_beams.plot_2d(
            x_centered, y_centered, beam_centered, phase_centered,
            plot_phase_dot=True)

    save_arr = np.array(
        [x_centered, y_centered, beam_centered, phase_centered])
    np.save(fname, save_arr)
    return


def roll_and_save_beam(x, y, beam, phase, roll_factors, fname, plot=False):
    '''Roll the beam to center it and save final arrays to fname.'''
    beam_rolled = utils.roll_beam(beam, *roll_factors)
    phase_rolled = utils.roll_beam(phase, *roll_factors)
    # I think we want this in degrees
    # if np.max(phase) > 2 * np.pi:
    #     print("converting from degrees to radians...")
    #     phase = np.deg2rad(phase)

    if plot:
        plot_beams.plot_2d(
            x, y, beam_rolled, phase_rolled, plot_phase_dot=True)
    save_arr = np.array([x, y, beam_rolled, phase_rolled])
    np.save(fname, save_arr)
