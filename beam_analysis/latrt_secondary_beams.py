#!/usr/bin/env python3
import numpy as np
from . import utils


def get_mid_region(x, y, beam):
    mid = np.where(
        (((abs(y) - 0.35) <= 1.4) | ((abs(x) - 0.35) <= 3 * 0.7)),
        abs(beam) / np.max(abs(beam)), 0)
    mid = np.where(
        (((abs(x) - 0.35) <= 1.4) | ((abs(y) - 0.35) <= 3 * 0.7)), mid, 0)
    mid = np.where((abs(y) < (9 * 0.7 / 2)) & (abs(x) < (9 * 0.7 / 2)), mid, 0)
    return mid


def calculate_spill(x, y, secondary_beam):
    beamfft_mid = get_mid_region(x, y, secondary_beam)
    spill = 1 - (np.sum(beamfft_mid ** 2) / np.sum((abs(
        secondary_beam) / np.max(abs(secondary_beam))) ** 2))
    return spill


def get_secondary_beam(x, y, amp, phase, freq, center_amp=False):
    if center_amp:
        amp_centered = amp - np.mean(amp)
        amp = amp_centered
    beamfft, phifft = b2a(abs(amp), phase)
    ang_x, ang_y = utils.coords_spat_to_ang(x / 1e2, y / 1e2, freq)

    # 12m away from measurement plane TODO: fix this magic number
    # d_source = 101 + (6.5 * 2.54) # I think it was this earlier
    d_source = 124.1 + (6.5 * 2.54) # window is ~1241.5mm from the focal plane
    x = (12 - (d_source / 1e2)) * np.tan(ang_x)

    # 12m away from measurement plane
    y = (12 - (d_source / 1e2)) * np.tan(ang_y)
    return x, y, beamfft, phifft


def b2a(beam, phase):
    """
    FFT angular space to aperture plane.

    phase is in DEGREES.
    """
    assert np.max(phase) > 2 * np.pi
    beam_complex = (abs(beam)) * np.exp(phase *
                                        np.pi / 180.0 * np.complex(0, 1))
    # beam_complex = (abs(beam)) * np.exp(phase * complex(0, 1))
    # tmp = np.fft.fftshift(beam_complex)
    tmp = np.fft.ifft2(beam_complex)
    aper_field = np.fft.fftshift(tmp)
    aper_phase = np.arctan2(np.imag(aper_field), np.real(aper_field))
    return aper_field, aper_phase
