#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from . import utils


def plot_2d(x, y, beam_complex, z, suptitle='',
            norm_factor=None, plot_phase_dot=False, **colormesh_kwargs):
    # z = np.transpose(z)
    # beam_complex = np.transpose(beam_complex)

    db_signal = utils.to_db(beam_complex, norm_factor=norm_factor)
    # signal = (abs(beam_complex) / np.max(abs(beam_complex))) ** 2
    print(f"max power level={np.max(abs(beam_complex)):.3e}")

    plt.figure(figsize=(8, 3.5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(x, y, z)
    plt.colorbar(label='Phase')
    if plot_phase_dot:
        # plt.scatter([3.2], [-9.8])
        plt.scatter([0], [0])
        pass
    plt.title("Phase")
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    # plt.axis("equal")
    plt.subplot(1, 2, 2)
    if 'vmin' in colormesh_kwargs:
        plt.pcolormesh(x, y, db_signal, shading='auto', **colormesh_kwargs)
    else:
        plt.pcolormesh(x, y, db_signal, shading='auto', **colormesh_kwargs,
                       vmin=-50, vmax=0)
    plt.title("Power [dB]")
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    # plt.axis("equal")
    plt.colorbar(label='Power')
    plt.suptitle(suptitle)
    plt.tight_layout()
    return


def plot_beam(x2d, y2d, beam, title='', convert_to_db=True,
              norm_factor=None, plot_colorbar=True, unit='cm',
              **colormesh_kwargs):
    if convert_to_db:
        beam = utils.to_db(beam, norm_factor=norm_factor)
        if 'vmin' in colormesh_kwargs:
            cols = plt.pcolormesh(
                x2d, y2d, beam, shading='auto', **colormesh_kwargs)
        else:
            cols = plt.pcolormesh(x2d, y2d, beam, shading='auto', **colormesh_kwargs,
                                  vmin=-50)
    else:
        cols = plt.pcolormesh(x2d, y2d, beam, shading='auto',
                              **colormesh_kwargs)
    plt.title(title)
    if unit is not None:
        plt.xlabel(f'[{unit}]')
        plt.ylabel(f'[{unit}]')
    if convert_to_db and plot_colorbar:
        plt.colorbar(label="dB")
    elif plot_colorbar:
        plt.colorbar()
    return cols


def plot_farfield(x, y, beam_ff, ax=None, plot_colorbar=True,
                  clabel='[dB]', unit='arcmin', title='',
                  **colormesh_kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    cols = ax.pcolormesh(x, y, utils.to_db(abs(beam_ff)),
                         shading='auto', **colormesh_kwargs)
    if plot_colorbar:
        plt.colorbar(cols, ax=ax, label=clabel)
    if unit is not None:
        ax.set_xlabel(f'[{unit}]')
        ax.set_ylabel(f'[{unit}]')
    ax.set_title(title)
    return cols


def plot_data_cuts(cut_data, log=True):
    plt.plot(cut_data['cut_x_vals'] - cut_data['center'][0], cut_data['cut_x'],
             label=f'x: {cut_data["x_fwhm"]:.3f}')
    plt.plot(cut_data['cut_y_vals'] - cut_data['center'][1], cut_data['cut_y'],
             label=f'y: {cut_data["y_fwhm"]: .3f}')
    plt.plot(cut_data['cut_45_vals'][2], cut_data['cut_45'],
             label=f'45: {cut_data["45_fwhm"]: .3f}')
    plt.plot(cut_data['cut_135_vals'][2], cut_data['cut_135'],
             label=f'135: {cut_data["135_fwhm"]: .3f}')
    plt.plot(cut_data['radii'], cut_data['means'],
             label=f'rad: {cut_data["rad_fwhm"]: .3f}', color='black', lw=2)
    plt.plot(-cut_data['radii'], cut_data['means'], color='black', lw=2)
    plt.legend(title="1d cut FWHM")
    if log:
        plt.yscale('log')
    plt.xlabel('far-field offset [arcmin]')
    plt.ylabel('beam taper on cut line')
    return


def plot_beam_stamp(cut_data, **plot_kwargs):
    # power is already squared, so take sqrt to convert to amplitude
    return plot_beam(cut_data['x'], cut_data['y'], np.sqrt(cut_data['power']),
                     convert_to_db=True, **plot_kwargs)
