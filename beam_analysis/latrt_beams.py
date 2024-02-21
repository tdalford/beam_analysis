#!/usr/bin/env python3
import numpy as np
from . import nearfield_beams
from . import utils


def add_scatter_and_convolve(x_sim, y_sim, amp_sim, phase_sim,
                             apert_sizes=[0.5, 0.5]):
    """Phase is in RADIANS"""
    assert np.max(phase_sim) <= np.pi
    val_mf2 = 5
    scatter_term = np.where((x_sim <= 21) & (x_sim > -20) & ((y_sim-(
        x_sim/1.5)) > -15) & ((y_sim+(x_sim/1.5)) > -15) & ((-y_sim-(
            x_sim/1.5)) > -35) & ((-y_sim+(x_sim/1.5)) > -35), -(
                x_sim**2 / 35)-35, -50)

    phase_mod = (-3.5*(y_sim+20)**2) + (2*(x_sim)**2)
    scatter_lin = 10**(scatter_term/20)
    phase_term = np.where((x_sim <= 21) & (x_sim > -20) & ((y_sim-(
        x_sim/1.5)) > -15) & ((y_sim+(x_sim/1.5)) > -15) & ((-y_sim-(
            x_sim/1.5)) > -35) & ((-y_sim+(x_sim/1.5)) > -35), phase_mod, 0)
    p_sim_temp = phase_term
    phase_term = np.mod(p_sim_temp*np.pi/180, 2*np.pi)
    scatter_beam = scatter_lin*np.exp(complex(0, 1)*phase_term)

    amp_sim = np.where((x_sim <= 21) & (x_sim > -20) & ((y_sim-(
        x_sim/1.5)) > -15) & ((y_sim+(x_sim/1.5)) > -15) & ((-y_sim-(
            x_sim/1.5)) > -35) & ((-y_sim+(x_sim/1.5)) > -35),
        (amp_sim/np.max(amp_sim)), 1e-4)
    b_sim = (amp_sim/np.max(amp_sim)) * \
        np.exp(complex(0, 1)*np.mod(phase_sim, 2*np.pi))
    bb_sim = b_sim + scatter_beam
    # bb_sim = (amp_sim / np.max(amp_sim))
    out = nearfield_beams.beam_convolve_forward(x_sim, y_sim, bb_sim, apert1=(
        apert_sizes[0]), apert2=(apert_sizes[1]))
    return out

def add_scatter_and_convolve_LF(x_sim, y_sim, amp_sim, phase_sim,
                                apert_sizes=[0.5, 0.5]):
    """Phase is in RADIANS"""
    assert np.max(phase_sim) <= np.pi
    val_mf2 = 5
    scatter_term = np.where((x_sim <= 21) & (x_sim > -20) & ((y_sim-(
        x_sim/1.5)) > -15) & ((y_sim+(x_sim/1.5)) > -15) & ((-y_sim-(
            x_sim/1.5)) > -35) & ((-y_sim+(x_sim/1.5)) > -35), -(
                x_sim**2 / 35)-35, -50)

    phase_mod = (-3.5*(y_sim+20)**2) + (2*(x_sim)**2)
    scatter_lin = 10**(scatter_term/20)
    phase_term = np.where((x_sim <= 21) & (x_sim > -20) & ((y_sim-(
        x_sim/1.5)) > -15) & ((y_sim+(x_sim/1.5)) > -15) & ((-y_sim-(
            x_sim/1.5)) > -35) & ((-y_sim+(x_sim/1.5)) > -35), phase_mod, 0)
    p_sim_temp = phase_term
    phase_term = np.mod(p_sim_temp*np.pi/180, 2*np.pi)
    scatter_beam = scatter_lin*np.exp(complex(0, 1)*phase_term)

    amp_sim = np.where((x_sim <= 19) & (x_sim > -22) & ((y_sim-(
        x_sim/1.4)) > -30) & ((y_sim + (x_sim/1.4)) > -30) & ((-y_sim-(
            x_sim/1.8) + 0) > -20) & ((-y_sim+(x_sim/1.8) + 3) > -20),
        (utils.normalize(amp_sim)), 1e-4)
    b_sim = (amp_sim/np.max(amp_sim)) * \
        np.exp(complex(0, 1)*np.mod(phase_sim, 2*np.pi))
    bb_sim = b_sim + scatter_beam
    out = nearfield_beams.beam_convolve_forward(x_sim, y_sim, bb_sim, apert1=(
        apert_sizes[0]), apert2=(apert_sizes[1]))
    # out = bb_sim
    return out


def running_mean(x, N):
    """ Calculates running mean of 1D array.
    """
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def get_beam_data_2d(txt_file):
    DATA_1 = np.loadtxt(txt_file, skiprows=1)
    DATA = []
    L_MEAN = 1
    N_INDIV = 7

    line_size = np.size(DATA_1[0])
    nsamp = np.size(DATA_1, 0)
    arr_x = np.zeros(nsamp)
    arr_y = np.zeros(nsamp)
    arr_phi = np.zeros(nsamp)
    amp_cross = np.zeros(nsamp)
    amp_AA = np.zeros(nsamp)
    amp_BB = np.zeros(nsamp)
    amp_var = np.zeros(nsamp)
    phase = np.zeros(nsamp)

    i_AA_begin = int(N_INDIV + (1-1)*(line_size-N_INDIV)/4)
    i_AA_end = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4) - 1
    i_BB_begin = int(N_INDIV + (2-1)*(line_size-N_INDIV)/4)
    i_BB_end = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4) - 1
    i_AB_begin = int(N_INDIV + (3-1)*(line_size-N_INDIV)/4)
    i_AB_end = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4) - 1
    i_phase_begin = int(N_INDIV + (4-1)*(line_size-N_INDIV)/4)
    i_phase_end = int(N_INDIV + (5-1)*(line_size-N_INDIV)/4) - 1

    i = int(0)

    jj = 1
    while (jj <= 1):
        i = int(0)
        if jj == 1:
            str_data = 'Dataset 1'
            DATA = DATA_1
        else:
            DATA = DATA_2
            str_data = 'Dataset 2'
        while (i < (nsamp)):
            # take in raw DATA
            arr_x[i] = DATA[i][1]
            arr_y[i] = DATA[i][2]
            arr_phi[i] = DATA[i][3]
            # use same index singal for both datasets. keep it simple for now.
            index_signal = DATA[i][4]
            arr_AA = np.array(running_mean(
                DATA[i][i_AA_begin: i_AA_end], L_MEAN))
            arr_BB = np.array(running_mean(
                DATA[i][i_BB_begin: i_BB_end], L_MEAN))
            arr_AB = np.array(running_mean(
                DATA[i][i_AB_begin: i_AB_end], L_MEAN))
            arr_phase = np.array(DATA[i][i_phase_begin: i_phase_end])
            n_channels = np.size(arr_AB)

            # make amplitude arrays, in case they need to be plotted.
            amp_cross[i] = np.power(arr_AB[int(n_channels/2)], 1)
            amp_var[i] = np.power(np.divide(arr_AB[int(n_channels/2)], np.sqrt(
                arr_AA[int(n_channels/2)])), 1)
            amp_AA[i] = arr_AA[int(n_channels/2)]
            amp_BB[i] = arr_BB[int(n_channels/2)]
            phase[i] = np.remainder(arr_phase[int(n_channels/2)], 360.)
            #print('phase[i] = '+str(phase[i]))
            i = i + 1

        arr_x = np.unique(arr_x)
        arr_y = np.unique(arr_y)
        X, Y = np.meshgrid(arr_x, arr_y)

        P_source = amp_AA
        # P_cross = amp_cross
        source = P_source / np.max(P_source)
        source[np.where(source < 0.1)] = np.mean(
            source[np.where(source > 0.1)])
        # # divide out source amplitude.
        # P = np.sqrt(P_cross ** 2 / source ** 2).reshape(len(arr_x), len(
        #     arr_y))

        P = amp_cross.reshape(len(arr_x), len(arr_y))
        # P = amp_AA.reshape(len(arr_x), len(arr_y))
        # P = (amp_BB).reshape(len(arr_x), len(arr_y))
        # P = amp_var.reshape(len(arr_x), len(arr_y))
        Z = phase.reshape(len(arr_x), len(arr_y))

        jj = jj + 1

    beam_complex = P * np.exp(Z * np.pi / 180. * complex(0, 1))
    return X, Y, Z, beam_complex
