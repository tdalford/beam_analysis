#!/usr/bin/env python3
import numpy as np
import nearfield_beams


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
            amp_var[i] = np.power(
                np.divide(arr_AB[int(n_channels/2)], arr_AA[int(n_channels/2)]), 1)
            amp_AA[i] = arr_AA[int(n_channels/2)]
            amp_BB[i] = arr_BB[int(n_channels/2)]
            phase[i] = np.remainder(arr_phase[int(n_channels/2)], 360.)
            #print('phase[i] = '+str(phase[i]))
            i = i + 1

        arr_x = np.unique(arr_x)
        arr_y = np.unique(arr_y)
        X, Y = np.meshgrid(arr_x, arr_y)

        # P_source = amp_AA
        # P_cross = amp_cross
        # source = P_source / np.max(P_source)
        # source[np.where(source < 0.1)] = np.mean(
        #     source[np.where(source > 0.1)])
        # divide out source amplitude.
        # beam = np.sqrt(P_cross ** 2 / source ** 2)

        P = amp_cross.reshape(len(arr_x), len(arr_y))
        Z = phase.reshape(len(arr_x), len(arr_y))

        jj = jj + 1

    beam_complex = P * np.exp(Z * np.pi / 180. * complex(0, 1))
    #Z = np.transpose(Z)
    #beam_complex = np.transpose(beam_complex)
    return X, Y, Z, beam_complex


def get_data_str(freq, field_type='E', angle='180', dims='1D', date='30-8-2022',
                 addendum_str='', has_crosspol=True, make_freq_float=False):
    """Get the the datafile for a recent holography measurement."""
    assert field_type in ['E', 'H', '']
    assert dims in ['1D', '2D']
    if dims == '1D':
        # specify E vs H field
        dim_str = dims + '_' + field_type
    else:
        # no specification
        dim_str = dims

    # if specified, write the freq as a float
    if make_freq_float:
        freq = float(freq)

    if has_crosspol:
        crosspol_str = '_crosspol360'
    else:
        crosspol_str = ''
    data_str = "%sGHz_%sdeg_%s_%s_%s%s.txt" % (
        freq, angle, dim_str, date, addendum_str,
        crosspol_str)
    return data_str


def get_all_close_NDF_params():
    mf1_freqs = [80, 85]
    mf1_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
        'NDF_close':
        {'date': '9-9-2022', 'addendum_str': 'NDF_close_mount'},
    }

    mf1_high_freqs = [90, 95, 100, 105, 110]
    mf1_higher_params = {
        'mount_only':
        {'date': '9-9-2022', 'addendum_str': 'close_mount'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
        'NDF_close':
        {'date': '9-9-2022', 'addendum_str': 'NDF_close_mount'},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in mf1_freqs:
        total_param_dict[freq] = mf1_params

    for freq in mf1_high_freqs:
        total_param_dict[freq] = mf1_higher_params
    return total_param_dict


def get_LPE_NDF_params():
    mf1_freqs = [80, 85, 90, 95, 100, 105, 110]
    mf1_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF_LPE':
        {'date': '15-9-2022', 'addendum_str': 'NDF_LPE'},
    }

    mf2_freqs = [130, 135, 140, 145, 150, 155]
    mf2_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF_LPE':
        {'date': '17-9-2022', 'addendum_str': 'NDF_LPE'},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in mf1_freqs:
        total_param_dict[freq] = mf1_params

    for freq in mf2_freqs:
        total_param_dict[freq] = mf2_params
    return total_param_dict


def get_comparison_NDF_params():
    mf1_freqs = [80, 85]
    mf1_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
        'NDF_new':
        {'date': '9-9-2022', 'addendum_str': 'NDF'},
    }

    mf1_high_freqs = [90, 95, 100, 105, 110]
    mf1_higher_params = {
        'mount_only':
        {'date': '9-9-2022', 'addendum_str': 'close_mount'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
        'NDF_new':
        {'date': '9-9-2022', 'addendum_str': 'NDF'},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in mf1_freqs:
        total_param_dict[freq] = mf1_params

    for freq in mf1_high_freqs:
        total_param_dict[freq] = mf1_higher_params
    return total_param_dict


def get_misaligned_NDF_params():
    mf1_freqs = [95, 100]
    mf1_params = {
        'no_NDF_misaligned':
        {'date': '12-9-2022', 'addendum_str': 'no_NDF_misaligned'},
        'NDF_misaligned':
        {'date': '12-9-2022', 'addendum_str': 'NDF_misaligned'},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in mf1_freqs:
        total_param_dict[freq] = mf1_params
    return total_param_dict


def get_original_crosspol_params():
    low_power_mf1_freqs = [80, 85, 90, 95, 100, 105, 110]
    low_power_mf1_params = {
        'no NDF':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
    }

    # eventually change to the 2.5GHz increments and take out the first iter
    old_mf2_freqs = [130, 135, 140, 145, 150, 155]
    mf2_params = {
        'no NDF':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '31-8-2022', 'addendum_str': 'NDF'},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in low_power_mf1_freqs:
        total_param_dict[freq] = low_power_mf1_params

    for freq in old_mf2_freqs:
        total_param_dict[freq] = mf2_params
    return total_param_dict


# TODO: Change 80GHz measurement to use twist instaed of no twist data
# TODO: add in twist data to be a possible systematic errorbar?
def get_all_crosspl_params():
    low_power_mf1_freqs = [80, 85, 90, 95, 100, 105, 110]
    low_power_mf1_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power'},
        'mount_only_twist':
        {'date': '30-8-2022', 'addendum_str': 'mount_only_twist'},
        'NDF_twist':
        {'date': '30-8-2022', 'addendum_str': 'NDF_low_power_twist'},
    }

    # eventually change to the 2.5GHz increments and take out the first iter
    old_mf2_freqs = [130, 135, 140, 145, 150, 155]
    mf2_params = {
        'mount_only':
        {'date': '30-8-2022', 'addendum_str': 'mount_only'},
        'NDF':
        {'date': '31-8-2022', 'addendum_str': 'NDF'},
        'NDF_2':
        {'date': '1-9-2022', 'addendum_str': 'NDF_low_power',
         'make_freq_float': True},
        'mount_only_twist':
        {'date': '30-8-2022', 'addendum_str': 'mount_only_twist'},
        'NDF_twist':
        {'date': '31-8-2022', 'addendum_str': 'NDF_twist'},
        'NDF_twist_2':
        {'date': '1-9-2022', 'addendum_str': 'NDF_low_power_twist',
         'make_freq_float': True},
    }

    total_param_dict = {}
    # make global params a key in this dict
    total_param_dict['global'] = {'has_crosspol': True}
    # go through the args of the different measurements
    for freq in low_power_mf1_freqs:
        total_param_dict[freq] = low_power_mf1_params

    # Add in the 2.5GHz increments
    mf2_freqs = [130, 132.5, 135, 137.5, 140, 142.5, 145, 147.5, 150, 152.5,
                 155, 157.5]
    for freq in mf2_freqs:
        if freq in old_mf2_freqs:
            total_param_dict[freq] = dict(mf2_params)
        else:
            # continue
            total_param_dict[freq] = {
                'NDF_2':
                {'date': '1-9-2022', 'addendum_str': 'NDF_low_power',
                 'make_freq_float': True},
                'NDF_twist_2':
                {'date': '1-9-2022', 'addendum_str': 'NDF_low_power_twist',
                 'make_freq_float': True},
            }

        # Add set of 5 runs we've taken for NDF, mount only, mount only twist
        for i in range(5):
            total_param_dict[freq].update({
                'NDF_%s' % (i + 3):
                {'date': '2-9-2022', 'addendum_str': 'NDF_low_power_%s' % i,
                 'make_freq_float': True}})
            # don't include lower than ~135 GHz for now?
            # might as well calculate and then take out later
            if freq > 135:  # 135
                total_param_dict[freq].update({
                    'mount_only_%s' % (i + 2):
                    {'date': '4-9-2022', 'addendum_str': 'mount_only_%s' % i,
                     'make_freq_float': True},
                    'mount_only_twist_%s' % (i + 2):
                    {'date': '5-9-2022', 'addendum_str':
                     'mount_only_twist_%s' % i, 'make_freq_float': True},
                })

        if freq == 145:
            for i in range(10):
                total_param_dict[freq].update(
                    {"NDF_twist_%s" % (i + 3):
                     {"date": "1-9-2022",
                      "addendum_str": "NDF_low_power_twist_%s" % i},
                     })

    return total_param_dict
