"""
Adapted from Nynke Nell MSc Thesis

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from glob import glob
import matplotlib.cm as cm
import matplotlib
import pickle
import gsw
matplotlib.use('Qt5Agg')


def main():
    # process_data()
    plot_results(fresh_water=True)


def process_data(pick_start_end=False):
    data_folder = '/Users/simenbootsma/Documents/PhD/Work/SedimentIce/ForceMeasurements/'
    overview = pd.read_csv('/Users/simenbootsma/Documents/PhD/Work/SedimentIce/Experiments_overview.csv', delimiter=';')
    files = sorted(glob(data_folder + 'fm*'))

    data = []
    if pick_start_end:
        start_end = {}
    else:
        with open('data/start_end.pkl', 'rb') as f:
            start_end = pickle.load(f)
    for fn in files:
        name = fn[len(data_folder):]
        _, dt, n = name.split('_')
        dt = "/".join(dt.split('-'))
        params = overview[(overview["Date"]==dt) & (overview["Measurement"]==int(n))]

        if len(params) == 0:
            print("could not find file '{:s}' in overview...".format(name))
            continue

        df = pd.read_csv(fn, delimiter=';', skiprows=2, names=['time', 'force', 'daytime'])
        t, F = df['time'], df['force']

        if pick_start_end:
            plt.plot(t, F)
            plt.title(name)
            points = plt.ginput(2, mouse_add=plt.MouseButton.RIGHT, mouse_pop=None)
            start_end[name] = (points[0][0], points[1][0])
            plt.close()

        F = F[(t > start_end[name][0]) & (t < start_end[name][1])]
        t = t[(t > start_end[name][0]) & (t < start_end[name][1])]
        F -= np.mean(F[-10:])

        T = float(params['Temperature'].values[0].replace(',', '.'))
        S = float(params['Salinity'].values[0].replace(',', '.'))
        is_clear_ice = params['Particle size [micron]'].values[0] == 'clear ice'
        is_stainless_steel = params['Particle size [micron]'].values[0] == '1000RVS'

        rho_f = gsw.rho_t_exact(S, T, 0)
        phi_s = 0 if is_clear_ice else 0.62  # NOTE: assumed packing fraction, check measured values
        rho_ice = 916  # kg/m3
        rho_sed = 7800 if is_stainless_steel else 2500  # kg/m3, TODO: check sediment densities with manufacturer
        rho_block = rho_ice * (1 - phi_s) + rho_sed * phi_s
        g = 9.81
        R0 = 0.05  # initial cylinder radius [m]
        V = F.values / (g * (rho_block - rho_f))

        # Compute dV/dt
        # Apply linear regression between 30% and 40% volume of block left
        V0 = V[0]
        # V0 = float(params['Mass block'].values[0].replace(',', '.')) * 1e-3 / rho_block
        I = (V/V0 > 0.3) & (V/V0 < 0.4)
        p1 = np.polyfit(t[I], V[I], deg=1)
        dVdt = p1[0]  # m3/s
        p2 = np.polyfit(t[I], np.sqrt(V[I]/V0), deg=1)
        dRdt = p2[0] * R0  # m/s

        # Save to Dataframe
        p_size = params['Particle size [micron]'].values[0]
        material = 'none' if is_clear_ice else 'stainless steel' if is_stainless_steel else 'glass'
        if is_clear_ice:
            dp = 0
            dp_err = 0
        elif is_stainless_steel:
            dp = 1000
            dp_err = 0
        elif '-' in p_size:
            min_dp, max_dp = p_size.split('-')
            dp = (int(min_dp) + int(max_dp)) // 2
            dp_err = (int(max_dp) - int(min_dp)) // 2
        else:
            dp = int(p_size)
            dp_err = 0

        data.append({'dp': dp, 'dp_err': dp_err, 'dvdt': dVdt, 'drdt': dRdt, 'temperature': T, 'salinity': S, 'material': material})
        if dp == 1000 and material == 'glass' and S == 0:
            print(name)

        # 18/11/2024, 40-70, fresh
        # 18/11/2024, 70-110, fresh

        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(t, V)
        # ax[0].plot(t[I], p1[0]*t[I] + p1[1], '-r')
        # ax[1].plot(t, np.sqrt(V/V[0]))
        # ax[1].plot(t[I], p2[0]*t[I] + p2[1], '-r')
        # plt.title(name)
        # plt.show()

    pd.DataFrame(data).to_csv('data/ablation_experiments.csv')

    if pick_start_end:
        with open('data/start_end.pkl', 'wb') as f:
            pickle.dump(start_end, f)


def plot_results(fresh_water=True):
    data = pd.read_csv('data/ablation_experiments.csv')

    if fresh_water:
        drdt_clear = data[(data['material'] == 'none') & (data['salinity'] == 0)]['drdt']
        data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    else:
        drdt_clear = data[(data['material'] == 'none') & (data['salinity'] > 0)]['drdt']
        data = data[(data['material'] == 'glass') & (data['salinity'] > 0)]
    print(drdt_clear.values)

    mean_marker = {'fmt': 'o', 'color': '#0398fc', 'mec': 'k', 'markersize': 10, 'label': 'experiments mean', 'capsize': 5, 'capthick': 1}
    single_marker = {'marker': 'o', 'color': '#025F9D', 'markersize': 10, 'alpha': 0.4, 'linestyle': '', 'label': 'experiments'}
    gprime = 9.81 * (2500-1000)/2500
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = gprime * (dp*1e-6)**3 / nu**2
    Ga_err = np.abs(gprime/nu**2 * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3 - data_means['Ga'].values)
    data['Ga'] = gprime * (data['dp']*1e-6)**3 / nu**2

    fig, ax = plt.subplots(num=2, figsize=(8, 8))
    ax.set_xscale("log")
    ax.plot(data['Ga'], -data['dvdt'] * 1e6, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['dvdt'] * 1e6, xerr=Ga_err, **mean_marker)
    plt.xlabel('Ga', fontsize=20)
    plt.ylabel('$\dot{V}$ [cm$^3$/s]', fontsize=20)
    plt.ylim(top=5, bottom=0)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    # plt.xlim([1e8, None])

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    fname = 'all_drdt.npz' if fresh_water else 'all_drdt_salt.npz'
    model = np.load(fname)
    min_drdt = np.nanmin(model['drdt'], axis=0)
    max_drdt = np.max(model['drdt'], axis=0)
    mean_drdt = np.nanmean(model['drdt'], axis=0)

    # for i in range(len(model['drdt'])):
    #     plt.plot(model['dp'], -model['drdt'][i, :], '-', color='0.7', lw=1, alpha=0.5)
    # plt.plot(model['dp'], -model['drdt'][0, :], '-r', lw=2)
    # plt.plot(model['dp'], -min_drdt, '-', color='0.4', lw=1)
    # plt.plot(model['dp'], -max_drdt, '-', color='0.4', lw=1)

    model_Ga = gprime * model['dp'] ** 3 / nu**2
    for i in range(len(model['drdt'])):
        plt.plot(model_Ga, -model['drdt'][i, :] * 1e3, '-', color='0.7', lw=1, alpha=0.5)
    plt.plot([], [], '-', color='0.7', lw=1, label='model spread')
    plt.plot(model_Ga, -mean_drdt * 1e3, '-k', lw=2, label='model mean')
    plt.plot(model_Ga, -min_drdt * 1e3, '-', color='0.4', lw=1)
    plt.plot(model_Ga, -max_drdt * 1e3, '-', color='0.4', lw=1)

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-2, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)
    plt.xlabel('Ga', fontsize=16)
    plt.ylabel('$\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([1.5e-2, 1.5e8])
    plt.ylim([0, None])

    plt.show()


if __name__ == "__main__":
    main()

