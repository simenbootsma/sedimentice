import numpy as np
import matplotlib.pyplot as plt
from PlumeModel import PlumeModel, ConvergenceError
import matplotlib
import pandas as pd
import pickle
from ablation_rate_experiments import data_Ward2024
import time
import cmcrameri.cm as cmc
import cmocean.cm as cmo

matplotlib.use('Qt5Agg')

BEST_FIT_PARAMETERS = {
    "T_inf": 20,  # ambient temperature (degrees C)
    "S_inf": 0,  # ambient salinity (g/kg)
    "b0": 5.38e-5,  # plume breadth at the top (m)
    "w0": 4.72e-2,  # plume velocity at the top (m/s)
    "phi0": 0.0,  # plume particle volume fraction at the top (-)
    "alpha": 7.73e-4,  # entrainment coefficient (-)
    "w_a": 0,  # ambient water velocity (m/s)
    "dRdt": -1e-5,  # ablation rate (m/s)
    "d_p": 1e-3,  # particle diameter (m)
    "z0": 1e-6,  # location of first grid point (m)
    "max_z": 0.1,  # cylinder height (m)
    "n_points": 1000  # number of grid points
}


BEST_FIT_PARAMETERS = {
    "T_inf": 20,  # ambient temperature (degrees C)
    "S_inf": 0,  # ambient salinity (g/kg)
    "b0": 1.67e-2,  # plume breadth at the top (m)
    "w0": 3.59e-5,  # plume velocity at the top (m/s)
    "phi0": 0.0,  # plume particle volume fraction at the top (-)
    "alpha": 4.64e-4,  # entrainment coefficient (-)
    "w_a": 0,  # ambient water velocity (m/s)
    "dRdt": -1e-5,  # ablation rate (m/s)
    "d_p": 1e-3,  # particle diameter (m)
    "z0": 1e-6,  # location of first grid point (m)
    "max_z": 0.1,  # cylinder height (m)
    "n_points": 1000  # number of grid points
}

BEST_FIT_SMALL_PARAMETERS = {
    "T_inf": 20,  # ambient temperature (degrees C)
    "S_inf": 0,  # ambient salinity (g/kg)
    "b0": 2.78e-3,  # plume breadth at the top (m)
    "w0": 2.15e-2,  # plume velocity at the top (m/s)
    "phi0": 0.6,  # plume particle volume fraction at the top (-)
    "alpha": 7.74e-2,  # entrainment coefficient (-)
    "w_a": 0,  # ambient water velocity (m/s)
    "dRdt": -1e-5,  # ablation rate (m/s)
    "d_p": 1e-3,  # particle diameter (m)
    "z0": 1e-6,  # location of first grid point (m)
    "max_z": 0.1,  # cylinder height (m)
    "n_points": 1000  # number of grid points
}


BEST_FIT_LARGE_PARAMETERS = {
    "T_inf": 20,  # ambient temperature (degrees C)
    "S_inf": 0,  # ambient salinity (g/kg)
    "b0": 4.64e-4,  # plume breadth at the top (m)
    "w0": 2.15e-2,  # plume velocity at the top (m/s)
    "phi0": 0.0,  # plume particle volume fraction at the top (-)
    "alpha": 1e-5,  # entrainment coefficient (-)
    "w_a": 0,  # ambient water velocity (m/s)
    "dRdt": -1e-5,  # ablation rate (m/s)
    "d_p": 1e-3,  # particle diameter (m)
    "z0": 1e-6,  # location of first grid point (m)
    "max_z": 0.1,  # cylinder height (m)
    "n_points": 1000  # number of grid points
}


NON_MONOTONIC_PARAMETERS = {
    "T_inf": 20,  # ambient temperature (degrees C)
    "S_inf": 0,  # ambient salinity (g/kg)
    "b0": 5e-4,  # plume breadth at the top (m)
    "w0": 0,  # plume velocity at the top (m/s)
    "phi0": 0.2,  # plume particle volume fraction at the top (-)
    "alpha": 0.004,  # entrainment coefficient (-)
    "w_a": 0,  # ambient water velocity (m/s)
    "dRdt": -1e-5,  # ablation rate (m/s)
    "d_p": 1e-3,  # particle diameter (m)
    "z0": 1e-6,  # location of first grid point (m)
    "max_z": 0.1,  # cylinder height (m)
    "n_points": 1000  # number of grid points
}

DEFAULT_PARAMETERS = BEST_FIT_PARAMETERS


# DEFAULT_PARAMETERS = {
#     "T_inf": 20,  # ambient temperature (degrees C)
#     "S_inf": 0,  # ambient salinity (g/kg)
#     "b0": 5e-5,  # plume breadth at the top (m)
#     "w0": 2e-2,  # plume velocity at the top (m/s)
#     "phi0": 0.0,  # plume particle volume fraction at the top (-)
#     "alpha": 1e-1,  # entrainment coefficient (-)
#     "w_a": 0,  # ambient water velocity (m/s)
#     "dRdt": -1e-5,  # ablation rate (m/s)
#     "d_p": 1e-3,  # particle diameter (m)
#     "z0": 1e-6,  # location of first grid point (m)
#     "max_z": 0.1,  # cylinder height (m)
#     "n_points": 1000  # number of grid points
# }


def main():
    # composite_melt_rate_curve()

    # average_phi_curve()
    # profiles()

    best_model_fit()
    # melt_rate_curve_comparison()
    # average_phi_curve()

    # all_parameter_dependence(pickled=False)
    # best_model_fit()
    # show_melt_rate_curve()
    # melt_rates_natural_forced_comparison()
    melt_rates_ws_comparison()

    # plt.ion()
    # avg_phi_parameter_dependence('alpha', (0, 0.1))
    # avg_phi_parameter_dependence('b0', (1e-5, 1e-2), log=True)
    # avg_phi_parameter_dependence('phi0', (0.01, 0.6))
    # avg_phi_parameter_dependence('w0', (1e-5, 1e-1), log=True)
    # plt.ioff()
    # plt.show()

    """ For checking the dependence of the melt rate curve on parameters """
    # parameter_dependence("phi_s", (0.55, 0.65))
    # multiple_parameter_dependence()

    # plt.ion()
    # parameter_dependence('alpha', (1e-4, 1e-1), log=True)
    # parameter_dependence('b0', (1e-8, 1e-2), log=True)
    # parameter_dependence('phi0', (0.0, 0.6))
    # parameter_dependence('w0', (1e-5, 1e-1), log=True)
    # plt.ioff()
    # plt.show()

    """ For checking the grid size """
    # plt.ion()
    # parameter_dependence('z0', (1e-8, 1e-2), log=True)
    # parameter_dependence('n_points', (100, 10000), log=True)
    # plt.ioff()
    # plt.show()

    """ For checking the melt rate convergence """
    # parameter_dependence('dRdt0', (1e-7, 1e-3), log=True)


def best_model_fit_scipy():
    from scipy.optimize import curve_fit

    def model_fit(x, a, b, c):
        pm = PlumeModel(**DEFAULT_PARAMETERS)
        pm.set_parameter('alpha', a)
        pm.set_parameter('b0', b)
        pm.set_parameter('phi0', c)
        y = np.zeros(len(x))
        for i in range(len(x)):
            pm.d_p = x[i]
            y[i] = pm.converge_melt_rate()
        return y

    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    dp = data_means.index.values * 1e-6  # particle diameter in m
    drdt = data_means['drdt'] * 1e3  # melt rate in m/s

    first_guess = [0.02, 5e-4, 0.06]
    res = curve_fit(model_fit, dp, drdt, first_guess)
    print(res)


def best_model_fit():
    # Experiments
    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    exp_dp = data_means.index.values * 1e-6  # particle diameter in m
    exp_drdt = data_means['drdt'].values  # melt rate in m/s

    # Model
    with open('data/model_4d_var_ws.pkl', 'rb') as f:
        drdt, dp, points = pickle.load(f)

    exp_ind = np.argwhere(exp_dp > 5e-4)

    rms = np.zeros(drdt.shape[0])
    for i in range(drdt.shape[0]):
        err, cnt = 0, 0
        for j in exp_ind:
            ind = np.argmin(np.abs(dp - exp_dp[j]))
            # if not np.isnan(drdt[i, ind]):
            err += (drdt[i, ind] - exp_drdt[j])**2
            cnt += 1
        rms[i] = np.nan if cnt == 0 else np.sqrt(err/cnt)

    opt_ind = np.nanargmin(rms)

    print("Best fit: alpha = {:.2e}  |  b0 = {:.2e} m  |  w0 = {:.2e}  |  phi0 = {:.3f}".format(*points[opt_ind]))

    plt.figure()
    plt.semilogx(exp_dp, exp_drdt, 'o')
    plt.semilogx(exp_dp[exp_ind], exp_drdt[exp_ind], 'o')
    plt.semilogx(dp, drdt[opt_ind, :], '-k', label="least-squares fit")
    plt.legend()
    plt.show()


def composite_model():
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    d_p = np.logspace(-6, -0.5, 200)
    d_pc = 6e-4  # crossover diameter
    a = 20  # 'sharpness' of transition between parameter sets

    drdt = np.zeros(len(d_p))
    phi_errors, conv_errors = 0, 0
    for i in range(len(d_p)):
        pm.d_p = d_p[i]

        for param in ['alpha', 'b0', 'w0', 'phi0']:
            fac = 1./(1 + np.exp(-a * (np.log10(d_p[i]) - np.log10(d_pc))))
            new_val = BEST_FIT_SMALL_PARAMETERS[param] + fac * (BEST_FIT_LARGE_PARAMETERS[param]-BEST_FIT_SMALL_PARAMETERS[param])
            pm.set_parameter(param, new_val)
        try:
            drdt[i] = pm.converge_melt_rate()
        except ArithmeticError:
            drdt[i] = np.nan
            phi_errors += 1
        except ConvergenceError:
            drdt[i] = np.nan
            conv_errors += 1
        print("\r{:d}/{:d}".format(i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))

    # show factors
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]
    Ga = gprime * d_p**3 / nu**2
    fac = 1. / (1 + np.exp(-a * (np.log10(d_p) - np.log10(d_pc))))

    plt.semilogx(Ga, fac, lw=3)
    plt.ylim([0, 1])
    plt.xlim([1.5e-2, 1.5e8])
    # plt.show()
    return d_p, drdt


def composite_melt_rate_curve():
    fresh_water = True
    model_dp, model_drdt = composite_model()

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
    ward_marker = {'markersize': 10, 'linestyle': '', 'mec': 'k', 'mew': 1.5}
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = np.sqrt(gprime * (dp*1e-6)**3) / nu
    Ga_err = np.abs(np.sqrt(gprime * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3)/nu - data_means['Ga'].values)
    data['Ga'] = np.sqrt(gprime * (data['dp']*1e-6)**3) / nu

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-1, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)

    model_Ga = np.sqrt(gprime * model_dp**3) / nu
    # plt.semilogx(model_Ga, -drdt * 1e3, '-k', lw=2, label='model least-squares fit', zorder=5)
    plt.semilogx(model_Ga, -model_drdt * 1e3, '-k', lw=2, label='composite model', zorder=5)

    with open('data/model_random_parameters_phi00.pkl', 'rb') as f:
        spread_drdt, spread_dp, _ = pickle.load(f)
    # with open('model_random_parameters.pkl', 'rb') as f:
    #     spread_drdt, spread_dp, points = pickle.load(f)

    # print('{:.2e} < alpha < {:.2e}'.format(min([p[0] for p in points]), max([p[0] for p in points])))
    # print('{:.2e} < b0 < {:.2e}'.format(min([p[1] for p in points]), max([p[1] for p in points])))
    # print('{:.2e} < phi0 < {:.2e}'.format(min([p[2] for p in points]), max([p[2] for p in points])))

    # spread_Ga = gprime * spread_dp ** 3 / nu ** 2
    # for i in range(spread_drdt.shape[0]):
    #     plt.semilogx(spread_Ga, -spread_drdt[i, :]*1e3, color='0.5', alpha=0.05, zorder=0)

    Ga_Ward = np.sqrt(gprime * (data_Ward2024['d'] * 1e-3)**3) / nu
    ax.plot(Ga_Ward, data_Ward2024['w10'], 's', color="#EA33F7", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w20'], 'o', color="#0000F5", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w30'], '^', color="#EA3323", **ward_marker)

    plt.xlabel('Ga', fontsize=16)
    plt.ylabel('$-\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([2e-2, 1.5e4])
    plt.ylim([0, None])

    plt.show()


def compute_melt_rate_curve(params=DEFAULT_PARAMETERS, use_natural=False, constant_ws=False):
    pm = PlumeModel(**params)
    d_p = np.logspace(-6, -0.5, 100)

    pm.constant_ws = constant_ws

    drdt = np.zeros(len(d_p))
    phi_errors, conv_errors = 0, 0
    for i in range(len(d_p)):
        pm.d_p = d_p[i]
        try:
            drdt[i] = pm.converge_melt_rate(natural=use_natural)
        except ArithmeticError:
            drdt[i] = np.nan
            phi_errors += 1
        except ConvergenceError:
            drdt[i] = np.nan
            conv_errors += 1
        print("\r{:d}/{:d}".format(i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))
    return d_p, drdt


def show_melt_rate_curve():
    fresh_water = True
    d_p, drdt0 = compute_melt_rate_curve(BEST_FIT_PARAMETERS)
    d_p, drdt1 = compute_melt_rate_curve(BEST_FIT_SMALL_PARAMETERS)
    d_p, drdt2 = compute_melt_rate_curve(BEST_FIT_LARGE_PARAMETERS)
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
    ward_marker = {'markersize': 10, 'linestyle': '', 'mec': 'k', 'mew': 1.5}
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = np.sqrt(gprime * (dp*1e-6)**3) / nu
    Ga_err = np.abs(np.sqrt(gprime * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3)/nu - data_means['Ga'].values)
    data['Ga'] = np.sqrt(gprime * (data['dp']*1e-6)**3) / nu

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-1, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)

    model_Ga = np.sqrt(gprime * d_p**3) / nu
    plt.semilogx(model_Ga, -drdt0 * 1e3, '-k', lw=2, label='model least-squares fit', zorder=5)
    plt.semilogx(model_Ga, -drdt1 * 1e3, ':', color='0.6', lw=1.6, label='model lsq fit (Ga < 50)', zorder=4)
    plt.semilogx(model_Ga, -drdt2 * 1e3, '--', color='0.6', lw=1.6, label='model lsq fit (Ga > 50)', zorder=4)
    # plt.semilogx(model_Ga, -drdt * 1e3, '-k', lw=2, label='model', zorder=5)

    with open('data/model_random_parameters_phi00.pkl', 'rb') as f:
        spread_drdt, spread_dp, _ = pickle.load(f)
    # with open('model_random_parameters.pkl', 'rb') as f:
    #     spread_drdt, spread_dp, points = pickle.load(f)

    # print('{:.2e} < alpha < {:.2e}'.format(min([p[0] for p in points]), max([p[0] for p in points])))
    # print('{:.2e} < b0 < {:.2e}'.format(min([p[1] for p in points]), max([p[1] for p in points])))
    # print('{:.2e} < phi0 < {:.2e}'.format(min([p[2] for p in points]), max([p[2] for p in points])))

    # spread_Ga = gprime * spread_dp ** 3 / nu ** 2
    # for i in range(spread_drdt.shape[0]):
    #     plt.semilogx(spread_Ga, -spread_drdt[i, :]*1e3, color='0.5', alpha=0.05, zorder=0)

    Ga_Ward = np.sqrt(gprime * (data_Ward2024['d'] * 1e-3)**3) / nu
    ax.plot(Ga_Ward, data_Ward2024['w10'], 's', color="#EA33F7", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w20'], 'o', color="#0000F5", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w30'], '^', color="#EA3323", **ward_marker)

    plt.xlabel(r"Ga=$\sqrt{g'd}/\nu$", fontsize=16)
    plt.ylabel('$-\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([2e-1, 1.5e4])
    plt.ylim([0, None])

    plt.show()


def melt_rates_natural_forced_comparison():
    fresh_water = True
    d_p, drdt0 = compute_melt_rate_curve(BEST_FIT_PARAMETERS)
    d_p, drdt1 = compute_melt_rate_curve(BEST_FIT_PARAMETERS, use_natural=True)
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
    ward_marker = {'markersize': 10, 'linestyle': '', 'mec': 'k', 'mew': 1.5}
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = np.sqrt(gprime * (dp*1e-6)**3) / nu
    Ga_err = np.abs(np.sqrt(gprime * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3)/nu - data_means['Ga'].values)
    data['Ga'] = np.sqrt(gprime * (data['dp']*1e-6)**3) / nu

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-1, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)

    model_Ga = np.sqrt(gprime * d_p**3) / nu
    plt.semilogx(model_Ga, -drdt0 * 1e3, '-k', lw=2, label='forced convection', zorder=5)
    plt.semilogx(model_Ga, -drdt1 * 1e3, '-', color='0.6', lw=1.6, label='natural convection', zorder=4)
    # plt.semilogx(model_Ga, -drdt * 1e3, '-k', lw=2, label='model', zorder=5)

    plt.xlabel(r"Ga=$\sqrt{g'd}/\nu$", fontsize=16)
    plt.ylabel('$-\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([2e-1, 1.5e4])
    plt.ylim([0, None])

    plt.show()


def melt_rates_ws_comparison():
    fresh_water = True
    d_p, drdt0 = compute_melt_rate_curve(BEST_FIT_PARAMETERS, constant_ws=True)
    d_p, drdt1 = compute_melt_rate_curve(BEST_FIT_SMALL_PARAMETERS)
    d_p, drdt2 = compute_melt_rate_curve(BEST_FIT_PARAMETERS, constant_ws=False)
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
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = np.sqrt(gprime * (dp*1e-6)**3) / nu
    Ga_err = np.abs(np.sqrt(gprime * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3)/nu - data_means['Ga'].values)
    data['Ga'] = np.sqrt(gprime * (data['dp']*1e-6)**3) / nu

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-1, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)

    model_Ga = np.sqrt(gprime * d_p**3) / nu
    plt.semilogx(model_Ga, -drdt0 * 1e3, '-k', lw=2, label='constant ws', zorder=5)
    plt.semilogx(model_Ga, -drdt1 * 1e3, ':', color='0.6', lw=1.6, label='constant ws (Ga < 50)', zorder=4)
    plt.semilogx(model_Ga, -drdt2 * 1e3, '--', color='0.6', lw=1.6, label='height-averaged ws', zorder=4)

    plt.xlabel(r"Ga=$\sqrt{g'd}/\nu$", fontsize=16)
    plt.ylabel('$-\dot{R}$ [mm/s]', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([2e-1, 1.5e4])
    plt.ylim([0, None])

    plt.show()


def average_phi_curve():
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    d_p = np.logspace(-6, -1, 50)

    phi_bar = np.zeros(len(d_p))
    drdt = np.zeros(len(d_p))
    phi_errors, conv_errors = 0, 0
    for i in range(len(d_p)):
        pm.d_p = d_p[i]
        try:
            drdt[i] = pm.converge_melt_rate()
            phi = pm.compute_quantities('phi')
            phi_bar[i] = np.sum(phi * np.abs(np.diff(pm.z))) / pm.H
        except ArithmeticError:
            drdt[i] = np.nan
            phi_bar[i] = np.nan
            phi_errors += 1
        except ConvergenceError:
            drdt[i] = np.nan
            phi_bar[i] = np.nan
            conv_errors += 1
        print("\r{:d}/{:d}".format(i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))

    color1 = (0, 0.4, 0.7)
    color2 = (0.7, 0, 0)

    plt.semilogx(d_p, phi_bar, color=color1)
    plt.ylim([0, None])
    plt.ylabel("$\overline{\phi}$", fontsize=14, color=color1)
    plt.tick_params(labelsize=12)
    plt.tick_params(color=color1, labelcolor=color1, axis='y')
    plt.xlabel("$d_p$ (m)", fontsize=14)

    ax2 = plt.gca().twinx()
    ax2.semilogx(d_p, -drdt*1e3, color=color2)
    ax2.set_ylabel("$-\dot{R}$ (mm/s)", fontsize=14, color=color2)
    ax2.tick_params(color=color2, labelcolor=color2, labelsize=12)
    ax2.set_ylim([0, None])
    plt.tight_layout()
    plt.show()


def avg_phi_parameter_dependence(param: str, limits: tuple, log=False):
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    cmaps = {'alpha': plt.get_cmap('Blues'), 'b0': plt.get_cmap('Greens'), 'phi0': plt.get_cmap('Reds')}
    cmap = plt.get_cmap('rainbow') if param not in cmaps else cmaps[param]

    d_p = np.logspace(-6, -1, 50)
    if log:
        q = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 10)
    else:
        q = np.linspace(limits[0], limits[1], 10)
    if param == 'dRdt0':
        q = -q

    phi_bar = np.zeros((len(q), len(d_p)))
    phi_errors, conv_errors = 0, 0
    for j in range(len(q)):
        if param != 'dRdt0':
            pm.set_parameter(param, q[j])
        for i in range(len(d_p)):
            pm.d_p = d_p[i]
            try:
                kwargs = {"dRdt0": q[j]} if param == 'dRdt0' else {}
                pm.converge_melt_rate(**kwargs)
                phi = pm.compute_quantities('phi')
                phi_bar[j, i] = np.sum(phi * np.abs(np.diff(pm.z))) / pm.H
            except ArithmeticError:
                phi_bar[j, i] = np.nan
                phi_errors += 1
            except ConvergenceError:
                phi_bar[j, i] = np.nan
                conv_errors += 1
            print("\r{:d}/{:d} ({:d}/{:d})".format(j+1, len(q), i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))

    plt.figure()
    label = {'alpha': r'$\alpha$', 'b0': '$b_0$', 'w0': '$w_0$', 'phi0': '$\phi_0$', 'dRdt': '$dR/dt$', 'w_a': '$w_a$',
             'z0': '$z_0$', 'n_points': 'N', 'max_z': '$H$', 'dRdt0': '$(dR/dt)_0$'}[param]
    unit = {'alpha': '', 'b0': 'm', 'w0': 'm/s', 'phi0': '', 'dRdt': 'm/s', 'w_a': 'm/s',
             'z0': 'm', 'n_points': 'N', 'max_z': 'm', 'dRdt0': 'm/s'}[param]
    for j in range(len(q)):
        color = cmap(0.4 + 0.6*(j/len(q)))
        lw = 3.5-3*(j/len(q)) if param == 'dRdt0' else 1.5
        if not np.all(np.isnan(phi_bar[j, :])):
            plt.semilogx(d_p, phi_bar[j, :], label=label+' = {:.1e} '.format(q[j]) + unit, color=color, lw=lw)
    plt.ylabel(r'$\bar{\phi}$')
    plt.xlabel('$d_p$ (m)')
    plt.title(label+'-dependence')
    plt.ylim([0, None])
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def parameter_dependence(param: str, limits: tuple, log=False):
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    cmaps = {'alpha': plt.get_cmap('Blues'), 'b0': plt.get_cmap('Greens'), 'phi0': plt.get_cmap('Reds'), 'w0': plt.get_cmap('Purples')}
    cmap = plt.get_cmap('rainbow') if param not in cmaps else cmaps[param]

    d_p = np.logspace(-6, -1, 50)
    if log:
        q = np.logspace(np.log10(limits[0]), np.log10(limits[1]), 10)
    else:
        q = np.linspace(limits[0], limits[1], 10)
    if param == 'dRdt0':
        q = -q

    drdt = np.zeros((len(q), len(d_p)))
    phi_errors, conv_errors = 0, 0
    for j in range(len(q)):
        if param != 'dRdt0':
            pm.set_parameter(param, q[j])
        for i in range(len(d_p)):
            pm.d_p = d_p[i]
            try:
                kwargs = {"dRdt0": q[j]} if param == 'dRdt0' else {}
                drdt[j, i] = pm.converge_melt_rate(**kwargs)
            except ArithmeticError:
                drdt[j, i] = np.nan
                phi_errors += 1
            except ConvergenceError:
                drdt[j, i] = np.nan
                conv_errors += 1
            print("\r{:d}/{:d} ({:d}/{:d})".format(j+1, len(q), i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))

    plt.figure()
    label = {'alpha': r'$\alpha$', 'b0': '$b_0$', 'w0': '$w_0$', 'phi0': '$\phi_0$', 'dRdt': '$dR/dt$', 'w_a': '$w_a$',
             'z0': '$z_0$', 'n_points': 'N', 'max_z': '$H$', 'dRdt0': '$(dR/dt)_0$', 'phi_s': '$\phi_s$'}[param]
    unit = {'alpha': '', 'b0': 'm', 'w0': 'm/s', 'phi0': '', 'dRdt': 'm/s', 'w_a': 'm/s',
             'z0': 'm', 'n_points': 'N', 'max_z': 'm', 'dRdt0': 'm/s', 'phi_s': ''}[param]
    for j in range(len(q)):
        color = cmap(0.4 + 0.6*(j/len(q)))
        lw = 3.5-3*(j/len(q)) if param == 'dRdt0' else 1.5
        if not np.all(np.isnan(drdt[j, :])):
            plt.semilogx(d_p, -drdt[j, :] * 1e3, label=label+' = {:.1e} '.format(q[j]) + unit, color=color, lw=lw)
    plt.ylabel('$-\dot{R}$ (mm/s)', fontsize=14)
    plt.xlabel('$d_p$ (m)', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.title(label+'-dependence')
    plt.ylim([0, 0.22])
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.tight_layout()
    plt.show()


def multiple_parameter_dependence():
    params = ['alpha', 'b0', 'phi0', 'w0']
    limits = [[1e-4, 1e-1], [1e-5, 1e-1], [0, 0.6], [1e-5, 1e-1]]
    logs = [True, True, False, True]

    cmaps = [plt.get_cmap('Blues'), plt.get_cmap('Greens'), plt.get_cmap('Reds'), plt.get_cmap('Purples')]
    d_p = np.logspace(-5, -1, 50)

    fig, axes = plt.subplots(1, len(params), figsize=[2+3*len(params), 6], sharey=True, sharex=True)

    for p in range(len(params)):
        pm = PlumeModel(**DEFAULT_PARAMETERS)
        if logs[p]:
            q = np.logspace(np.log10(limits[p][0]), np.log10(limits[p][1]), 10)
        else:
            q = np.linspace(limits[p][0], limits[p][1], 10)
        if params[p] == 'dRdt0':
            q = -q

        drdt = np.zeros((len(q), len(d_p)))
        phi_errors, conv_errors = 0, 0
        for j in range(len(q)):
            if params[p] != 'dRdt0':
                pm.set_parameter(params[p], q[j])
            for i in range(len(d_p)):
                pm.d_p = d_p[i]
                try:
                    kwargs = {"dRdt0": q[j]} if params[p] == 'dRdt0' else {}
                    drdt[j, i] = pm.converge_melt_rate(**kwargs)
                except ArithmeticError:
                    drdt[j, i] = np.nan
                    phi_errors += 1
                except ConvergenceError:
                    drdt[j, i] = np.nan
                    conv_errors += 1
                print("\r[{:s}] {:d}/{:d} ({:d}/{:d})".format(params[p], j+1, len(q), i+1, len(d_p)), end='')
        print('\ndone!')
        print('{:d} errors on phi < 0'.format(phi_errors))
        print('{:d} convergence errors'.format(conv_errors))

        label = {'alpha': r'$\alpha$', 'b0': '$b_0$', 'w0': '$w_0$', 'phi0': '$\phi_0$', 'dRdt': '$dR/dt$', 'w_a': '$w_a$',
                 'z0': '$z_0$', 'n_points': 'N', 'max_z': '$H$', 'dRdt0': '$(dR/dt)_0$'}[params[p]]
        unit = {'alpha': '', 'b0': 'm', 'w0': 'm/s', 'phi0': '', 'dRdt': 'm/s', 'w_a': 'm/s',
                 'z0': 'm', 'n_points': 'N', 'max_z': 'm', 'dRdt0': 'm/s'}[params[p]]
        for j in range(len(q)):
            color = cmaps[p](0.4 + 0.6*(j/len(q)))
            lw = 3.5-3*(j/len(q)) if params[p] == 'dRdt0' else 1.5
            if not np.all(np.isnan(drdt[j, :])):
                axes[p].semilogx(d_p, -drdt[j, :] * 1e3, label=label+' = {:.1e} '.format(q[j]) + unit, color=color, lw=lw)

        axes[p].set_xlabel('$d_p$ (m)', fontsize=14)
        axes[p].tick_params(labelsize=12, labelleft=p==0, right=p<len(params)-1)
        axes[p].set_title(label, fontsize=16)
        axes[p].set_ylim([0, 0.22])
        axes[p].set_xlim([1e-5, 1e-1])

    axes[0].set_ylabel('$-\dot{R}$ (mm/s)', fontsize=14)
    axes[0].set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    fig.tight_layout()
    plt.show()


def all_parameter_dependence(pickled=False):
    if not pickled:
        pm = PlumeModel(**DEFAULT_PARAMETERS)

        d_p = np.logspace(-6, -1, 50)

        # alpha_arr = [0] + list(np.logspace(-3, -1, 9))
        # b0_arr = np.logspace(-5, -1, 10)
        # phi0_arr = np.linspace(0, 0.6, 10)
        # points = [(alpha, b0, phi0) for alpha in alpha_arr for b0 in b0_arr for phi0 in phi0_arr]

        a_limits = [-5, 0]
        b_limits = [-8, -1]
        w_limits = [-5, 0]
        limits = [a_limits, b_limits, w_limits]
        points = [[10**(np.random.random()*(lim[1]-lim[0]) + lim[0]) for lim in limits] for _ in range(1000)]

        drdt = np.zeros((len(points), len(d_p)))
        phi_errors, conv_errors = 0, 0
        for j in range(len(points)):
            pm.set_parameter('alpha', points[j][0])
            pm.set_parameter('b0', points[j][1])
            pm.set_parameter('w0', points[j][2])
            for i in range(len(d_p)):
                pm.d_p = d_p[i]
                try:
                    drdt[j, i] = pm.converge_melt_rate()
                except ArithmeticError:
                    drdt[j, i] = np.nan
                    phi_errors += 1
                except ConvergenceError:
                    drdt[j, i] = np.nan
                    conv_errors += 1
                print("\r{:d}/{:d} [".format(j+1, len(points)) + "#"*int((i+1)/len(d_p)*10) + "-"*(10-int((i+1)/len(d_p)*10))+"]", end='')
        print('\ndone!')
        print('{:d} errors on phi < 0'.format(phi_errors))
        print('{:d} convergence errors'.format(conv_errors))

        with open('data/model_random_parameters_phi00_local_dT.pkl', 'wb') as f:
            pickle.dump([drdt, d_p, points], f)
    else:
        with open('data/model_random_parameters_phi00_local_dT.pkl', 'rb') as f:
            drdt, d_p, points = pickle.load(f)

    min_idx = np.nanargmin(np.mean(np.abs(drdt), axis=1))
    max_idx = np.nanargmax(np.mean(np.abs(drdt), axis=1))

    non_nan_drdt = np.array([drdt[j, :] for j in range(drdt.shape[0]) if not np.any(np.isnan(drdt[j, :]))])

    plt.figure()
    for j in range(len(points)):
        plt.semilogx(d_p, -drdt[j, :] * 1e3, color='0.5', alpha=0.05)
    # plt.semilogx(d_p, -np.nanmean(non_nan_drdt, axis=0) * 1e3, color='k', lw=2, label='mean')
    plt.semilogx(d_p, -drdt[min_idx, :] * 1e3, color=(0, 0, 0.6), lw=1.5, label='min')
    plt.semilogx(d_p, -drdt[max_idx, :] * 1e3, color=(0.6, 0, 0), lw=1.5, label='max')

    min_text = r'($\alpha = {:.2f}$, $b_0 = {:.1e}$, $w_0 = {:.2f}$)'.format(*points[min_idx])
    max_text = r'($\alpha = {:.2f}$, $b_0 = {:.1e}$, $w_0 = {:.2f}$)'.format(*points[max_idx])
    plt.text(1e-6, 0.05, min_text, color=(0, 0, 0.6))
    plt.text(1e-6, 0.28, max_text, color=(0.6, 0, 0))

    print(points[min_idx])
    print(points[max_idx])

    plt.ylabel('-dR/dt (mm/s)')
    plt.xlabel('$d_p$ (m)')
    plt.ylim([0, 0.3])
    plt.legend()
    plt.tight_layout()
    plt.show()


def profiles():
    particle_sizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # in m
    cmap = plt.get_cmap('Oranges')
    colors = [cmap((i+1)/(len(particle_sizes) + 1)) for i in range(len(particle_sizes))]

    pm = PlumeModel(**DEFAULT_PARAMETERS)

    b = np.zeros((pm.n_points-1, len(particle_sizes)))
    w = np.zeros((pm.n_points-1, len(particle_sizes)))
    phi = np.zeros((pm.n_points-1, len(particle_sizes)))
    T = np.zeros((pm.n_points-1, len(particle_sizes)))
    S = np.zeros((pm.n_points-1, len(particle_sizes)))

    for i in range(len(particle_sizes)):
        pm.d_p = particle_sizes[i]
        try:
            pm.converge_melt_rate()
            b[:, i], w[:, i], phi[:, i], T[:, i], S[:, i] = pm.compute_quantities()
        except ArithmeticError:
            print("[profiles] phi < 0 on dp = {:.0f} um".format(particle_sizes[i] * 1e6))
            b[:, i], w[:, i], phi[:, i], T[:, i], S[:, i] = np.nan, np.nan, np.nan, np.nan, np.nan
        except ConvergenceError:
            print("[profiles] Convergence error on dp = {:.0f} um".format(particle_sizes[i] * 1e6))
            b[:, i], w[:, i], phi[:, i], T[:, i], S[:, i] = np.nan, np.nan, np.nan, np.nan, np.nan

    fig, axes = plt.subplots(1, 5, figsize=[12, 6], sharey=True)
    z = pm.z[1:]
    for i in range(len(particle_sizes)):
        axes[0].plot(b[:, i]*1e3, z, color=colors[i], lw=2)
        axes[1].plot(w[:, i], z, color=colors[i], lw=2)
        axes[2].plot(phi[:, i], z, color=colors[i], lw=2)
        axes[3].plot(T[:, i], z, color=colors[i], lw=2)
        axes[4].plot(S[:, i], z, color=colors[i], lw=2, label="$d_p = 10^{"+"{:.0f}".format(np.log10(particle_sizes[i]))+"}$ m")

    for a in range(len(axes)):
        axes[a].tick_params(labelleft=(a==0), right=(a<len(axes)-1), labelsize=12)
        axes[a].set_xlim([0, None])
        axes[a].set_ylim([-pm.H, 0])
    axes[0].set_xlabel('$b$ (mm)', fontsize=16)
    axes[1].set_xlabel('$w$ (m/s)', fontsize=16)
    axes[2].set_xlabel('$\phi$ (-)', fontsize=16)
    axes[3].set_xlabel('$T$ ($\degree$C)', fontsize=16)
    axes[4].set_xlabel('$S$ (g/kg)', fontsize=16)
    axes[0].set_ylabel('$z$ (m)', fontsize=16)
    axes[-1].legend(fontsize=12, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()

