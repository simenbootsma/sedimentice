import numpy as np
import matplotlib.pyplot as plt
from PlumeModel import PlumeModel, ConvergenceError
import matplotlib
import pandas as pd
import pickle
import time
import cmcrameri.cm as cmc
import cmocean.cm as cmo

matplotlib.use('Qt5Agg')

DEFAULT_PARAMETERS = {
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


# DEFAULT_PARAMETERS = {
#     "T_inf": 20,  # ambient temperature (degrees C)
#     "S_inf": 0,  # ambient salinity (g/kg)
#     "b0": 5e-4,  # plume breadth at the top (m)
#     "w0": 0,  # plume velocity at the top (m/s)
#     "phi0": 0.2,  # plume particle volume fraction at the top (-)
#     "alpha": 0.004,  # entrainment coefficient (-)
#     "w_a": 0,  # ambient water velocity (m/s)
#     "dRdt": -1e-5,  # ablation rate (m/s)
#     "d_p": 1e-3,  # particle diameter (m)
#     "z0": 1e-6,  # location of first grid point (m)
#     "max_z": 0.1,  # cylinder height (m)
#     "n_points": 1000  # number of grid points
# }


def main():
    is_model_stable()
    is_there_a_maximum()
    # parameter_dependence_4d(pickled=True)
    # profiles()

    # best_model_fit()
    # melt_rate_curve_comparison()
    # average_phi_curve()

    # all_parameter_dependence(pickled=False)
    # best_model_fit()
    melt_rate_curve()

    # plt.ion()
    # avg_phi_parameter_dependence('alpha', (0, 0.1))
    # avg_phi_parameter_dependence('b0', (1e-5, 1e-2), log=True)
    # avg_phi_parameter_dependence('phi0', (0.01, 0.6))
    # avg_phi_parameter_dependence('w0', (1e-5, 1e-1), log=True)
    # plt.ioff()
    # plt.show()

    """ For checking the dependence of the melt rate curve on parameters """
    # parameter_dependence("phi_s", (0.55, 0.65))
    multiple_parameter_dependence()

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
    with open('data/model_random_parameters_phi00.pkl', 'rb') as f:
        drdt, dp, points = pickle.load(f)

    rms = np.zeros(drdt.shape[0])
    for i in range(drdt.shape[0]):
        err, cnt = 0, 0
        for j in range(len(exp_dp)):
            ind = np.argmin(np.abs(dp - exp_dp[j]))
            # if not np.isnan(drdt[i, ind]):
            err += (drdt[i, ind] - exp_drdt[j])**2
            cnt += 1
        rms[i] = np.sqrt(err/cnt)

    opt_ind = np.nanargmin(rms)

    print("Best fit: alpha = {:.2e}  |  b0 = {:.2e} m  |  w0 = {:.2e}".format(*points[opt_ind]))

    plt.figure()
    plt.semilogx(exp_dp, exp_drdt, 'o')
    plt.semilogx(dp, drdt[opt_ind, :], '-k', label="least-squares fit")
    plt.legend()
    plt.show()


def melt_rate_curve():
    fresh_water = True
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    d_p = np.logspace(-6, -0.5, 100)

    drdt = np.zeros(len(d_p))
    phi_errors, conv_errors = 0, 0
    for i in range(len(d_p)):
        pm.d_p = d_p[i]
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

    fig, ax = plt.subplots(num=3, figsize=(9, 8))
    ax.set_xscale("log")

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-2, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)

    model_Ga = gprime * d_p**3 / nu**2
    # plt.semilogx(model_Ga, -drdt * 1e3, '-k', lw=2, label='model least-squares fit', zorder=5)
    plt.semilogx(model_Ga, -drdt * 1e3, '-k', lw=2, label='model', zorder=5)

    with open('data/model_random_parameters_phi00.pkl', 'rb') as f:
        spread_drdt, spread_dp, _ = pickle.load(f)
    # with open('model_random_parameters.pkl', 'rb') as f:
    #     spread_drdt, spread_dp, points = pickle.load(f)

    # print('{:.2e} < alpha < {:.2e}'.format(min([p[0] for p in points]), max([p[0] for p in points])))
    # print('{:.2e} < b0 < {:.2e}'.format(min([p[1] for p in points]), max([p[1] for p in points])))
    # print('{:.2e} < phi0 < {:.2e}'.format(min([p[2] for p in points]), max([p[2] for p in points])))

    spread_Ga = gprime * spread_dp ** 3 / nu ** 2
    for i in range(spread_drdt.shape[0]):
        plt.semilogx(spread_Ga, -spread_drdt[i, :]*1e3, color='0.5', alpha=0.05, zorder=0)

    plt.xlabel('Ga', fontsize=16)
    plt.ylabel('$-\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([1.5e-2, 1.5e8])
    plt.ylim([0, None])

    plt.show()


def average_phi_curve():
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    d_p = np.logspace(-6, -1, 50)

    phi_bar = np.zeros(len(d_p))
    phi_errors, conv_errors = 0, 0
    for i in range(len(d_p)):
        pm.d_p = d_p[i]
        try:
            pm.converge_melt_rate()
            phi = pm.compute_quantities('phi')
            phi_bar[i] = np.sum(phi * np.abs(np.diff(pm.z))) / pm.H
        except ArithmeticError:
            phi_bar[i] = np.nan
            phi_errors += 1
        except ConvergenceError:
            phi_bar[i] = np.nan
            conv_errors += 1
        print("\r{:d}/{:d}".format(i+1, len(d_p)), end='')
    print('\ndone!')
    print('{:d} errors on phi < 0'.format(phi_errors))
    print('{:d} convergence errors'.format(conv_errors))

    plt.semilogx(d_p, phi_bar)
    plt.ylim([0, None])
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


def parameter_dependence_4d(pickled=False):
    if not pickled:
        pm = PlumeModel(**DEFAULT_PARAMETERS)

        d_p = np.logspace(-6, -1, 30)

        N = 10
        a_arr = np.logspace(-5, 0, N)
        b_arr = np.logspace(-8, -1, N)
        w_arr = np.logspace(-5, 0, N)
        p_arr = np.linspace(0, 0.6, N)
        points = [[a, b, w, p] for a in a_arr for b in b_arr for w in w_arr for p in p_arr]

        drdt = np.zeros((len(points), len(d_p)))
        phi_errors, conv_errors = 0, 0
        st = time.time()
        last10ticks = np.zeros(10)
        tick_ind = 0
        for j in range(len(points)):
            pm.set_parameter('alpha', points[j][0])
            pm.set_parameter('b0', points[j][1])
            pm.set_parameter('w0', points[j][2])
            pm.set_parameter('phi0', points[j][3])
            st2 = time.time()
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
            sec_left = (time.time() - st) * (len(points) / (j+1) - 1)
            last10ticks[tick_ind] = time.time() - st2
            tick_ind = (tick_ind + 1) % len(last10ticks)
            tm_str = "done in {:02d}:{:02d}:{:02d} (tick: {:d} ms)".format(int(sec_left/3600), int((sec_left % 3600)/60), int(sec_left%60), int(np.mean(last10ticks)*1000))
            print("\r{:d}/{:d} | {:s}".format(j+1, len(points), tm_str), end='')
        print('\ndone!')
        print('{:d} errors on phi < 0'.format(phi_errors))
        print('{:d} convergence errors'.format(conv_errors))

        with open('data/model_4d.pkl', 'wb') as f:
            pickle.dump([drdt, d_p, points], f)
    else:
        with open('data/model_4d.pkl', 'rb') as f:
            drdt, d_p, points = pickle.load(f)

    min_idx = np.nanargmin(np.mean(np.abs(drdt), axis=1))
    max_idx = np.nanargmax(np.mean(np.abs(drdt), axis=1))

    # non_nan_drdt = np.array([drdt[j, :] for j in range(drdt.shape[0]) if not np.any(np.isnan(drdt[j, :]))])

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


def is_there_a_maximum():
    with open('data/model_4d.pkl', 'rb') as f:
        drdt, d_p, points = pickle.load(f)

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))

    max_dp = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    indices = np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)), dtype=np.int32)
    for n in range(len(points)):
        nn_ind = np.where(~np.isnan(drdt[n, :]))[0]
        if len(nn_ind) > 2:
            first, last, in_between = np.abs(drdt[n, nn_ind[0]]), np.abs(drdt[n, nn_ind[-1]]), np.abs(drdt[n, nn_ind[1:-1]])
            if np.any(in_between > first) and np.any(in_between > last):
                i = np.argmin(np.abs(points[n][0] - alpha_arr))
                j = np.argmin(np.abs(points[n][1] - b0_arr))
                k = np.argmin(np.abs(points[n][2] - w0_arr))
                l = np.argmin(np.abs(points[n][3] - phi0_arr))
                max_dp[i, j, k, l] = d_p[np.nanargmax(np.abs(drdt[n, :]))]
                indices[i, j, k, l] = n

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(max_dp[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))
    #
    # plt.figure()
    # i = -1
    # for j in range(len(b0_arr)):
    #     n = indices[i, j, k, l]
    #     plt.semilogx(d_p, -drdt[n, :]*1e3, label="$b_0 = {:.1e}$ m".format(b0_arr[j]))
    # plt.title(r"$\alpha = {:.1e}$".format(alpha_arr[i]))
    # plt.legend()
    # plt.show()

    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmc.show_cmaps()
    cmap = cmo.matter_r

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(np.log10(max_dp[:, :, K[ki], L[li]])), extent=extent, vmin=-6, vmax=-2, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]  #list(range(int(extent[0]), int(extent[1]+1)))
    yt = [-4, -2, 0]  #list(range(int(extent[2]), int(extent[3]+1)))
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        # axes[ki, 0].set_ylabel(r"$\alpha$", fontsize=12)
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        # axes[ki, -1].set_ylabel('$w_0 = {:.1f}'.format(w0*10**(-w0_exp))+'\cdot 10^{'+"{:d}".format(w0_exp)+'}$ m/s',
        #                         fontsize=10, rotation=270, labelpad=20)
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        # axes[-1, li].set_xlabel("$b_0$ (m)", fontsize=12)
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$d_p^*$ (m)\n", fontsize=12)
    cbt = [-6.5, -6.25] + list(range(-6, -1))
    cbtl = ["$10^{"+"{:.0f}".format(val)+"}$" for val in cbt]
    cbtl[0] = ""
    cbtl[1] = "no maximum"
    cb.set_ticks(cbt, labels=cbtl)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def is_model_stable():
    with open('data/model_4d.pkl', 'rb') as f:
        drdt, d_p, points = pickle.load(f)

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))

    stability = np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    indices = np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)), dtype=np.int32)
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))
        stability[i, j, k, l] = np.sum(~np.isnan(drdt[n, :])) / len(d_p)
        indices[i, j, k, l] = n

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(max_dp[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))
    #
    # plt.figure()
    # i = -1
    # for j in range(len(b0_arr)):
    #     n = indices[i, j, k, l]
    #     plt.semilogx(d_p, -drdt[n, :]*1e3, label="$b_0 = {:.1e}$ m".format(b0_arr[j]))
    # plt.title(r"$\alpha = {:.1e}$".format(alpha_arr[i]))
    # plt.legend()
    # plt.show()

    # All slices
    delta = 2  # Show every delta-th slice in k and l
    # cmc.show_cmaps()
    cmap = plt.get_cmap('RdYlGn_r')

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(100*(1-stability[:, :, K[ki], L[li]])), extent=extent, vmin=0, vmax=100, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]  #list(range(int(extent[0]), int(extent[1]+1)))
    yt = [-4, -2, 0]  #list(range(int(extent[2]), int(extent[3]+1)))
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        # axes[ki, 0].set_ylabel(r"$\alpha$", fontsize=12)
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        # axes[ki, -1].set_ylabel('$w_0 = {:.1f}'.format(w0*10**(-w0_exp))+'\cdot 10^{'+"{:d}".format(w0_exp)+'}$ m/s',
        #                         fontsize=10, rotation=270, labelpad=20)
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        # axes[-1, li].set_xlabel("$b_0$ (m)", fontsize=12)
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("% NaN", fontsize=12)
    cb.ax.tick_params(labelsize=12)

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

