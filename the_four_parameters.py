from PlumeModel import PlumeModel, ConvergenceError, settling_velocity
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cmocean.cm as cmo
import cmocean
from plume_model_results import DEFAULT_PARAMETERS
import time
from multiprocessing import Pool, Process, Manager
from glob import glob
import warnings
import os
from scipy.stats import pearsonr
import pandas as pd


warnings.filterwarnings("ignore")
SALINE_WATER = False
SAVE_FOLDER = 'data/var_ws/saline_profiles/' if SALINE_WATER else 'data/var_ws/fresh_profiles/'

NO_ERROR = 0
PHI_ERROR = 1
CONV_ERROR = 2


def main():
    # breadth_gradient_at_top(1e-3)
    # is_model_stable()
    compute_4d_melt_rate_parallel(save_path='data/model_4d_var_ws.pkl')
    compute_4d_profiles_parallel()
    # print(res)

    # with open(save_path, 'wb') as f:
    #     pickle.dump([res, d_p, points], f)
    # compute_4d_melt_rate(pickled=False, save_path='data/model_4d_var_ws.pkl')
    # model_experiment_difference()
    # phi_bar_inflection_with_dp()
    # phi_bar_melt_rate_correlation()
    # does_b_decrease_with_z(1e-5)
    # does_b_only_decrease_with_z()
    # does_phi_bar_increase_with_dp()
    # average_phi()
    # is_there_a_maximum()
    # is_there_a_maximum_in_avg_momentum()
    # is_there_a_maximum_in_phi_bar()
    # is_model_stable()

    # compute_profiles_4d_parallel()


def target(args):
    p, queue, d_p = args
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    pm.constant_ws = False
    pm.set_parameter('alpha', p[0])
    pm.set_parameter('b0', p[1])
    pm.set_parameter('w0', p[2])
    pm.set_parameter('phi0', p[3])
    drdt = np.zeros(len(d_p))
    for i in range(len(d_p)):
        pm.d_p = d_p[i]
        try:
            drdt[i] = pm.converge_melt_rate()
        except ArithmeticError:
            drdt[i] = np.nan
        except ConvergenceError:
            drdt[i] = np.nan
    queue.put(drdt)
    return drdt


def progress(queue, max_len):
    cnt = 0
    while cnt < max_len:
        print("\r[progress] {:.1f}%".format(cnt / max_len * 100), end='')
        queue.get()
        cnt += 1
        time.sleep(0.1)
    print(' \033[42mdone!\033[0m')


def compute_4d_melt_rate_parallel(save_path='data/model_4d.pkl'):
    d_p = np.logspace(-6, -1, 30)

    N = 10
    a_arr = np.logspace(-5, 0, N)
    b_arr = np.logspace(-8, -1, N)
    w_arr = np.logspace(-5, 0, N)
    p_arr = np.linspace(0, 0.6, N)
    points = [[a, b, w, p] for a in a_arr for b in b_arr for w in w_arr for p in p_arr]

    q = Manager().Queue()
    proc = Process(target=progress, args=(q, N**4))
    proc.start()
    pool = Pool()
    result = pool.map_async(target, [(pnt, q, d_p) for pnt in points])
    res = result.get()

    drdt = np.zeros((len(points), len(d_p)))
    for i, arr in enumerate(res):
        drdt[i, :] = arr

    with open(save_path, 'wb') as f:
        pickle.dump([drdt, d_p, points], f)

    print("Saved to {:s}!".format(save_path))


def compute_4d_profiles_parallel(n_cores=10):
    n_files = len(glob(SAVE_FOLDER + "*.npz"))
    if n_files > 0 and input("\033[91mdelete\033[0m all \033[36m{:d}\033[0m files in folder \033[95m{:s}\033[0m? (y/n) ".format(n_files, SAVE_FOLDER)).lower() == "y":
        for fn in glob(SAVE_FOLDER + "*.npz"):
            os.remove(fn)

    N = 10
    a_arr = np.logspace(-5, 0, N)
    b_arr = np.logspace(-8, -1, N)
    w_arr = np.logspace(-5, 0, N)
    p_arr = np.linspace(0, 0.6, N)

    # a_arr = 2.**np.arange(-3, 4) * DEFAULT_PARAMETERS['alpha']
    # b_arr = 2. ** np.arange(-3, 4) * DEFAULT_PARAMETERS['b0']
    # w_arr = 8.**np.arange(-2, 3) * DEFAULT_PARAMETERS['w0']
    # p_arr = [0, 5e-3, 1e-2, 2e-2, 4e-2]

    points = [[a, b, w, p] for a in a_arr for b in b_arr for w in w_arr for p in p_arr]
    chunk_size = int(np.ceil(len(points)/n_cores))
    points_split = [points[i*chunk_size:min((i+1)*chunk_size, len(points))] for i in range(n_cores)]

    up_proc = Process(target=update, args=(SAVE_FOLDER, len(points)))
    up_proc.start()
    with Pool(n_cores) as pool:
        pool.map(compute_4d_profiles, points_split)
    up_proc.join()


def update(folder, n_files):
    st = time.time()
    while len(glob(folder + "*.npz")) < n_files:
        frac = len(glob(folder + "*.npz")) / n_files
        sec_left = np.nan if frac == 0 else (time.time() - st) * (1/frac - 1)
        tm_str = '-' if frac == 0 else "{:02d}:{:02d}:{:02d} remaining".format(int(sec_left / 3600), int((sec_left % 3600)/60), int(sec_left % 60))
        print("\r[\033[32m{:.0f}%\033[0m] {:s}".format(100*frac, tm_str), end='')
        time.sleep(0.1)
    print("\r[\033[32m100%\033[0m] done!", end='')


def compute_4d_profiles(points):
    pm = PlumeModel(**DEFAULT_PARAMETERS)
    if SALINE_WATER:
        pm.s_inf = 35
    d_p = np.logspace(-6, -1, 30)

    for j in range(len(points)):
        pm.set_parameter('alpha', points[j][0])
        pm.set_parameter('b0', points[j][1])
        pm.set_parameter('w0', points[j][2])
        pm.set_parameter('phi0', points[j][3])

        args = []
        for p in points[j]:
            p_exp = 0 if p==0 else int(np.floor(np.log10(p))) - 2
            args += [int(p * 10**(-p_exp)), p_exp]
        fpath = SAVE_FOLDER + "a{:d}e{:d}_b{:d}e{:d}_w{:d}e{:d}_p{:d}e{:d}.npz".format(*args)
        b = np.zeros((len(d_p), len(pm.z)-1))
        w = np.zeros((len(d_p), len(pm.z)-1))
        phi = np.zeros((len(d_p), len(pm.z)-1))
        T = np.zeros((len(d_p), len(pm.z)-1))
        S = np.zeros((len(d_p), len(pm.z)-1))
        errors = np.zeros(len(d_p))
        drdt = np.zeros(len(d_p))
        for i in range(len(d_p)):
            pm.d_p = d_p[i]
            try:
                drdt[i] = pm.converge_melt_rate()
                b[i, :], w[i, :], phi[i, :], T[i, :], S[i, :] = pm.compute_quantities()
            except ArithmeticError:
                b[i, :], w[i, :], phi[i, :], T[i, :], S[i, :] = np.nan, np.nan, np.nan, np.nan, np.nan
                errors[i] = PHI_ERROR
                drdt[i] = np.nan
            except ConvergenceError:
                b[i, :], w[i, :], phi[i, :], T[i, :], S[i, :] = np.nan, np.nan, np.nan, np.nan, np.nan
                errors[i] = CONV_ERROR
                drdt[i] = np.nan
        np.savez(fpath, b=b, w=w, phi=phi, T=T, S=S, dp=d_p, errors=errors, drdt=drdt)


def compute_4d_melt_rate(pickled=True, save_path='data/model_4d.pkl'):
    if not pickled:
        pm = PlumeModel(**DEFAULT_PARAMETERS)
        pm.constant_ws = False

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

        with open(save_path, 'wb') as f:
            pickle.dump([drdt, d_p, points], f)
    else:
        with open(save_path, 'rb') as f:
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
    cmap = cmo.matter_r

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K = list(range(0, len(w0_arr)))[-5:]

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
        w0_val = w0 * 10**(-w0_exp)
        axes[ki, -1].set_ylabel('$w_0 = {:.0f}'.format(w0_val)+'\cdot 10^{'+"{:d}".format(w0_exp)+'}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
        # axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
        #                         fontsize=12, rotation=270, labelpad=20)
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
    with open('data/model_4d_var_ws.pkl', 'rb') as f:
        drdt, d_p, points = pickle.load(f)

    # fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    # points = [fname_to_point(fn) for fn in fnames]

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

        # errors = np.load(fnames[n])['errors']
        # stability[i, j, k, l] = np.sum(errors != NO_ERROR) / len(errors)
        stability[i, j, k, l] = np.mean(np.isnan(drdt[n, :]))
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
    cmap = plt.get_cmap('RdYlGn_r')

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(100*(stability[:, :, K[ki], L[li]])), extent=extent, vmin=0, vmax=100, cmap=cmap)
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


def model_experiment_difference():
    # Experiments
    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    exp_dp = data_means.index.values * 1e-6  # particle diameter in m
    exp_drdt = data_means['drdt'].values  # melt rate in m/s

    # Model
    with open('data/model_4d.pkl', 'rb') as f:
        drdt, d_p, points = pickle.load(f)

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))

    rms = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    indices = np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)), dtype=np.int32)
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        err, cnt = 0, 0
        for exp_i in range(len(exp_dp)):
            ind = np.argmin(np.abs(d_p - exp_dp[exp_i]))
            err += (drdt[n, ind] - exp_drdt[exp_i]) ** 2
            cnt += 1
        rms[i, j, k, l] = np.sqrt(err / cnt)
        indices[i, j, k, l] = n

    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = plt.get_cmap('coolwarm')

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K = list(range(0, len(w0_arr)))[-5:]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    X, Y = np.meshgrid(np.linspace(extent[0], extent[1], rms.shape[0]),
                       np.linspace(extent[2], extent[3], rms.shape[1])[::-1])

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(rms[:, :, K[ki], L[li]]) * 1e3, extent=extent, vmin=0, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
            # axes[ki, li].contour(X, Y, np.flipud(rms[:, :, K[ki], L[li]]*1e3), colors='k', levels=[0.02])
    xt = [-6, -4, -2]  #list(range(int(extent[0]), int(extent[1]+1)))
    yt = [-4, -2, 0]  #list(range(int(extent[2]), int(extent[3]+1)))
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        # axes[ki, 0].set_ylabel(r"$\alpha$", fontsize=12)
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        w0_val = w0 * 10**(-w0_exp)
        axes[ki, -1].set_ylabel('$w_0 = {:.0f}'.format(w0_val)+'\cdot 10^{'+"{:d}".format(w0_exp)+'}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
        # axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
        #                         fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        # axes[-1, li].set_xlabel("$b_0$ (m)", fontsize=12)
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("RMS of $\dot{R}$ (mm/s)\n", fontsize=12)
    # cbt = [-6.5, -6.25] + list(range(-6, -1))
    # cbtl = ["$10^{"+"{:.0f}".format(val)+"}$" for val in cbt]
    # cbtl[0] = ""
    # cbtl[1] = "no maximum"
    # cb.set_ticks(cbt, labels=cbtl)
    # cb.ax.tick_params(labelsize=12)
    # cb.ax.set_facecolor('0.8')

    plt.show()



def average_phi(particle_size=None):
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    mpb = np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))
        d_p = np.load(fnames[n])['dp']

        phi = np.load(fnames[n])['phi']
        if particle_size is None:
            phi_bar = [np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z))) for di in range(phi.shape[0])]
            mpb[i, j, k, l] = np.nanmax(phi_bar)
        else:
            di = np.argmin(np.abs(d_p - particle_size))
            mpb[i, j, k, l] = np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z)))

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 1  # Show every delta-th slice in k and l
    cmap = cmo.turbid

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]
    # L = list(range(0, len(phi0_arr)))[:5]
    # K = list(range(0, len(w0_arr)))[-5:]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9], sharex=True, sharey=True)
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    X, Y = np.meshgrid(np.linspace(extent[0], extent[1], mpb.shape[0]),
                       np.linspace(extent[2], extent[3], mpb.shape[1])[::-1])

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(mpb[:, :, K[ki], L[li]]), extent=extent, vmin=0, vmax=0.6, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-8, -1])
            axes[ki, li].contour(X, Y, np.flipud(mpb[:, :, K[ki], L[li]]), colors='r', levels=[0.6])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        w0_val = w0 * 10**(-w0_exp)
        axes[ki, -1].set_ylabel('$w_0 = '+'{:.0f}'.format(w0_val)+' \cdot 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()

    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07, extend='max')
    if particle_size is None:
        cb.ax.set_title("$\overline{\phi}_\mathrm{max}$\n", fontsize=12)
    else:
        psz_exp = int(np.floor(np.log10(particle_size)))
        unit = "$\mu$m" if psz_exp < -3 else "mm" if psz_exp == -3 else "cm" if psz_exp == -2 else "m"
        multiplier = 1e6 if psz_exp < -3 else 1e3 if psz_exp == -3 else 1e2 if psz_exp == -2 else 1
        fmat = "{:.0f} " if psz_exp < -3 else "{:.2f} " if psz_exp == -1 else "{:.1f} "
        psz_str = fmat.format(particle_size * multiplier) + unit
        cb.ax.set_title("$\overline{\phi}$("+psz_str+")\n", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    cb.ax.plot([0, 1], [0.6, 0.6], '-r', lw=5)

    plt.show()


def is_there_a_maximum_in_phi_bar():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    max_dp = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        phi = np.load(fnames[n])['phi']
        phi_bar = np.array([np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z))) for di in range(phi.shape[0])])
        d_p = np.load(fnames[n])['dp']

        nn_ind = np.where(~np.isnan(phi_bar))[0]
        if len(nn_ind) > 2:
            first, last, in_between = np.abs(phi_bar[nn_ind[0]]), np.abs(phi_bar[nn_ind[-1]]), np.abs(phi_bar[nn_ind[1:-1]])
            if np.any(in_between > first) and np.any(in_between > last):
                max_dp[i, j, k, l] = d_p[np.nanargmax(np.abs(phi_bar))]

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.matter_r

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(np.log10(max_dp[:, :, K[ki], L[li]])), extent=extent, vmin=-6, vmax=-2, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$d_p^*$ (m)\n", fontsize=12)
    cbt = [-6.5, -6.25] + list(range(-6, -1))
    cbtl = ["$10^{" + "{:.0f}".format(val) + "}$" for val in cbt]
    cbtl[0] = ""
    cbtl[1] = "no maximum"
    cb.set_ticks(cbt, labels=cbtl)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def breadth_gradient_at_top(sz):
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    dbdz = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        b = np.load(fnames[n])['b']
        dp = np.load(fnames[n])['dp']
        ind = np.argmin(np.abs(sz-dp))
        dbdz[i, j, k, l] = -(b[ind, :][1] - b[ind, :][0]) / (z[1] - z[0])

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.curl

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(dbdz[:, :, K[ki], L[li]]), extent=extent, cmap=cmap, vmin=-4, vmax=4)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$db/dz$\n", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()



def does_phi_bar_increase_with_dp():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    increase = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        phi = np.load(fnames[n])['phi']
        phi_bar = np.array([np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z))) for di in range(phi.shape[0])])

        nn_ind = np.where(~np.isnan(phi_bar))[0]
        if len(nn_ind) > 2:
            first, other = phi_bar[nn_ind[0]], phi_bar[nn_ind[1:]]
            if np.any(other > first):
                increase[i, j, k, l] = np.nanmax(other) - first

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.algae

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(increase[:, :, K[ki], L[li]]), extent=extent, vmin=0, vmax=1, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$\Delta \overline{\phi}$\n", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def phi_bar_inflection_with_dp():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    inf_dp = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        phi = np.load(fnames[n])['phi']
        d_p = np.load(fnames[n])['dp']
        phi_bar = np.array([np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z))) for di in range(phi.shape[0])])
        d_phi_bar = np.diff(phi_bar)
        if np.sum(~np.isnan(d_phi_bar)) > 0:
            if np.nanmax(d_phi_bar) <= 0:
                inf_dp[i, j, k, l] = np.nan
            else:
                inf_dp[i, j, k, l] = d_p[np.nanargmax(d_phi_bar)]

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.matter_r

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(np.log10(inf_dp[:, :, K[ki], L[li]])), extent=extent, vmin=-6, vmax=-2, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$d_p^\circ$ (m)\n", fontsize=12)
    cbt = [-6.5, -6.25] + list(range(-6, -1))
    cbtl = ["$10^{"+"{:.0f}".format(val)+"}$" for val in cbt]
    cbtl[0] = ""
    cbtl[1] = "no maximum"
    cb.set_ticks(cbt, labels=cbtl)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def does_b_only_decrease_with_z():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    mdb = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        b = np.load(fnames[n])['b']

        does_b_decrease = [np.all(b[di, :] <= b[di, 0]) for di in range(b.shape[0])]
        n_not_nan = np.sum(~np.isnan(does_b_decrease))
        mdb[i, j, k, l] = np.nan if n_not_nan == 0 else np.nansum(does_b_decrease) / n_not_nan

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.amp

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(100 * mdb[:, :, K[ki], L[li]]), extent=extent, vmin=0, vmax=100, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("%\n", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def phi_bar_melt_rate_correlation():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    pcc = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        phi = np.load(fnames[n])['phi']
        drdt = np.load(fnames[n])['drdt']
        phi_bar = np.array([np.sum(phi[di, :] * np.abs(np.diff(z))) / np.sum(np.abs(np.diff(z))) for di in range(phi.shape[0])])
        non_nan_ind = ~np.isnan(phi_bar * drdt)
        if np.sum(non_nan_ind) > 2:
            pcc[i, j, k, l] = pearsonr(phi_bar[non_nan_ind]/phi_bar[0], drdt[non_nan_ind]/drdt[0]).statistic

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.curl_r
    cmap = cmocean.tools.crop_by_percent(cmap, 25, which='both', N=None)

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(pcc[:, :, K[ki], L[li]]), extent=extent, vmin=-1, vmax=1, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("PCC\n", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def is_there_a_maximum_in_avg_momentum():
    fnames = sorted(glob(SAVE_FOLDER + '*.npz'))
    points = [fname_to_point(fn) for fn in fnames]

    alpha_arr = np.sort(np.unique([p[0] for p in points]))
    b0_arr = np.sort(np.unique([p[1] for p in points]))
    w0_arr = np.sort(np.unique([p[2] for p in points]))
    phi0_arr = np.sort(np.unique([p[3] for p in points]))
    z = -np.logspace(np.log10(DEFAULT_PARAMETERS['z0']), np.log10(DEFAULT_PARAMETERS['max_z']), DEFAULT_PARAMETERS['n_points'])

    max_dp = np.nan * np.zeros((len(alpha_arr), len(b0_arr), len(w0_arr), len(phi0_arr)))
    for n in range(len(points)):
        i = np.argmin(np.abs(points[n][0] - alpha_arr))
        j = np.argmin(np.abs(points[n][1] - b0_arr))
        k = np.argmin(np.abs(points[n][2] - w0_arr))
        l = np.argmin(np.abs(points[n][3] - phi0_arr))

        b = np.load(fnames[n])['b']
        w = np.load(fnames[n])['w']
        phi = np.load(fnames[n])['phi']
        d_p = np.load(fnames[n])['dp']
        avg_dFmdz = np.zeros(len(d_p))
        for dpi in range(len(d_p)):
            ws = settling_velocity(d_p[dpi], 273.15, 0, 2500)
            Fm = b[dpi, :] * w[dpi, :]**2 * (1-phi[dpi, :]) + b[dpi, :] * (w[dpi, :] + ws)**2 * phi[dpi, :] * 2.5
            avg_dFmdz[dpi] = (Fm[-1] - Fm[0]) / np.abs(z[-1])

        nn_ind = np.where(~np.isnan(avg_dFmdz))[0]
        if len(nn_ind) > 2:
            first, last, in_between = np.abs(avg_dFmdz[nn_ind[0]]), np.abs(avg_dFmdz[nn_ind[-1]]), np.abs(avg_dFmdz[nn_ind[1:-1]])
            if np.any(in_between > first) and np.any(in_between > last):
                max_dp[i, j, k, l] = d_p[np.nanargmax(np.abs(avg_dFmdz))]

    # # Single slice
    # k, l = 0, 1
    # plt.figure()
    # extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))
    # plt.imshow(np.flipud(phi_bar[:, :, k, l]), extent=extent)
    # xt = list(range(int(extent[0]), int(extent[1]+1)))
    # plt.xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
    # yt = list(range(int(extent[2]), int(extent[3]+1)))
    # plt.yticks(yt, ["$10^{"+"{:d}".format(val) + "}$" for val in yt])
    # plt.tick_params(labelsize=12)
    # plt.xlabel('$b_0$ (m)', fontsize=14)
    # plt.ylabel(r'$\alpha$', fontsize=14)
    # plt.title("$w_0 = {:.1e}$ m/s  |  $\phi_0 = {:.2f}$".format(w0_arr[k], phi0_arr[l]))


    # All slices
    delta = 2  # Show every delta-th slice in k and l
    cmap = cmo.matter_r

    K, L = list(range(0, len(w0_arr), delta)), list(range(0, len(phi0_arr), delta))
    # K, L = list(range(0, len(w0_arr)))[-5:], list(range(0, len(phi0_arr)))[:5]

    fig, axes = plt.subplots(len(K), len(L), figsize=[14, 9])
    extent = (np.log10(b0_arr[0]), np.log10(b0_arr[-1]), np.log10(alpha_arr[0]), np.log10(alpha_arr[-1]))

    for ki in range(len(K)):
        for li in range(len(L)):
            im = axes[ki, li].imshow(np.flipud(np.log10(max_dp[:, :, K[ki], L[li]])), extent=extent, vmin=-6, vmax=-2, cmap=cmap)
            axes[ki, li].tick_params(left=li==0, labelleft=li==0, bottom=ki==len(K)-1, labelbottom=ki==len(K)-1, labelsize=12)
            axes[ki, li].set_facecolor('0.8')
            axes[ki, li].set_xlim([-7, -1])
    xt = [-6, -4, -2]
    yt = [-4, -2, 0]
    for ki in range(len(K)):
        axes[ki, 0].set_yticks(yt, ["$10^{" + "{:d}".format(val) + "}$" for val in yt])
        axes[ki, -1].yaxis.set_label_position("right")
        w0 = w0_arr[K[ki]]
        w0_exp = int(np.floor(np.log10(w0)))
        axes[ki, -1].set_ylabel('$w_0 = 10^{' + "{:d}".format(w0_exp) + '}$ m/s',
                                fontsize=12, rotation=270, labelpad=20)
    for li in range(len(L)):
        axes[-1, li].set_xticks(xt, ["$10^{"+"{:d}".format(val) + "}$" for val in xt])
        axes[0, li].set_title("$\phi_0 = {:.2f}$".format(phi0_arr[L[li]]), fontsize=12)
    fig.supxlabel('$b_0$ (m)', fontsize=16)
    fig.supylabel(r'$\alpha$', fontsize=16)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.07)
    cb.ax.set_title("$d_p^*$ (m)\n", fontsize=12)
    cbt = [-6.5, -6.25] + list(range(-6, -1))
    cbtl = ["$10^{" + "{:.0f}".format(val) + "}$" for val in cbt]
    cbtl[0] = ""
    cbtl[1] = "no maximum"
    cb.set_ticks(cbt, labels=cbtl)
    cb.ax.tick_params(labelsize=12)
    cb.ax.set_facecolor('0.8')

    plt.show()


def fname_to_point(fn):
    name = fn.split('/')[-1][:-4]
    values = [[int(v) for v in n[1:].split('e')] for n in name.split('_')]
    point = [v1 * 10**v2 for v1, v2 in values]
    return point


if __name__ == '__main__':
    main()


