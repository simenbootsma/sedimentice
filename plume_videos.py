import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
from copy import deepcopy
import time
import pickle
from scipy.signal import fftconvolve, correlate2d, convolve2d
from scipy.integrate import odeint
from typing import Union
import cmocean.cm as cmo
from scipy.stats import linregress

import pandas as pd

from PlumeModel import settling_velocity, rho_water, settling_velocity_at_z

matplotlib.use('Qt5Agg')

ccal = {'1mm': 93.0, '3mm': 109.5, '375um': 103.0, '200um': 103.0, '4mm': 139.4, '2mm': 139.4, '90um_old': 139.4,
        '4mm_end': 139.4, '8mm': 115.0, '875um': 115.0, '90um': 127.8, '55um': 117.3}  # calibration constant in um/px, 2*sigma uncertainty order 1 um/px
bottom = {'1mm': 1600, '3mm': 1680, '375um': 1520, '200um': 1750, '4mm': 1400, '2mm': 1450, '90um_old': 1450, '4mm_end': 1550,
          '8mm': 1330, '875um': 1450, '55um': 1300, '90um': 1300}
top = {'375um': 300, '200um': 250, '1mm': 250, '3mm': 300, '4mm': 400, '2mm': 400, '90um_old': 600, '4mm_end': 1150,
       '8mm': 200, '875um': 200, '55um': 100, '90um': 200}
xmid = {'1mm': 550, '2mm': 520, '3mm': 570, '4mm': 580, '375um': 560, '200um': 500, '90um_old': 525, '8mm': 530,
        '875um': 575, '55um': 560, '90um': 520}
top_ice = {'1mm': 450, '2mm': 560, '3mm': 520, '4mm': 550, '375um': 420, '200um': 469, '90um_old': 740, '8mm': 330,
           '875um': 380, '55um': 450, '90um': 380}
folder = '/Users/simenbootsma/Documents/PhD/Work/SedimentIce/plumeVideos/'


def main():
    # breadth_profiles()
    # velocity_profiles()
    # phi_profiles()
    # particle_track_curvature()
    particle_velocity()
    # plot_ablation_rates()

    # velocity_from_ablation_rate_test()
    # vel_profile_settling_comparison()

    sz = '90um'
    # show_frame(sz, 0)

    # # Contours
    # extract_contours(sz, debug=False)
    # remove_holder_from_contours(sz)
    # animate_contours(sz)
    # compute_width_and_height(sz, debug=False)
    # plot_width_and_height_over_time(sz)

    # # Breadth
    # compute_breadth_from_tracks(sz, debug=True)
    # compute_breadth_heatmap(sz, debug=False)
    # show_breadth(sz)

    # # Velocity - correlation
    # piv_sides(sz, debug=False)
    # show_piv_test(sz)

    # # Velocity - tracking
    # cloud = generate_particle_cloud(sz, debug=False)
    # tracks = tracks_from_particle_cloud(cloud, debug=False)
    # save_tracks_as_csv(sz, cloud, tracks)
    # analyse_tracks(sz)
    # add_tracks_to_video(sz)

    # # Ablation rate profile
    # ablation_rate_profiles(sz)


def velocity_from_ablation_rate_test():
    with open('plumeVideoAblationRateProfiles.pkl', 'rb') as f:
        abr_dct = pickle.load(f)

    sizes = ['90um', '200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    cmap = cmo.dense
    fig, ax = plt.subplots(1, len(sizes), figsize=(16, 8))
    with open('velocity_profiles.pkl', 'rb') as f:
        velocity_dct = pickle.load(f)

    for i, sz in enumerate(sizes):
        dp = float(sz[:-2]) * ({'mm': 1e-3, 'um': 1e-6}[sz[-2:]])
        label = "{:.1f} mm".format(dp * 1e3)
        color = cmap((sizes.index(sz)+2)/(len(sizes)+2))
        w, wz, we = velocity_dct[sz]

        if sz == '90um':
            w[wz < -.085] = np.nan

        z, mr_l, _, mr_r, _ = abr_dct[sz]
        mr = (mr_l + mr_r) / 2 * 1e-6
        # mr = running_average(mr, n=5)
        z = (z[0] - z) * 1e-6

        if sz == '90um':
            mr[z < -0.085] = np.nan

        gamma = 1e-6 / (0.332**2 * 7**(2/3)) * ((0.4 * 910 * 334e3)/(2.2 * 20))**2
        w_mr = gamma * mr**2 * z
        ws = np.array([-settling_velocity_at_z(dp, 293.15, 0, 2500, z_val) for z_val in z])

        fc = [c for c in color[:3]] + [0.4]
        ax[i].fill_betweenx(wz, w-we, w+we, color=fc)
        ax[i].plot(w, wz, color=color, label=label, lw=2)
        ax[i].plot(-w_mr, z, '--k')
        ax[i].plot(ws, z, ':k')
        ax[i].plot(ws-w_mr, z, '-k', lw=2)
        ax[i].set_title(label)

    for i, a in enumerate(ax):
        a.set_ylim([-0.1, 0])
        a.tick_params(labelsize=12, right=i < len(ax)-1, labelleft=(i==0))
        a.set_xlim([0, None])
        a.set_xlabel('$w$ (m/s)', fontsize=16)
        # a.set_facecolor('0.9')
    ax[0].set_ylabel('$z$ (m)', fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_width_and_height_over_time(sz):
    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        heights, widths, _, _ = pickle.load(f)[sz]
    heights *= ccal[sz] * 1e-4
    widths *= ccal[sz] * 1e-4

    t = np.arange(len(widths)) / 120  # s
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    ax[0].scatter(t, widths, 10, [[0, 0.2, 0.8, 0.1]])
    ax[0].plot(t, running_average(widths, n=120), color='C1')
    ax[1].scatter(t, heights, 10, [[0, 0.2, 0.8, 0.1]])
    ax[1].plot(t, running_average(heights, n=120), color='C1')

    I = ~np.isnan(widths)
    p = np.polyfit(t[I], widths[I], 1)
    ax[0].text(0.95, 0.9, "$w(t) = {:.3f}t + {:.3f}$".format(*p), ha='right', va='top', transform=ax[0].transAxes, fontsize=14)

    ax[0].set_ylabel('$w(t)$ (cm)', fontsize=16)
    ax[0].tick_params(labelsize=12)
    ax[1].set_ylabel('$h(t)$ (cm)', fontsize=16)
    ax[1].set_xlabel('$t$ (s)', fontsize=16)
    ax[1].tick_params(labelsize=12, top=True)

    ax[0].set_title(sz)
    plt.show()


def plot_ablation_rates():
    sizes = ['55um', '90um', '200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        dct = pickle.load(f)

    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    exp_dp = data_means.index.values * 1e-6  # particle diameter in m
    exp_drdt = data_means['drdt'].values  # melt rate in m/s

    dp = np.array([float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]] for sz in sizes])
    dRdt = np.zeros(len(sizes))
    for i, sz in enumerate(sizes):
        w = dct[sz][1] * ccal[sz] * 1e-3
        t = np.arange(len(w)) / 120  # s
        I = ~np.isnan(w)
        p = np.polyfit(t[I], w[I], 1)
        dRdt[i] = p[0] / 2

    plt.figure()
    plt.semilogx(exp_dp, -exp_drdt * 1e3, 'o')
    plt.semilogx(dp, -dRdt, 's')
    plt.ylabel('$-\dot{R}$ (mm/s)', fontsize=16)
    plt.tick_params(labelsize=12)
    plt.xlabel('$d_p$ (m)', fontsize=16)
    plt.tight_layout()
    plt.ylim([0, None])

    plt.show()


def compute_width_and_height(sz, debug=False):
    with open('plumeVideoContours_with_holder.pkl', 'rb') as f:
        dct = pickle.load(f)

    contours = dct[sz]
    use_low_threshold = False
    if sz + '_low' in dct:
        contours_low = dct[sz + '_low']
        use_low_threshold = True

    widths, heights = np.zeros(len(contours)), np.zeros(len(contours))
    y_top, y_bot = np.zeros(len(contours)), np.zeros(len(contours))
    for n in range(len(contours)):
        c = contours[n]
        c = c[c[:, 0] > top[sz], :]
        c = c[c[:, 0] < bottom[sz], :]
        I = c[:, 0] < top[sz] + 20
        holder_width = np.mean(np.abs(c[I, 1] - np.mean(c[I, 1])))
        y_ice = top[sz] + 10
        while y_ice < bottom[sz]:
            I = (c[:, 0] > (y_ice - 10)) & (c[:, 0] <= (y_ice + 10))
            width = np.mean(np.abs(c[I, 1] - np.mean(c[I, 1])))
            if width > 1.1 * holder_width:
                break
            y_ice += 1

        I = c[:, 0] < y_ice
        J = c[:, 0] >= y_ice
        if sum(J) < 100:
            widths[n] = np.nan
            heights[n] = np.nan
            y_top[n] = np.nan
            y_bot[n] = np.nan
            continue

        y2 = np.sort(c[J, 0])[sum(J)//50]
        y98 = np.sort(c[J, 0])[::-1][sum(J) // 50]
        x10 = np.sort(c[J, 1])[sum(J)//5]
        x90 = np.sort(c[J, 1])[::-1][sum(J) // 5]

        if use_low_threshold:
            cl = contours_low[n]
            Jl = cl[:, 0] >= y_ice
            y98l = np.sort(cl[Jl, 0])[::-1][sum(Jl) // 50]
            y98 = min(y98, y98l)

        widths[n] = x90 - x10
        heights[n] = y98 - y2
        y_top[n] = y2
        y_bot[n] = y98

        if debug:
            plt.plot(c[I, 1], c[I, 0], '-k', label='holder')
            plt.plot(c[J, 1], c[J, 0], '-', label='ice')
            if use_low_threshold:
                plt.plot(cl[Jl, 1], cl[Jl, 0], '-', label='ice (low thresh)')

            plt.plot([x10, x10], [y2, y98], color='C2')
            plt.plot([x90, x90], [y2, y98], color='C2')
            plt.plot([x10, x90], [y2, y2], color='C2')
            plt.plot([x10, x90], [y98, y98], color='C2')

            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title("dp = {:s}  |  n = {:d}".format(sz, n))

            plt.show()
        print("\r[compute_width_and_height({:s})] {:.1f}%".format(sz, (n+1)/len(contours) * 100), end='')

    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        hw_dct = pickle.load(f)

    hw_dct[sz] = (heights, widths, y_top, y_bot)

    with open('plumeVideoHeightsWidths.pkl', 'wb') as f:
        pickle.dump(hw_dct, f)

    print(" \033[42mdone\033[0m")


def ablation_rate_profiles(sz, skip_n_frames=10):
    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        heights, widths, y_top, y_bot = pickle.load(f)[sz]

    with open('plumeVideoContours.pkl', 'rb') as f:
        contours = pickle.load(f)[sz]

    # ind = np.argwhere(widths * ccal[sz] * 1e-6 < 0.05)
    ind = np.argwhere(heights/heights[0] < 0.9)
    N = len(contours) if len(ind) == 0 else ind[0][0]

    contours = [c * ccal[sz] for c in contours[:N]]
    t = np.arange(len(contours)) / 120

    contours = contours[::skip_n_frames]
    t = t[::skip_n_frames]

    # plt.plot(contours[0][:, 0], contours[0][:, 1])
    # plt.gca().set_aspect('equal')
    # plt.show()

    mrp = compute_melt_rate_profile(contours, t, bin_size=2000)
    z, mr_l, mr_l_se, mr_r, mr_r_se = mrp

    with open('plumeVideoAblationRateProfiles.pkl', 'rb') as f:
        dct = pickle.load(f)

    dct[sz] = z, mr_l, mr_l_se, mr_r, mr_r_se

    with open('plumeVideoAblationRateProfiles.pkl', 'wb') as f:
        pickle.dump(dct, f)

    # plt.plot(mr_l, z)
    # plt.plot(mr_r, z)
    # plt.show()


def compute_melt_rate_profile(contours, times, bin_size=1000, tmax=None, ignore_nan=True, debug=False):
    """ Contours and Bin size in um, times in seconds"""

    max_ind = len(contours) if tmax is None else np.argmin(np.abs(times - tmax))
    times = times[:max_ind]

    bin_edges = np.arange(np.min(contours[0][:, 1]), np.max(contours[0][:, 1]), bin_size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    xmean = np.mean(contours[0][:, 0])
    x_bins_l = np.zeros((len(bin_centers), max_ind))
    x_bins_r = np.zeros((len(bin_centers), max_ind))
    for i in range(max_ind):
        left = contours[i][contours[i][:, 0] < xmean]
        right = contours[i][contours[i][:, 0] >= xmean]
        dig_l = np.digitize(left[:, 1], bin_edges)
        dig_r = np.digitize(right[:, 1], bin_edges)
        for j in range(len(bin_centers)):
            if ignore_nan:
                x_bins_l[j, i] = abs(np.nanmean(left[dig_l == j+1, 0]) - xmean)
                x_bins_r[j, i] = abs(np.nanmean(right[dig_r == j + 1, 0]) - xmean)
            else:
                x_bins_l[j, i] = abs(np.mean(left[dig_l == j + 1, 0]) - xmean)
                x_bins_r[j, i] = abs(np.mean(right[dig_r == j + 1, 0]) - xmean)
        print("\r[\033[32mcompute_melt_rate_profile\033[0m] {:.1f}%".format((i+1)/max_ind*100), end='')

    fits_l = [linregress(times, x_bins_l[i, :]) for i in range(x_bins_l.shape[0])]
    fits_r = [linregress(times, x_bins_r[i, :]) for i in range(x_bins_r.shape[0])]

    # dr/dt in um/s
    mr_l = np.array([val.slope for val in fits_l])
    mr_r = np.array([val.slope for val in fits_r])
    mr_l_se = np.array([val.stderr for val in fits_l])
    mr_r_se = np.array([val.stderr for val in fits_r])
    return bin_centers, mr_l, mr_l_se, mr_r, mr_r_se


def particle_velocity():
    ptv_sizes = ['875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    piv_sizes = ['55um', '90um', '200um', '375um']
    diameters = np.logspace(-5, -1, 30)
    wt = [settling_velocity(d, 293.15, 0, 2500) for d in diameters]
    # wa = [np.abs(settling_velocity_above_z(d, 293.15, 0, 2500, -0.1)) for d in diameters]

    ws_avg, ws_std = np.zeros(len(diameters)), np.zeros(len(diameters))
    for i in range(len(diameters)):
        _, w = settling_velocity_above_z(diameters[i], 293.15, 0, 2500, -0.1, return_full=True)
        ws_avg[i] = np.abs(np.nanmean(w))
        ws_std[i] = np.nanstd(w)

    ws_exp = {}
    # for sz in ptv_sizes:
    #     ws_exp[sz] = ptv_velocity(sz)
    # for sz in piv_sizes:
    #     ws_exp[sz] = piv_velocity(sz)
    with open('velocity_profiles.pkl', 'rb') as f:
        dct = pickle.load(f)

    for sz in ptv_sizes + piv_sizes:
        w, wz, we = dct[sz]
        I = wz >= -0.1
        ws_exp[sz] = (np.nanmean(w[I]), np.nanmean(we[I]))

    plt.figure()
    plt.plot(diameters, wt, '-k', lw=2, label=r'$w_s(z\rightarrow\infty)$')
    # plt.plot(diameters, wa, '--k', lw=2, label=r'$\langle w_s\rangle_z$')
    plt.fill_between(diameters, ws_avg - ws_std, ws_avg + ws_std, color=(0, 0, 0, 0.15))
    plt.plot(diameters, ws_avg, '--k', lw=2, label=r'$\langle w_s\rangle_z$')
    for sz in ptv_sizes:
        mu, sigma = ws_exp[sz]
        dp = float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]]
        plt.errorbar(dp, mu, yerr=sigma, capsize=3, fmt='o', color=(0, 0.6, 0.8),
                     label='tracking experiments' if ptv_sizes.index(sz)==0 else None)
    for sz in piv_sizes:
        mu, sigma = ws_exp[sz]
        dp = float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]]
        plt.errorbar(dp, mu, yerr=sigma, capsize=3, fmt='s', color=(0, 0.6, 0.8), markerfacecolor='w',
                     label='correlation experiments' if piv_sizes.index(sz)==0 else None)
    plt.xscale('log')
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=12)
    plt.ylabel('$w$ (m/s)', fontsize=16)
    plt.xlabel('$d_p$ (m)', fontsize=16)
    plt.xlim([5e-5, 2.1e-2])
    plt.ylim([0, 0.6])
    plt.tight_layout()

    # Logarithmic y-axis
    plt.figure()
    plt.plot(diameters, wt, '-k', lw=2, label=r'$w_s(z\rightarrow\infty)$')
    # plt.plot(diameters, wa, '--k', lw=2, label=r'$\langle w_s\rangle_z$')
    plt.fill_between(diameters, ws_avg - ws_std, ws_avg + ws_std, color=(0, 0, 0, 0.15))
    plt.plot(diameters, ws_avg, '--k', lw=2, label=r'$\langle w_s\rangle_z$')
    for sz in ptv_sizes:
        mu, sigma = ws_exp[sz]
        dp = float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]]
        plt.errorbar(dp, mu, yerr=sigma, capsize=3, fmt='o', color=(0, 0.6, 0.8),
                     label='tracking experiments' if ptv_sizes.index(sz)==0 else None)
    for sz in piv_sizes:
        mu, sigma = ws_exp[sz]
        dp = float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]]
        plt.errorbar(dp, mu, yerr=sigma, capsize=3, fmt='s', color=(0, 0.6, 0.8), markerfacecolor='w',
                     label='correlation experiments' if piv_sizes.index(sz)==0 else None)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=12)
    plt.ylabel('$w$ (m/s)', fontsize=16)
    plt.xlabel('$d_p$ (m)', fontsize=16)
    plt.xlim([5e-5, 2.1e-2])
    plt.ylim([1e-2, 1])
    plt.tight_layout()
    plt.show()


def settling_velocity_above_z(d_p, temp, sal, rho_p, z, return_full=False):
    z = np.abs(z)
    z0 = np.arange(-z, 0, z/100)
    v_avg = np.zeros(len(z0))
    for i in range(len(z0)):
        v_avg[i] = settling_velocity_at_z(d_p, temp, sal, rho_p, z=z0[i])
    v_avg[np.isinf(v_avg)] = np.nan

    # plt.plot(z0, v_avg)
    # plt.title("dp = {:.1e} m".format(d_p))
    # plt.show()

    if return_full:
        return z0, v_avg

    dz0 = np.diff(z0)
    va = (v_avg[1:] + v_avg[:-1])/2
    return 1/z * np.nansum(va * dz0)


def particle_track_curvature():
    sizes = ['1mm', '2mm', '3mm', '4mm'][:1]
    dt = 1/120  # time interval between frames
    cmap = cmo.dense
    colors = {'1mm': cmap(0.2), '2mm': cmap(0.4), '3mm': cmap(0.6), '4mm': cmap(0.8)}

    plt.figure()
    plt.semilogy([0, 0], [1e-4, 1], '-k', lw=1)
    for sz in sizes:

        tracks = get_tracks_xyn(sz)

        curvature = []
        for trk in tracks:
            if len(trk) > 2:
                for j in range(1, len(trk)-1):
                    dxdt = (trk[j + 1, 0] - trk[j - 1, 0]) / (2 * dt)
                    dydt = (trk[j + 1, 1] - trk[j - 1, 1]) / (2 * dt)
                    d2xdt2 = (trk[j + 1, 0] - trk[j, 0] + trk[j - 1, 0]) / dt ** 2
                    d2ydt2 = (trk[j + 1, 1] - trk[j, 1] + trk[j - 1, 1]) / dt ** 2
                    k = (dxdt*d2ydt2 - dydt*d2xdt2) / (dxdt**2 + dydt**2)**(3/2)
                    sgn = -1 if trk[j, 0] < xmid[sz] else 1
                    curvature.append(sgn * k)

        mu, sigma = np.mean(curvature), np.std(curvature)
        bins = np.arange(mu-3*sigma, mu+3*sigma, sigma/20)
        h, b = np.histogram(curvature, bins, density=True)
        x = (b[1:] + b[:-1])/2
        plt.semilogy(x, h, '-', color=colors[sz], markersize=4, lw=2, label='$d_p$ = '+sz)
    plt.text(0.02, 0.98, 'away from ice', transform=plt.gca().transAxes, fontsize=16, color='0.3', va='top')
    plt.text(0.98, 0.98, 'toward ice', transform=plt.gca().transAxes, fontsize=16, color='0.3', va='top', ha='right')
    plt.ylim([1e-4, 1])
    plt.xlim([-10, 10])
    plt.tick_params(labelsize=12)
    plt.ylabel('PDF($\kappa$)  (m$^{-1}$)', fontsize=16)
    plt.xlabel('$\kappa$ (m$^{-1}$)', fontsize=16)
    # plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()


def velocity_profiles():
    piv_sizes = ['55um', '90um', '200um', '375um']
    ptv_sizes = ['875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    # ptv_sizes = ['4mm']
    sizes = piv_sizes + ptv_sizes
    bins = np.arange(-1000, 2000, 64)
    tf = 120  # time period to use in seconds (max. 180s)
    dt = 1/120  # 1 / fps
    nf = int(tf/dt)  # number of frames to use
    min_n_ptv = 10  # minimum number of data points in bins
    min_n_piv = 1  # minimum number of data points in bins
    cmap = cmo.dense
    colors = {k: cmap((sizes.index(k)+1)/(len(sizes) + 1)) for k in sizes}

    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        hw_dct = pickle.load(f)

    vy = {sz: [[] for _ in range(len(bins)-1)] for sz in sizes}

    # Compute tracking profiles
    for sz in ptv_sizes:
        _, _, yt, _ = hw_dct[sz]
        yt = running_average(yt, n=240)  # y-coordinate of top of ice block
        for trk in get_tracks_xyn(sz):
            for j in range(len(trk)-1):
                n = int((trk[j, 2] + trk[j+1, 2]) / 2)
                if n < nf:
                    y = (trk[j, 1] + trk[j+1, 1])/2 - yt[n]  # distance to top of the block
                    ind = np.argmin(np.abs(y-bins[y >= bins]))
                    vy[sz][ind].append((trk[j+1, 1] - trk[j, 1]) * ccal[sz] * 1e-6 / dt)
                    # inst_vel = np.sqrt((trk[j+1, 0] - trk[j, 0])**2 + (trk[j+1, 1] - trk[j, 1])**2) * ccal[sz] * 1e-6 / dt
                    # vy[sz][ind].append(inst_vel)

    # Compute correlation profiles
    for sz in piv_sizes:
        cutoff_vy = -10 if sz == '90um' else 0 if sz == '55um' else 1
        _, _, yt, _ = hw_dct[sz]
        yt = running_average(yt, n=240)  # y-coordinate of top of ice block

        with open('particle_correlations/piv_sides_{:s}_32_64.pkl'.format(sz), 'rb') as f:
            _, vectors = pickle.load(f)

        for v in vectors:
            n = int(v[4])
            y = v[1] - yt[n] + top[sz]
            ind = np.argmin(np.abs(y - bins[y >= bins]))
            if v[3] > cutoff_vy:
                vy[sz][ind].append(v[3] * ccal[sz] * 1e-6 / dt)

    # Show profiles
    plt.figure(figsize=(6, 10))
    dct = {}
    for sz in piv_sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        vy_mean = np.array([np.mean(arr) for arr in vy[sz] if len(arr) > min_n_piv])
        vy_std = np.array([np.std(arr) for arr in vy[sz] if len(arr) >= min_n_piv])
        y = (bins[1:] + bins[:-1])/2
        y = y[(cnt > min_n_piv)]
        y = -y * ccal[sz] * 1e-6

        fc = [c for c in colors[sz][:3]] + [0.2]
        plt.fill_betweenx(y, vy_mean-vy_std, vy_mean+vy_std, color=fc)
        plt.plot(vy_mean, y, color=colors[sz], lw=2)
        dct[sz] = (vy_mean, y, vy_std)
    for sz in ptv_sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        vy_mean = np.array([np.mean(arr) for arr in vy[sz] if len(arr) >= max(min_n_ptv, 0.01*np.max(cnt))])
        vy_std = np.array([np.std(arr) for arr in vy[sz] if len(arr) >= max(min_n_ptv, 0.01*np.max(cnt))])
        y = (bins[1:] + bins[:-1])/2
        y = y[(cnt >= min_n_ptv) & (cnt > 0.01 * np.max(cnt))]
        y = -y * ccal[sz] * 1e-6

        vy_mean = vy_mean[y < 0]
        vy_std = vy_std[y < 0]
        y = y[y < 0]

        fc = [c for c in colors[sz][:3]] + [0.2]
        plt.fill_betweenx(y, vy_mean-vy_std, vy_mean+vy_std, color=fc)
        plt.plot(vy_mean, y, color=colors[sz], lw=2)
        dct[sz] = (vy_mean, y, vy_std)
    plt.xlim([0, None])
    plt.ylim([-0.13, 0])
    plt.tick_params(labelsize=12)
    plt.xlabel("$w$ (m/s)", fontsize=16)
    plt.ylabel('$z$ (m)', fontsize=16)

    handles = []
    for sz in sizes:
        fc = [c for c in colors[sz][:3]] + [0.2]
        fl = plt.fill_betweenx([], [], [], color=fc)
        ln, = plt.plot([], [], color=colors[sz], lw=2)
        handles.append((fl, ln))
    plt.legend(handles, sizes, bbox_to_anchor=(1.03, 1.), loc='upper left', fontsize=14)
    plt.tight_layout()

    plt.figure()
    for sz in sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        y = (bins[1:] + bins[:-1])/2
        y = -y * ccal[sz] * 1e-6
        plt.plot(cnt/np.nanmax(cnt), y, '-o', color=colors[sz])

    with open('velocity_profiles.pkl', 'wb') as f:
        pickle.dump(dct, f)

    plt.show()


def particle_volume_fraction_profiles():
    piv_sizes = ['90um', '200um', '375um']
    ptv_sizes = ['875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    # ptv_sizes = ['4mm']
    sizes = piv_sizes + ptv_sizes
    bins = np.arange(-1000, 2000, 64)
    tf = 120  # time period to use in seconds (max. 180s)
    dt = 1/120  # 1 / fps
    nf = int(tf/dt)  # number of frames to use
    min_n = 10  # minimum number of data points in bins
    cmap = cmo.dense
    colors = {'875um': cmap(0.1), '1mm': cmap(0.2), '2mm': cmap(0.4), '3mm': cmap(0.6), '4mm': cmap(0.8), '8mm': cmap(0.9)}
    colors = {k: cmap((sizes.index(k)+1)/(len(sizes) + 1)) for k in sizes}

    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        hw_dct = pickle.load(f)

    vy = {sz: [[] for _ in range(len(bins)-1)] for sz in sizes}

    # Compute tracking profiles
    for sz in ptv_sizes:
        _, _, yt, _ = hw_dct[sz]
        yt = running_average(yt, n=240)  # y-coordinate of top of ice block
        for trk in get_tracks_xyn(sz):
            for j in range(len(trk)-1):
                n = int((trk[j, 2] + trk[j+1, 2]) / 2)
                if n < nf:
                    y = (trk[j, 1] + trk[j+1, 1])/2 - yt[n]
                    ind = np.argmin(np.abs(y-bins[y >= bins]))
                    vy[sz][ind].append((trk[j+1, 1] - trk[j, 1]) * ccal[sz] * 1e-6 / dt)

    # Compute correlation profiles
    cutoff_vy = 0.8

    for sz in piv_sizes:
        _, _, yt, _ = hw_dct[sz]
        yt = running_average(yt, n=240)  # y-coordinate of top of ice block

        with open('particle_correlations/piv_sides_{:s}_32_64.pkl'.format(sz), 'rb') as f:
            _, vectors = pickle.load(f)

        for v in vectors:
            n = int(v[4])
            y = v[1] - yt[n] + top[sz]
            ind = np.argmin(np.abs(y - bins[y >= bins]))
            if v[3] > cutoff_vy:
                vy[sz][ind].append(v[3] * ccal[sz] * 1e-6 / dt)

    # Show profiles
    plt.figure(figsize=(6, 10))
    for sz in piv_sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        vy_mean = np.array([np.mean(arr) for arr in vy[sz] if len(arr) > max(min_n, 0.01*np.max(cnt))])
        vy_std = np.array([np.std(arr) for arr in vy[sz] if len(arr) > max(min_n, 0.01*np.max(cnt))])
        y = (bins[1:] + bins[:-1])/2
        y = y[(cnt > min_n) & (cnt > 0.01 * np.max(cnt))]
        y = -y * ccal[sz] * 1e-6

        fc = [c for c in colors[sz][:3]] + [0.2]
        plt.fill_betweenx(y, vy_mean-vy_std, vy_mean+vy_std, color=fc)
        plt.plot(vy_mean, y, color=colors[sz], lw=2)
    for sz in ptv_sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        vy_mean = np.array([np.mean(arr) for arr in vy[sz] if len(arr) > max(min_n, 0.01*np.max(cnt))])
        vy_std = np.array([np.std(arr) for arr in vy[sz] if len(arr) > max(min_n, 0.01*np.max(cnt))])
        y = (bins[1:] + bins[:-1])/2
        y = y[(cnt > min_n) & (cnt > 0.01 * np.max(cnt))]
        y = -y * ccal[sz] * 1e-6

        vy_mean = vy_mean[y < 0]
        vy_std = vy_std[y < 0]
        y = y[y < 0]

        fc = [c for c in colors[sz][:3]] + [0.2]
        plt.fill_betweenx(y, vy_mean-vy_std, vy_mean+vy_std, color=fc)
        plt.plot(vy_mean, y, color=colors[sz], lw=2)
    plt.xlim([0, None])
    plt.ylim([-0.13, 0])
    plt.tick_params(labelsize=12)
    plt.xlabel("$w$ (m/s)", fontsize=16)
    plt.ylabel('$z$ (m)', fontsize=16)

    handles = []
    for sz in sizes:
        fc = [c for c in colors[sz][:3]] + [0.2]
        fl = plt.fill_betweenx([], [], [], color=fc)
        ln, = plt.plot([], [], color=colors[sz], lw=2)
        handles.append((fl, ln))
    plt.legend(handles, sizes, bbox_to_anchor=(1.03, 1.), loc='upper left', fontsize=14)
    plt.tight_layout()

    plt.figure()
    for sz in sizes:
        cnt = np.array([len(arr) for arr in vy[sz]])
        y = (bins[1:] + bins[:-1])/2
        y = -y * ccal[sz] * 1e-6
        plt.plot(cnt/np.nanmax(cnt), y, '-o', color=colors[sz])

    plt.show()



def show_piv_test(sz):
    cutoff_vy = -10 if sz == '90um' else 0 if sz == '55um' else 1

    with open('particle_correlations/piv_sides_{:s}_32_64.pkl'.format(sz), 'rb') as f:
        vel, vel_avg = pickle.load(f)

    fig, ax = plt.subplots(2, 1)
    vy = {}
    vx = {}
    t = {}
    vel = vel_avg
    for i in range(len(vel)):
        yi = int(vel[i][1])
        if yi in vy:
            if vel[i][3] > cutoff_vy:
                vx[yi].append(vel[i][2])
                vy[yi].append(vel[i][3])
                t[yi].append(vel[i][4])
        else:
            vy[yi] = []
            vx[yi] = []
            t[yi] = []

    cmap = plt.get_cmap('Reds')
    for yi in t:
        ax[0].scatter(t[yi], vx[yi], 20, cmap(yi/1500))
        ax[1].scatter(t[yi], vy[yi], 20, cmap(yi/1500))
    ax[0].legend()

    plt.figure()
    for yi in t:
        plt.scatter(vx[yi], vy[yi], 20, cmap(yi/1500))

    plt.figure()
    y_arr = [yi for yi in t]
    mv_arr = [np.mean(vy[yi]) for yi in t]
    sv_arr = [np.std(vy[yi]) for yi in t]
    n_arr = [len(vy[yi]) for yi in t]
    svn_arr = np.array(sv_arr) / np.sqrt(n_arr)
    y_m = (top_ice[sz] - top[sz] - np.array(y_arr)) * ccal[sz] * 1e-6
    fac = ccal[sz] * 1e-6 * 120
    plt.errorbar(np.array(mv_arr) * fac, y_m, xerr=np.array(sv_arr) * fac, fmt='o', capsize=3)
    plt.xlabel('w(z) (m/s)', fontsize=16)
    plt.ylabel('z (m)', fontsize=16)

    plt.figure()
    plt.plot(y_m, np.array(n_arr), '-o')
    plt.xlabel('distance from top')
    plt.ylabel('fraction above cutoff')
    # plt.ylim([0, 1])

    plt.figure()
    plt.boxplot([vy[yi] for yi in t], vert=True, positions=y_arr, widths=30, manage_ticks=False)
    plt.show()


def animate_contours(sz):
    with open('plumeVideoContours.pkl', 'rb') as f:
        dct = pickle.load(f)

    contours = dct[sz]

    fig, ax = plt.subplots()
    plt.ion()
    for n in range(0, len(contours), 50):
        c = contours[n]
        if sz in top:
            c = c[c[:, 0] > top[sz]]
        ax.clear()
        ax.plot(c[:, 0], -c[:, 1], '-k')

        if sz + '_low' in dct:
            cl = dct[sz + '_low'][n]
            if sz in top:
                cl = cl[cl[:, 0] > top[sz]]
            ax.plot(cl[:, 0], -cl[:, 1], '-b')

        ax.set_aspect('equal')
        ax.set_xlim([0, 1100])
        ax.set_ylim([-1700, 0])
        ax.set_title("dp = {:s}  |  n = {:d}".format(sz, n))
        plt.pause(0.01)
    plt.ioff()
    plt.show()


def extract_contours(sz, n=None, debug=False):
    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)
    bin_thresh = {'8mm': 180, '4mm': 180, '3mm': 180, '2mm': 180, '1mm': 180, '875um': 180, '375um': 100, '200um': 100,
                  '90um': 100, '55um': 100}[sz]
    low_thresh = {'200um': 50, '90um': 15, '55um': 15}

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    N = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    n0 = 0
    contours = []
    contours_low_thresh = []

    if n is not None:
        cap.set(cv.CAP_PROP_POS_FRAMES, n)
        n0 = n
        N = n + 1
    for n in range(n0, N):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        # gray[:, bottom[sz]:] = 255
        gray[:, :top[sz]] = 0
        mask = find_ice_mask(gray, mask_thresh=bin_thresh)
        edges = find_edges(mask, largest_only=True)
        contours.append(edges)

        if debug:
            plt.figure()
            plt.imshow(gray)
            plt.plot(edges[:, 0], edges[:, 1], '-r')
            plt.title('dp = {:s}  |  n = {:d}'.format(sz, n))
            plt.show()

        if sz in low_thresh:
            sp = (940, 520) if sz == '90um' else None
            mask = find_ice_mask(gray, mask_thresh=low_thresh[sz], seed_point=sp)
            edges = find_edges(mask, largest_only=True)
            contours_low_thresh.append(edges)

            if debug:
                plt.figure()
                plt.imshow(gray)
                plt.plot(edges[:, 0], edges[:, 1], '-r')
                plt.title('dp = {:s}  |  n = {:d}  low threshold'.format(sz, n))
                plt.show()
        print("\r[extract_contours({:s})] {:.1f}%".format(sz, (n+1)/N*100), end='')

    if not debug:
        with open('plumeVideoContours_with_holder.pkl', 'rb') as f:
            dct = pickle.load(f)

        dct[sz] = contours
        if len(contours_low_thresh) > 0:
            dct[sz + '_low'] = contours_low_thresh

        with open('plumeVideoContours_with_holder.pkl', 'wb') as f:
            pickle.dump(dct, f)
        print(" \033[42mdone\033[0m")


def remove_holder_from_contours(sz, debug=False):
    with open('plumeVideoContours_with_holder.pkl', 'rb') as f:
        contours = pickle.load(f)[sz]

    new_contours = []
    cnt = 0
    for c in contours:
        if debug:
            plt.plot(c[:, 1], c[:, 0], label='before')
        c = c[c[:, 0] > top[sz], :]
        c = c[c[:, 0] < bottom[sz], :]
        I = c[:, 0] < top[sz] + 20
        holder_width = np.mean(np.abs(c[I, 1] - np.mean(c[I, 1])))
        y_ice = top[sz] + 10
        while y_ice < bottom[sz]:
            I = (c[:, 0] > (y_ice - 10)) & (c[:, 0] <= (y_ice + 10))
            width = np.mean(np.abs(c[I, 1] - np.mean(c[I, 1])))
            if width > 1.1 * holder_width:
                break
            y_ice += 1

        c = c[c[:, 0] >= y_ice, :]
        if debug:
            plt.plot(c[:, 1], c[:, 0], label='after')
            plt.gca().set_aspect('equal')
            plt.legend()
            plt.show()
        c = np.fliplr(c)  # make first coordinate the x-direction
        new_contours.append(c)
        cnt += 1
        print("\r[remove_holder_from_contours({:s})] {:.1f}%".format(sz, cnt/len(contours)*100), end='')

    with open('plumeVideoContours.pkl', 'rb') as f:
        dct = pickle.load(f)

    dct[sz] = new_contours

    with open('plumeVideoContours.pkl', 'wb') as f:
        pickle.dump(dct, f)

    print(" \033[42mdone\033[0m")


def find_ice_mask(img, mask_thresh=40, seed_point=None):
    """ img: gray image """

    _, binry = cv.threshold(img, mask_thresh, 255, cv.THRESH_BINARY)
    binry = 255 - binry

    mask = np.zeros((binry.shape[0] + 2, binry.shape[1] + 2), dtype=np.uint8)
    if seed_point is None:
        seed_point = [[i, 0] for i in range(binry.shape[0]) if binry[i, 0] == 255][0]
    cv.floodFill(binry, mask, seed_point, (255,))
    mask = mask[1:-1, 1:-1]

    mask = cv.dilate(mask, kernel=np.ones((3, 3)), iterations=3)
    mask = cv.erode(mask, kernel=np.ones((3, 3)), iterations=3)
    return mask


def find_edges(img, largest_only=False, remove_outside=False):
    if largest_only:
        cont, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(cont) == 0:
            return None  # no contours found

        idx = np.argmax([len(c) for c in cont])  # index of largest contour
        # idx = np.argsort([len(c) for c in cont])[-2]  # index of second-largest contour
        edges = np.reshape(cont[idx], (cont[idx].shape[0], 2))
        return edges
    else:
        conts, hierarchy = cv.findContours(img.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        if len(conts) == 0:
            return None  # no contours found

        # stack together
        edges = np.array([0, 0])
        for c in conts:
            edges = np.vstack((edges, np.reshape(c, (c.shape[0], 2))))

        # remove box edges
        if remove_outside:
            edges = edges[edges[:, 0] > 0]
            edges = edges[edges[:, 0] < img.shape[1]-1]
            edges = edges[edges[:, 1] > 0]
            edges = edges[edges[:, 1] < img.shape[0]-1]
        return edges[1:, :]


def breadth_profiles():
    sizes = ['55um', '90um', '200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    cmap = cmo.dense
    # colors = {'90um': cmap(0.4), '200um': cmap(0.6), '375um': cmap(0.8)}
    # fig, ax = plt.subplots(1, 3, figsize=(8, 10), sharey=True, sharex=True)
    fig, ax = plt.subplots(figsize=(5, 8))
    fits = {}
    dct = {}
    for sz in sizes:
        bins = np.linspace(top_ice[sz], 2000, 100, dtype=np.int32)
        y = (bins[1:] + bins[:-1]) / 2

        with open('breadth_profiles/breadths_'+sz+'.pkl', 'rb') as f:
            breadth = pickle.load(f)

        avg_l = np.nanmean(breadth[0], axis=0) * ccal[sz] * 1e-3
        avg_r = np.nanmean(breadth[1], axis=0) * ccal[sz] * 1e-3
        std_l = np.nanstd(breadth[0], axis=0) * ccal[sz] * 1e-3
        std_r = np.nanstd(breadth[1], axis=0) * ccal[sz] * 1e-3

        z = (top_ice[sz] - y) * ccal[sz] * 1e-6

        if sz == '90um':
            avg_l[z < -0.085] = np.nan
            avg_r[z < -0.085] = np.nan
        if sz == '55um':
            avg_l[:] = np.nan
            avg_r[z < -0.09] = np.nan

        # fc = [c for c in colors[sz][:3]] + [0.2]
        # ax[0].fill_betweenx(z, avg_l-std_l, avg_l+std_l, color=fc)
        # ax[1].fill_betweenx(z, avg_r - std_r, avg_r + std_r, color=fc)

        color = cmap((sizes.index(sz)+1)/(len(sizes)+1))
        # ax[0].plot(avg_l, z, label=sz, color=color)
        # ax[1].plot(avg_r, z, label=sz, color=color)
        b = np.nanmean(np.vstack((avg_l, avg_r)), axis=0)
        ax.plot(b, z, color=color, label=sz.replace('u', '$\mu$'))
        I = ~np.isnan(b) & (z >= -0.1)
        p = np.polyfit(z[I], b[I], deg=1)
        ax.plot(p[0]*z + p[1], z, '-k', label=None)
        fits[sz] = p
        dct[sz] = (b * 1e-3, z)

        print("[{:s} left]: abs std = {:.1f} mm  |  rel std = {:.1f}%".format(sz, np.nanmean(std_l), np.nanmean(std_l/avg_l)*100))
        print("[{:s} right]: abs std = {:.1f} mm  |  rel std = {:.1f}%".format(sz, np.nanmean(std_r), np.nanmean(std_r/avg_r)*100))

    # ax[0].set_xlabel('$b$ (mm)', fontsize=16)
    # ax[1].set_xlabel('$b$ (mm)', fontsize=16)
    # ax[0].set_ylabel('$z$ (m)', fontsize=16)
    # ax[0].set_xlim([0, 20])
    # ax[0].set_ylim([-0.1, 0])
    # ax[1].set_xlim([0, 20])
    # ax[1].set_ylim([-0.1, 0])
    # ax[0].tick_params(labelsize=12, right=True)
    # ax[1].tick_params(labelsize=12, left=False, labelleft=False)
    # ax[1].legend(fontsize=12)

    # ax.plot([15, 16.5], [-0.04, -0.04], '-k', lw=1.5)
    # ax.plot([15, 15], [-0.0405, -0.0395], '-k', lw=1.5)
    # ax.plot([16.5, 16.5], [-0.0405, -0.0395], '-k', lw=1.5)
    # ax.text(15.75, -0.0395, '1$\sigma$', fontsize=14, ha='center', va='bottom')

    ax.set_xlabel('$b$ (mm)', fontsize=16)
    ax.set_ylabel('$z$ (m)', fontsize=16)
    ax.set_xlim([0, 20])
    ax.set_ylim([-0.1, 0])
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    dp = [float(sz[:-2]) * ({'mm': 1e-3, 'um': 1e-6}[sz[-2:]]) for sz in sizes]
    slope = [fits[sz][0] for sz in sizes]
    intercept = [fits[sz][1] for sz in sizes]
    ax[0].semilogx(dp, np.array(slope)*1e-3, '-o')
    ax[0].set_ylabel('$db/dz$ (-)', fontsize=16)
    ax[1].loglog(dp, intercept, '-o')
    ax[1].set_ylabel('$b_0$ (mm)', fontsize=16)
    ax[1].loglog(np.logspace(-4, -2, 100), np.logspace(-4, -2, 100)*1e3)
    ax[1].set_xlabel('$d_p$ (m)', fontsize=16)
    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(top=True, labelsize=12, which='both')

    with open('breadth_slopes.pkl', 'wb') as f:
        pickle.dump([dp, np.array(slope) * 1e-3], f)

    with open('breadth_profiles.pkl', 'wb') as f:
        pickle.dump(dct, f)

    plt.show()


def phi_profiles():
    sizes = ['200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    cmap = cmo.dense
    fig, ax = plt.subplots(figsize=(5, 8))
    with open('breadth_profiles.pkl', 'rb') as f:
        breadth_dct = pickle.load(f)

    with open('velocity_profiles.pkl', 'rb') as f:
        velocity_dct = pickle.load(f)

    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    exp_dp = data_means.index.values * 1e-6  # particle diameter in m
    exp_drdt = data_means['drdt'].values  # melt rate in m/s
    exp_drdt = exp_drdt[[3, 5, 6, 7, 8, 9, 10, 11]]
    exp_drdt = {sz: arr for sz, arr in zip(sizes, exp_drdt)}

    ax.plot([0, 0], [-0.1, 0], '-', color='0.4')
    for sz in sizes:
        color = cmap((sizes.index(sz)+1)/(len(sizes)+1))
        b, bz = breadth_dct[sz]
        w, wz = velocity_dct[sz]
        wi = -np.interp(-bz, -wz, w, left=np.nan, right=np.nan)
        phi = -exp_drdt[sz] * 0.6 * bz / (b * wi)

        ax.plot(phi, bz, color=color, label=sz.replace('u', '$\mu$'))
    ax.set_ylim([-0.1, 0])
    ax.set_xlabel('$\phi$', fontsize=16)
    ax.set_ylabel('$z$ (m)', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def vel_profile_settling_comparison():
    sizes = ['90um', '200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    cmap = cmo.dense
    fig, ax = plt.subplots(1, len(sizes), figsize=(16, 8))
    with open('velocity_profiles.pkl', 'rb') as f:
        velocity_dct = pickle.load(f)

    z = -np.logspace(-6, -1, 100)
    for i, sz in enumerate(sizes):
        dp = float(sz[:-2]) * ({'mm': 1e-3, 'um': 1e-6}[sz[-2:]])
        label = "{:.1f} mm".format(dp * 1e3)
        color = cmap((sizes.index(sz)+2)/(len(sizes)+2))
        w, wz, we = velocity_dct[sz]

        if sz == '90um':
            w[wz < -.085] = np.nan

        ws = np.array([-settling_velocity_at_z(dp, 293.15, 0, 2500, z_val) for z_val in z])
        ws_i = np.interp(wz, z, ws)

        fc = [c for c in color[:3]] + [0.4]
        ax[i].fill_betweenx(wz, w-we, w+we, color=fc)
        ax[i].plot(w, wz, color=color, label=label, lw=2)
        ax[i].plot(ws, z-dp/2, '-k', lw=2)
        # ax[i].plot(ws_i/w, wz, '-k', lw=2)
        ax[i].set_title(label)

    for i, a in enumerate(ax):
        a.set_ylim([-0.1, 0])
        a.tick_params(labelsize=12, right=i < len(ax)-1, labelleft=(i==0))
        a.set_xlim([0, None])
        a.set_xlabel('$w$ (m/s)', fontsize=16)
        # a.set_facecolor('0.9')
    ax[0].set_ylabel('$z$ (m)', fontsize=16)
    plt.tight_layout()
    plt.show()



def all_exp_profiles():
    sizes = ['55um', '90um', '200um', '375um', '875um', '1mm', '2mm', '3mm', '4mm', '8mm']
    cmap = cmo.dense
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    with open('breadth_profiles.pkl', 'rb') as f:
        breadth_dct = pickle.load(f)

    with open('velocity_profiles.pkl', 'rb') as f:
        velocity_dct = pickle.load(f)

    data = pd.read_csv('data/ablation_experiments.csv')
    data = data[(data['material'] == 'glass') & (data['salinity'] == 0)]
    data_means = data.groupby('dp').mean(numeric_only=True)
    exp_drdt = data_means['drdt'].values  # melt rate in m/s
    exp_drdt = exp_drdt[[1, 3, 5, 6, 7, 8, 9, 10, 11]]
    exp_drdt = {sz: arr for sz, arr in zip(sizes, exp_drdt)}

    ax[2].plot([0, 0], [-0.1, 0], '-', color='0.4')
    for sz in sizes:
        dp = float(sz[:-2]) * ({'mm': 1e-3, 'um': 1e-6}[sz[-2:]])
        label = "{:.1f} mm".format(dp * 1e3)
        color = cmap((sizes.index(sz)+1)/(len(sizes)+1))
        b, bz = breadth_dct[sz]
        b = running_average(b, n=5)
        w, wz, we = velocity_dct[sz]

        if sz == '90um':
            b[bz < -.085] = np.nan
            w[wz < -.085] = np.nan

        wi = -np.interp(-bz, -wz, w, left=np.nan, right=np.nan)
        phi = -exp_drdt[sz] * 0.6 * bz / (b * wi)

        ax[0].plot(b*1e3, bz, color=color, label=label, lw=2)
        fc = [c for c in color[:3]] + [0.2]
        ax[1].fill_betweenx(wz, w-we, w+we, color=fc)
        ax[1].plot(w, wz, color=color, label=label, lw=2)
        ax[2].plot(phi, bz, color=color, label=label, lw=2)

    for i, a in enumerate(ax):
        a.set_ylim([-0.1, 0])
        a.tick_params(labelsize=12, right=i < len(ax)-1, labelleft=(i==0))
        a.set_xlim([0, None])
    ax[0].set_ylabel('$z$ (m)', fontsize=16)
    ax[0].set_xlabel('$b$ (mm)', fontsize=16)
    ax[1].set_xlabel('$w$ (m/s)', fontsize=16)
    ax[2].set_xlabel('$\phi$', fontsize=16)
    ax[0].legend(fontsize=14)
    ax[0].set_xlim([0, 26])
    plt.tight_layout()
    plt.show()


def show_breadth(sz):
    bins = np.linspace(top_ice[sz], 2000, 100, dtype=np.int32)
    y = (bins[1:] + bins[:-1]) / 2

    with open('breadth_profiles/breadths_'+sz+'.pkl', 'rb') as f:
        breadth = pickle.load(f)

    cmap = plt.get_cmap('Greens')
    fig, ax = plt.subplots(1, 2, figsize=(6, 10), sharey=True, sharex=True)
    for i in range(len(breadth[0])):
        color = cmap(i/len(breadth[0]))
        ax[0].plot(np.array(breadth[0][i]) * ccal[sz] * 1e-3, (top_ice[sz]-y) * ccal[sz] * 1e-6, color=color)
        ax[1].plot(np.array(breadth[1][i]) * ccal[sz] * 1e-3, (top_ice[sz]-y) * ccal[sz] * 1e-6, color=color)

    ax[0].plot(np.nanmean(breadth[0], axis=0) * ccal[sz] * 1e-3, (top_ice[sz]-y) * ccal[sz] * 1e-6, '-k')
    ax[1].plot(np.nanmean(breadth[1], axis=0) * ccal[sz] * 1e-3, (top_ice[sz] - y) * ccal[sz] * 1e-6, '-k')

    ax[0].set_xlabel('$b$ (mm)', fontsize=16)
    ax[1].set_xlabel('$b$ (mm)', fontsize=16)
    ax[0].set_ylabel('$z$ (m)', fontsize=16)
    ax[0].set_xlim([0, None])
    ax[0].set_ylim([None, 0])
    ax[1].set_xlim([0, None])
    ax[1].set_ylim([None, 0])
    ax[0].tick_params(labelsize=12, right=True)
    ax[1].tick_params(labelsize=12, left=False, labelleft=False)
    plt.tight_layout()
    plt.show()


def compute_breadth_heatmap(sz, debug=False):
    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)
    bin_thresh = 230 if sz == '90um_old' else 220
    hm_thresh = 5  # heatmap threshold
    hm_peak_thresh = 0.5  # heatmap peak threshold
    n_avg = int(0.5 * 120)  # number of frames to use for running average (0.5 s * 120 fps)
    bins = np.linspace(top_ice[sz], 2000, 100, dtype=np.int32)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    _, frame0 = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    N = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) // 3
    print(N)
    if sz == '90um':
        N = 120 * 60

    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        _, _, _, ybot = pickle.load(f)[sz]

    breadth = [[], []]
    memry = []
    st = time.time()
    for n in range(N):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        gray[:, :top[sz]] = 0
        gray[:, bottom[sz]:] = 255

        # 1) Obtain mask for ice-sediment block
        ice_mask = find_ice_mask(gray)
        ice_mask = cv.dilate(ice_mask, kernel=np.ones((3, 3)), iterations=2)

        # 2) Obtain mask for background
        _, binry = cv.threshold(gray, bin_thresh, 255, cv.THRESH_BINARY)

        binry[ice_mask == 1] = 255

        bg_mask = np.zeros((binry.shape[0] + 2, binry.shape[1] + 2), dtype=np.uint8)
        # seed_point = [[i, binry.shape[1]//2] for i in range(10, binry.shape[0]) if binry[i, binry.shape[1]//2] == 255][0]
        seed_point = [binry.shape[0]//2, binry.shape[1]//2]
        cv.floodFill(binry, bg_mask, seed_point, (255,))
        p_mask = (255 - binry)/255

        # 3) Update buffer
        memry.append(p_mask)
        if len(memry) >= n_avg:
            # 4 ) Compute breadth
            heatmap = np.sum(memry, axis=0)
            heatmap[ice_mask == 1] = np.nan

            if debug:
                edges = find_edges(ice_mask, largest_only=True)
                plt.figure()
                plt.imshow(gray.T)
                plt.plot(seed_point[0], seed_point[1], 'ok', mfc='w')
                plt.plot(edges[:, 1], edges[:, 0], '-r')

                plt.figure()
                plt.imshow(p_mask.T)

                plt.figure()
                plt.imshow(heatmap.T, cmap=cmo.haline)
                plt.colorbar()

            b0, b1 = [], []
            for i in range(len(bins)-1):
                slc_l = heatmap[:xmid[sz], bins[i]:bins[i+1]]
                slc_r = heatmap[xmid[sz]:, bins[i]:bins[i + 1]]
                bl = np.nanmean(slc_l, axis=1)
                br = np.nanmean(slc_r, axis=1)

                if np.nanmax(bl) >= hm_thresh:
                    bl = running_average(bl, n=5)
                    bl[np.isnan(bl)] = 0
                    ind_m = np.nanargmax(bl)
                    dr_lst = np.argwhere(bl[ind_m+1:] < hm_peak_thresh)
                    dl_lst = np.argwhere(bl[:ind_m] < hm_peak_thresh)
                    dr = len(bl) - 1 - ind_m if len(dr_lst) == 0 else dr_lst[0][0] + 1
                    dl = 0 if len(dl_lst) == 0 else ind_m - dl_lst[-1][0]
                    b0.append(dr + dl)
                else:
                    b0.append(np.nan)

                if np.nanmax(br) >= hm_thresh:
                    br = running_average(br, n=5)
                    br[np.isnan(br)] = 0
                    ind_m = np.nanargmax(br)
                    dr_lst = np.argwhere(br[ind_m+1:] < hm_peak_thresh)
                    dl_lst = np.argwhere(br[:ind_m] < hm_peak_thresh)
                    dr = len(bl) - 1 - ind_m if len(dr_lst) == 0 else dr_lst[0][0] + 1
                    dl = 0 if len(dl_lst) == 0 else ind_m - dl_lst[-1][0]
                    b1.append(dr + dl)
                else:
                    b1.append(np.nan)

            if debug:
                plt.figure()
                y = (bins[1:] + bins[:-1])/2
                I = y < ybot[n]
                plt.plot(np.array(b0)[I], y[I], label='left')
                plt.plot(np.array(b1)[I], y[I], label='right')
                plt.legend()
                plt.show()

            breadth[0].append(b0)
            breadth[1].append(b1)
            memry = []

        eta = (time.time() - st) * (N-n-1) / (n+1)
        h, m, s = int(eta / 3600), int((eta % 3600) / 60), int(eta % 60)
        print("\r[compute_breadth_heatmap] {:s}: {:.1f}% ({:02d}:{:02d}:{:02d} remaining)".format(sz, (n+1)/N*100, h, m, s), end='')

    print(" \033[42mdone\033[0m")
    with open('breadth_profiles/breadths_'+sz+'.pkl', 'wb') as f:
        pickle.dump(breadth, f)


def compute_breadth_from_tracks(sz, debug=False):
    def get_indices_for_disk(xy, r):
        return np.array([[x, y] for x in range(int(xy[0]-r), int(xy[0]+r+1)) for y in range(int(xy[1]-r), int(xy[1]+r)+1)
               if np.sqrt((x - xy[0])**2 + (y-xy[1])**2) <= r])

    t_avg = {'8mm': 24, '4mm': 12, '3mm': 9, '2mm': 6, '1mm': 3, '875um': 1}[sz]
    n_avg = int(t_avg * 120)  # number of frames to use for average
    img_sz = (2000, 1500)
    hm_thresh = 5
    hm_peak_thresh = 1.5

    with open('plumeVideoHeightsWidths.pkl', 'rb') as f:
        _, _, yt, _ = pickle.load(f)[sz]

    tracks = get_tracks_xyn(sz)
    max_n = np.max([trk[-1][2] for trk in tracks])
    dp = float(sz[:-2]) / ({'mm': 1e-3, 'um': 1}[sz[-2:]] * ccal[sz])  # particle diameter in pixels
    breadth = [[], []]
    breadth_y = []
    st = time.time()
    bins = np.linspace(top_ice[sz], img_sz[0], 100, dtype=np.int32)
    for n in range(0, int(max_n)-n_avg, n_avg):
        heat_map = np.zeros(img_sz)
        for trk in tracks:
            trk = trk[(trk[:, 2] >= n) & (trk[:, 2] < (n + n_avg)), :]
            if len(trk) > 0:
                trk_map = np.zeros(heat_map.shape)
                for i in range(len(trk)):
                    ind = get_indices_for_disk(trk[i][:2], dp / 2)
                    trk_map[ind[:, 1], ind[:, 0]] = 1
                if len(trk) > 1:
                    vec = trk[1:, :2] - trk[:-1, :2]
                    vec = np.vstack((vec, vec[-1, :]))
                    vec_orth = vec[:, ::-1] * dp/2 / np.reshape(np.sqrt(np.sum(vec**2, axis=1)), (vec.shape[0], 1))
                    vec_orth[:, 0] *= -1
                    p0 = trk[:, :2] + vec_orth
                    p1 = trk[:, :2] - vec_orth
                    cv.fillPoly(trk_map, [np.vstack((p0, p1)).astype(np.int32)], (1, 1, 1))

                heat_map += trk_map

        if debug:
            plt.figure()
            plt.imshow(heat_map, cmap=cmo.haline)
            plt.colorbar()
            plt.show()

        b0, b1 = [], []
        for i in range(len(bins) - 1):
            slc_l = heat_map[bins[i]:bins[i + 1], :xmid[sz]]
            slc_r = heat_map[bins[i]:bins[i + 1], xmid[sz]:]
            bl = np.nanmean(slc_l, axis=0)
            br = np.nanmean(slc_r, axis=0)

            if np.nanmax(bl) >= hm_thresh:
                bl = running_average(bl, n=5)
                bl[np.isnan(bl)] = 0
                ind_m = np.nanargmax(bl)
                dr_lst = np.argwhere(bl[ind_m + 1:] < hm_peak_thresh)
                dl_lst = np.argwhere(bl[:ind_m] < hm_peak_thresh)
                dr = len(bl) - 1 - ind_m if len(dr_lst) == 0 else dr_lst[0][0] + 1
                dl = 0 if len(dl_lst) == 0 else ind_m - dl_lst[-1][0]
                b0.append(dr + dl)
            else:
                b0.append(np.nan)

            if np.nanmax(br) >= hm_thresh:
                br = running_average(br, n=5)
                br[np.isnan(br)] = 0
                ind_m = np.nanargmax(br)
                dr_lst = np.argwhere(br[ind_m + 1:] < hm_peak_thresh)
                dl_lst = np.argwhere(br[:ind_m] < hm_peak_thresh)
                dr = len(bl) - 1 - ind_m if len(dr_lst) == 0 else dr_lst[0][0] + 1
                dl = 0 if len(dl_lst) == 0 else ind_m - dl_lst[-1][0]
                b1.append(dr + dl)
            else:
                b1.append(np.nan)

        breadth[0].append(b0)
        breadth[1].append(b1)
        avg_top_y = np.nanmean(yt[n:(n+n_avg)])
        breadth_y.append((bins[1:] + bins[:-1])/2 - avg_top_y)

        eta = (time.time() - st) * (max_n - n - 1) / (n + 1)
        h, m, s = int(eta / 3600), int((eta % 3600) / 60), int(eta % 60)
        print("\r[compute_breadth_tracks] {:s}: {:.1f}% ({:02d}:{:02d}:{:02d} remaining)".format(sz, (n + 1) / max_n * 100, h, m, s), end='')
    print(" \033[42mdone\033[0m")
    with open('breadth_profiles/breadths_' + sz + '.pkl', 'wb') as f:
        pickle.dump((breadth, breadth_y), f)



def piv_sides(sz, debug=False):
    """
    375um displacement: roughly 6 px per frame -> window size at least 24 px
    """

    path = folder + sz + '/' + sz + '.MOV'
    mask_thresh = 100
    cwsz = (32, 64)  # correlation window size
    overlap = 0.5  # correlation window overlap fraction
    n_avg = 10  # int(0.5 * 120)  # number of frames to use for correlation window averaging (0.5 s * 120 fps)
    save_path = 'particle_correlations/piv_sides_{:s}_{:d}_{:d}_thresh100.pkl'.format(sz, cwsz[0], cwsz[1])

    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open file")
        exit()

    _, frame0 = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    N = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) // 5
    cw_y = np.arange(0, bottom[sz]-top[sz]-cwsz[1], (1-overlap)*cwsz[1], dtype=np.int32)  # upper y-positions of correlation windows

    avg_cw = [[np.zeros((2*cwsz[0]-1, 2*cwsz[1]-1)) for _ in cw_y] for _ in range(2)]  # placeholder for average correlation windows
    avg_cwx = [[0.0 for _ in cw_y] for _ in range(2)]  # placeholder for average x-locations
    avg_cnt = [[0 for _ in cw_y] for _ in range(2)]
    prev_cw = [[None for _ in cw_y] for _ in range(2)]  # placeholder for previous correlation windows
    vel = []  # placeholder for velocity vectors (x, y, u, v, n)
    vel_avg = []
    st = time.time()
    for n in range(0, N):
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        # 1) Obtain mask for ice-sediment block
        gray[:, bottom[sz]-1:] = 255
        gray[:, :top[sz]-5] = 0
        mask = find_ice_mask(gray, mask_thresh=mask_thresh)
        mask = cv.dilate(mask, kernel=np.ones((3, 3)), iterations=2)
        mask = mask[:, top[sz]:bottom[sz]]
        gray = gray[:, top[sz]:bottom[sz]]

        # 2) Replace mask with Gaussian noise similar to background
        bg = gray[:50, :]  # background
        bg_mean, bg_std = np.mean(bg), np.std(bg)
        R = (bg_std * np.random.randn(*mask.shape) + bg_mean).astype(np.uint8)

        mg = deepcopy(gray)
        bg_thresh = bg_mean - 3 * bg_std
        mg[mg > bg_thresh] = 255
        mg[mask == 1] = R[mask == 1]
        mg = ((mg - mask_thresh).astype(np.float64) * 255/(255-mask_thresh)).astype(np.uint8)  # enhance contrast
        mg = 255 - mg

        # mg[mg <= 50] = 250 - R[mg <= 50]

        # 3) Determine correlation window positions
        edges = find_edges(mask, largest_only=True)
        holder_width = np.max(edges[edges[:, 0] < 10, 1]) - np.min(edges[edges[:, 0] < 10, 1])
        top_y = None
        for i in range(10, gray.shape[1]):
            I = np.abs(edges[:, 0] - i) < 5
            width = np.max(edges[I, 1]) - np.min(edges[I, 1])
            if width > 1.2 * holder_width:
                top_y = i
                break

        edges = edges[edges[:, 0] > 0, :]
        edges = edges[edges[:, 0] < np.max(edges[:, 0]), :]
        cw_x = np.nan * np.zeros((2, cw_y.size))
        for i in range(cw_y.size):
            if cw_y[i] >= top_y:
                I = (edges[:, 0] >= cw_y[i]) & (edges[:, 0] < cw_y[i] + cwsz[1])
                edge_i = edges[I, :]
                if np.std(np.abs(edge_i[:, 1]-np.mean(edge_i[:, 1]))) < cwsz[0]:
                    cx = np.mean(edge_i[:, 1])
                    cw_x[0, i] = np.mean(edge_i[edge_i[:, 1] < cx, 1]) - cwsz[0]
                    cw_x[1, i] = np.mean(edge_i[edge_i[:, 1] > cx, 1])

        # 4) Get correlation windows
        cw = [[None for _ in cw_y] for _ in range(2)]
        for i in range(cw_y.size):
            for k in [0, 1]:
                if not np.isnan(cw_x[k, i]):
                    j0, j1 = int(cw_x[k, i]), int(cw_y[i])
                    cw[k][i] = mg[j0:j0+cwsz[0], j1:j1+cwsz[1]]

        # 5) Perform correlation between frames
        for i in range(len(cw_y)):
            for k in [0, 1]:
                if (cw[k][i] is not None) and (prev_cw[k][i] is not None):
                    corr = correlate2d(cw[k][i].astype(np.float64), prev_cw[k][i].astype(np.float64))
                    corr = np.sqrt(corr / np.prod(cwsz))
                    avg_cw[k][i] += corr
                    avg_cwx[k][i] += cw_x[k, i]
                    avg_cnt[k][i] += 1

                    ind = np.unravel_index(np.argmax(corr), corr.shape)

                    try:
                        ym, _ = ind[1] + parabolic_peak_fit([-1, 0, 1], corr[ind[0], (ind[1] - 1):(ind[1] + 2)])
                        xm, _ = ind[0] + parabolic_peak_fit([-1, 0, 1], corr[(ind[0] - 1):(ind[0] + 2), ind[1]])

                        dy = ym - cwsz[1] + 1
                        dx = xm - cwsz[0] + 1

                        vel.append([cw_x[k, i], cw_y[i], dx, dy, n])
                    except ValueError:
                        pass

                    if debug:
                        plt.figure()
                        plt.imshow(frame[:, top[sz]:, :])
                        plt.gca().add_artist(plt.Rectangle((int(cw_y[i]), int(cw_x[k][i])), cwsz[1], cwsz[0], facecolor='none', edgecolor='r', linewidth=2))

                        fig, ax = plt.subplots(3, 1)
                        ax[1].imshow(cw[k][i])
                        ax[0].imshow(prev_cw[k][i])
                        ax[2].imshow(corr)
                        ax[0].set_title("dx = {:.1f} px,  dy = {:.1f} px".format(dx, dy))

                        plt.figure()
                        N = corr.shape[1]
                        x = np.arange(N) - N//2
                        plt.plot(x, corr[31, :])
                        plt.plot([0, 0], [0, np.max(corr)], '-k')
                        plt.show()

        # 6) Average correlation windows and compute velocity vectors
        if n > 0 and n % n_avg == 0:
            # with open('temp.pkl', 'wb') as f:
            #     pickle.dump([avg_cw, avg_cnt, avg_cwx], f)
            for i in range(cw_y.size):
                for k in [0, 1]:
                    if avg_cnt[k][i] > 0:
                        corr = avg_cw[k][i] / avg_cnt[k][i]
                        y = cw_y[i] + cwsz[1] / 2
                        x = avg_cwx[k][i] + cwsz[0] / 2

                        ind = np.unravel_index(np.argmax(corr), corr.shape)
                        # if (0 < ind[0] < cwsh[0] - 1) and (0 < ind[1] < cwsh[1] - 1):
                        ym, _ = ind[1] + parabolic_peak_fit([-1, 0, 1], corr[ind[0], (ind[1] - 1):(ind[1] + 2)])
                        xm, _ = ind[0] + parabolic_peak_fit([-1, 0, 1], corr[(ind[0] - 1):(ind[0] + 2), ind[1]])

                        dx = xm - cwsz[0] + 1
                        dy = ym - cwsz[1] + 1

                        vel_avg.append([x, y, dx, dy, n-n_avg/2])

            # reset averages
            avg_cw = [[np.zeros((2 * cwsz[0] - 1, 2 * cwsz[1] - 1)) for _ in cw_y] for _ in
                      range(2)]  # placeholder for average correlation windows
            avg_cwx = [[0.0 for _ in cw_y] for _ in range(2)]  # placeholder for average x-locations
            avg_cnt = [[0 for _ in cw_y] for _ in range(2)]

        if debug:
            plt.figure()
            plt.imshow(frame[:, top[sz]:, :])
            plt.plot(edges[:, 0], edges[:, 1], '-g')

            for x, y, u, v, _ in vel:
                plt.gca().add_artist(plt.Arrow(y, x, 10*v, 10*u, width=10, color='r'))
            plt.show()



        prev_cw = cw  # remember current correlation windows
        eta = (time.time() - st) * (N-n-1) / (n+1)
        h, m, s = int(eta / 3600), int((eta % 3600) / 60), int(eta % 60)
        print("\r[piv_sides] {:s}: {:.1f}% ({:02d}:{:02d}:{:02d} remaining)".format(sz, (n+1)/N*100, h, m, s), end='')

    with open(save_path, 'wb') as f:
        pickle.dump([vel, vel_avg], f)
    print(" SAVED!")


def piv(sz):
    """
    375um displacement: roughly 6 px per frame -> window size at least 24 px
    """

    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)
    mask_thresh = 80

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    masked_frames = []
    frames = []
    for _ in range(2):

        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        frames.append(gray)

        mask = find_ice_mask(frame, mask_thresh=80)  # higher mask threshold as we only care about the sides
        mask = cv.dilate(mask, kernel=np.ones((3, 3)), iterations=2)
        mask = mask[:, top[sz]:bottom[sz]]
        gray = gray[:, top[sz]:bottom[sz]]

        bg = gray[:50, :]  # background
        bg_mean, bg_std = np.mean(bg), np.std(bg)
        R = (bg_std * np.random.randn(*mask.shape) + bg_mean).astype(np.uint8)
        mg = deepcopy(gray)
        mg[mask == 1] = R[mask == 1]
        mg = ((mg - mask_thresh).astype(np.float64) * 255/(255-mask_thresh)).astype(np.uint8)
        # mg[ero == 255] = 215
        masked_frames.append(mg)

        # gray[mask[1:-1, 1:-1] == 1] = 255
        # gray = gray[:, top[sz]:bottom[sz]]
        #
        # _, binry2 = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
        # binry2 = (255 - binry2) // 255

    X, Y, U, V = piv_process_image_pair(255-masked_frames[0], 255-masked_frames[1], (32, 128), overlap=(0.9, 0.5))

    np.savez('piv375um_test2.npz', X=X, Y=Y, U=U, V=V)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax[0, 0].imshow(255-frames[0])
    ax[0, 1].imshow(255-frames[1])
    ax[1, 0].imshow(255-masked_frames[0])
    ax[1, 1].imshow(255-masked_frames[1])

    plt.figure()
    plt.imshow(255 - frames[0], cmap='gray')
    plt.quiver(X, Y, U, V, color=(.6, 0, 0))

    plt.figure()
    plt.imshow(U)

    plt.show()


def test_piv_process_image_pair(im1, im2, cwsh, overlap=0.0):
    """
    :param im1: (MxN) ndarray, gray image of current frame
    :param im2: (MxN) ndarray, gray image of next frame
    :param cwsh: tuple, correlation window shape
    :param overlap: float, fraction of window overlap
    :return X, Y, U, V: position and direction matrices of displacement vectors
    """

    assert im1.shape == im2.shape, "the two frames have different sizes"

    i, j = 220, 900
    cw1 = im1[i:(i+cwsh[0]), j:(j+cwsh[1])].astype(np.float64)
    cw2 = im2[i:(i+cwsh[0]), j:(j+cwsh[1])].astype(np.float64)
    corr = correlate2d(cw1, cw2, mode='full')
    # cw2 = np.flipud(cw2)
    conv = convolve2d(cw1, np.flipud(cw2), mode='full')
    fconv = fftconvolve(cw1, np.flipud(cw2))

    mat = corr

    ind = np.unravel_index(np.argmax(mat), mat.shape)
    # if (0 < ind[0] < cwsh[0] - 1) and (0 < ind[1] < cwsh[1] - 1):
    xm, _ = ind[1] + parabolic_peak_fit([-1, 0, 1], mat[ind[0], (ind[1]-1):(ind[1]+2)])
    ym, _ = ind[0] + parabolic_peak_fit([-1, 0, 1], mat[(ind[0] - 1):(ind[0] + 2), ind[1]])

    dx = xm - cwsh[1] + 1  #/2
    dy = ym - cwsh[0] + 1  #/2

    print("dx = {:.1f} px  |  dy = {:.1f} px".format(dx, dy))

    fig, ax = plt.subplots(4, 1)
    ax[0].imshow(cw1)
    ax[1].imshow(cw2)
    ax[2].imshow(mat)
    ax[2].plot(ind[1], ind[0], 'xr')
    ax[3].imshow(conv)

    plt.show()


def piv_process_image_pair(im1, im2, cwsh, overlap: Union[float, tuple]):
    """
    :param im1: (MxN) ndarray, gray image of current frame
    :param im2: (MxN) ndarray, gray image of next frame
    :param cwsh: tuple, correlation window shape
    :param overlap: float, fraction of window overlap
    :return X, Y, U, V: position and direction matrices of displacement vectors
    """

    assert im1.shape == im2.shape, "the two frames have different sizes"

    if type(overlap) is float:
        overlap = (overlap, overlap)

    I = range(0, im1.shape[0] - cwsh[0], int((1 - overlap[0]) * cwsh[0]))
    J = range(0, im1.shape[1] - cwsh[1], int((1 - overlap[1]) * cwsh[1]))
    X, Y = np.meshgrid(J, I)
    X += cwsh[1]//2
    Y += cwsh[0]//2
    U, V = np.zeros(X.shape), np.zeros(X.shape)
    st = time.time()
    for ii, i in enumerate(I):
        for jj, j in enumerate(J):
            cw1 = im1[i:(i + cwsh[0]), j:(j + cwsh[1])].astype(np.float64)
            cw2 = im2[i:(i + cwsh[0]), j:(j + cwsh[1])].astype(np.float64)
            corr = correlate2d(cw1, cw2, mode='full')

            ind = np.unravel_index(np.argmax(corr), corr.shape)
            # if not((0 < ind[0] < cwsh[0] - 1) and (0 < ind[1] < cwsh[1] - 1)):
            #     continue

            xm, _ = ind[1] + parabolic_peak_fit([-1, 0, 1], corr[ind[0], (ind[1]-1):(ind[1]+2)])
            ym, _ = ind[0] + parabolic_peak_fit([-1, 0, 1], corr[(ind[0] - 1):(ind[0] + 2), ind[1]])

            dx = xm - cwsh[1] + 1  #/2
            dy = ym - cwsh[0] + 1  #/2

            if abs(dx) < cwsh[0] and abs(dy) < cwsh[1]:
                U[ii, jj] = dx
                V[ii, jj] = dy
            else:
                U[ii, jj] = np.nan
                V[ii, jj] = np.nan

            # if dy < -20:
            #     plt.imshow(im1)
            #     plt.gca().add_artist(plt.Rectangle((j, i), cwsh[0], cwsh[1], facecolor='none', edgecolor='r', linewidth=2))
            #     print("dx = {:.1f} px  |  dy = {:.1f} px".format(dx, dy))
            #
            #     fig, ax = plt.subplots(1, 3)
            #     ax[0].imshow(cw1)
            #     ax[1].imshow(cw2)
            #     ax[2].imshow(corr)
            #     ax[2].plot(ind[1], ind[0], 'xr')
            #
            #     plt.show()
            frac = (ii * len(J) + jj + 1)/(len(I)*len(J))
            eta = (time.time() - st) * (1-frac)/frac
            h, m, s = int(eta / 3600), int((eta % 3600) / 60), int(eta % 60)
            print("\rprogress: {:.1f}%  (done in {:02d}:{:02d}:{:02d})".format(frac*100, h, m, s), end='')
    return X, Y, U, V


def generate_particle_cloud(sz, debug=False):
    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    if debug:
        n_frames = 5

    cloud = []
    st = time.time()
    # dp = float(sz[:-2]) / (1e-3 * ccal[sz])  # particle diameter in pixels
    # dp = float(sz[:-6]) / (1e-3 * ccal[sz])  # particle diameter in pixels
    dp = float(sz[:-2]) / ({'mm': 1e-3, 'um': 1}[sz[-2:]] * ccal[sz])
    # print(dp)
    min_radius = 0.6 * dp/2
    max_radius = 1.4 * dp/2
    for n in range(n_frames):
        ret, frame = cap.read()

        if np.any([fs == 0 for fs in frame.shape]):
            continue

        frame = frame[:, :bottom[sz]]
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        _, binry = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
        binry = 255 - binry

        mask = np.zeros((binry.shape[0]+2, binry.shape[1]+2), dtype=np.uint8)
        cv.floodFill(binry, mask, (0, 0), (0,))

        if sz == '875um':
            gray = (gray.astype(np.float64) - 80) * 255 / 200  # increase contrast
        else:
            gray = (gray.astype(np.float64) - 55) * 255 / 200  # increase contrast
        gray[gray < 0] = 0
        gray = gray.astype(np.uint8)
        gray[mask[1:-1, 1:-1] == 1] = 255
        circles = find_particle_images_sbd(gray, min_radius, max_radius)
        for c in circles:
            cloud.append(c + [n])

        if debug:
            plt.figure()
            plt.imshow(gray)
            for x, y, radius in circles:
                plt.gca().add_artist(plt.Circle((y, x), radius, edgecolor='r', facecolor='none', linewidth=2))
                plt.gca().add_artist(plt.Circle((y, x), radius/10, color=(.6, 0, 0)))
            plt.show()

        eta = (time.time() - st) * (n_frames - n - 1)/(n+1)
        h, m, s = int(eta // 3600), int((eta % 3600) // 60), int(eta % 60)
        print("\r[detection] {:.1f}%   done in {:02d}:{:02d}:{:02d}".format((n+1)/n_frames*100, h, m, s), end='')

    cloud = np.array(cloud)
    print("\nDone! Particle count: {:d}".format(len(cloud)))
    return cloud


def tracks_from_particle_cloud(cloud, debug=False):
    def get_frame(num):
        sz = '875um'
        path = folder + sz + '/' + sz + '.MOV'
        cap = cv.VideoCapture(path)
        cap.set(cv.CAP_PROP_POS_FRAMES, num)
        _, frame = cap.read()
        return frame

    search_radius = 10#15  # search radius in px (used 10 for 875um, 15 otherwise)
    search_angle = np.pi/4  # search angle in radians (only used for 875um)
    dy_pred = 15#20  # vertical displacement to use for prediction if no history available in px (used 15 for 875um, 20 otherwise)

    # cloud = cloud[cloud[:, 3] < 100, :]

    id_arr = np.reshape(np.arange(cloud.shape[0], dtype=np.int32), (cloud.shape[0], 1))
    cloud = np.hstack((id_arr, cloud))

    # Pairing
    back = {int(i): None for i in id_arr}
    forward = {int(i): None for i in id_arr}

    st = time.time()
    for n in np.unique(cloud[:, 4]):
        cn = cloud[cloud[:, 4] == n]        # particles in frame n
        cn1 = cloud[cloud[:, 4] == n + 1]   # particles in frame n + 1

        if len(cn1) == 0:
            continue

        if debug:
            plt.figure()
            plt.imshow(np.fliplr(cv.rotate(get_frame(n), cv.ROTATE_90_CLOCKWISE)))
            plt.plot(cn[:, 1], cn[:, 2], 'o', color='C0')
            plt.plot(cn1[:, 1], cn1[:, 2], 'o', color='C1')

        for i in cn[:, 0].astype(np.int32):
            # apply predictor step if possible, otherwise assume particle to fall straight down
            p = cloud[i, 1:3] + np.array([0, dy_pred]) if back[i] is None else 2 * cloud[i, 1:3] - cloud[back[i], 1:3]

            if debug:
                plt.plot([cloud[i, 1], p[0]], [cloud[i, 2], p[1]], '-', color='C0')
                plt.gca().add_artist(plt.Circle(p, search_radius, edgecolor='C0', facecolor='none'))

            # compute distance to all other particles and find the closest
            d = np.sqrt(np.sum((cn1[:, 1:3] - p) ** 2, axis=1))
            if np.min(d) <= search_radius:
                min_id = cn1[np.argmin(d), 0].astype(np.int32)  # ID of closest particle
                v1, v2 = p - cloud[i, 1:3], cloud[min_id, 1:3] - cloud[i, 1:3]
                angle = np.acos(v1.dot(v2) / np.sqrt(v1.dot(v1) * v2.dot(v2)))
                if back[min_id] is None and angle <= search_angle/2:
                    # Only allow one particle connected
                    forward[i] = min_id
                    back[min_id] = i

                    if debug:
                        plt.plot([cloud[min_id, 1], cloud[i, 1]], [cloud[min_id, 2], cloud[i, 2]], '-or', markersize=2)

        if debug:
            plt.show()

        print("\r[pairing] {:.1f}%".format((n+1)/len(np.unique(cloud[:, 4]))*100), end='')
    print("  done in {:.0f} seconds!".format(time.time() - st))

    # Backtracking
    def recur_track(trk):
        if forward[trk[-1]] is not None and forward[trk[-1]] not in visited:
            return recur_track(trk + [forward[trk[-1]]])
        return trk

    tracks = []
    visited = []
    st = time.time()
    for i in forward:
        if i not in visited:
            track = recur_track([i])
            tracks.append(track)
            visited += track
        print("\r[backtracking] {:.1f}%".format((i + 1) / len(id_arr) * 100), end='')
    print("  done in {:.0f} seconds!".format(time.time() - st))
    print("Track count: {:d} tracks longer than 1!".format(len([1 for track in tracks if len(track) > 1])))
    return tracks


def save_tracks_as_csv(sz, cloud, tracks):
    # add track index to each particle
    cloud = np.hstack((cloud, np.zeros((cloud.shape[0], 1))))
    for j, trk in enumerate(tracks):
        for i in trk:
            cloud[i, -1] = j

    # x, y positions in px, radius in pixel, frame number, track number
    df = pd.DataFrame(data=cloud, columns=['x', 'y', 'r', 'n', 'track_id'])
    df.to_csv('particle_tracks/' + sz + '.csv')
    print("Saved tracks to {:s}!".format(sz + '.csv'))


def ptv_velocity(sz):
    dt = 1/120  # time interval between frames
    tracks = [trk for trk in get_tracks_xyn(sz) if len(trk) > 1]
    vel = [np.sqrt(np.sum((trk[j, :2] - trk[j+1, :2])**2)) * ccal[sz] * 1e-6 / dt
           for trk in tracks for j in range(len(trk)-1)]
    return np.mean(vel), np.std(vel)


def piv_velocity(sz):
    dt = 1/120  # time interval between frames
    cutoff_vy = 0.8

    with open('particle_correlations/piv_sides_{:s}_32_64.pkl'.format(sz), 'rb') as f:
        vel, vel_avg = pickle.load(f)

    inst_v = [np.sqrt(v[2]**2 + v[3]**2) for v in vel_avg if v[3] > cutoff_vy]
    inst_v = np.array(inst_v) * ccal[sz] * 1e-6 / dt
    return np.mean(inst_v), np.std(inst_v)


def analyse_tracks(sz):
    dt = 1/120  # time interval between frames
    cmap = cmo.dense
    colors = {'875um': cmap(0.1), '1mm': cmap(0.2), '2mm': cmap(0.4), '3mm': cmap(0.6), '4mm': cmap(0.8), '8mm': cmap(0.9)}

    tracks = get_tracks_xyn(sz)

    # Number of particles over time
    df = pd.read_csv('particle_tracks/' + sz + '.csv')
    n_arr, cnt_arr = np.unique(df['n'].values, return_counts=True)
    sn = 300
    sc = [np.mean(cnt_arr[i:(i+sn)]) for i in range(len(cnt_arr)-sn)]
    plt.plot(n_arr[:-sn] * 4 / 120, sc, '-o')
    plt.xlabel('time (s)')
    plt.ylabel('particle count')

    # Compute velocities
    left_vel, right_vel = [], []
    vectors = []
    curvature = []
    xmid = 500
    for trk in tracks:
        if len(trk) > 1:
            for j in range(len(trk)-1):
                d = np.sqrt(np.sum((trk[j, :2] - trk[j+1, :2])**2))
                v = d * ccal[sz] * 1e-6 / dt
                vy = (trk[j+1, 1] - trk[j, 1]) * ccal[sz] * 1e-6 / dt
                if trk[j, 0] < xmid:
                    left_vel.append(v)
                else:
                    right_vel.append(v)
                vectors.append(np.hstack((trk[j+1, :2] - trk[j, :2], trk[j+1, 1])))
        if len(trk) > 2:
            for j in range(1, len(trk)-1):
                dxdt = (trk[j + 1, 0] - trk[j - 1, 0]) / (2 * dt)
                dydt = (trk[j + 1, 1] - trk[j - 1, 1]) / (2 * dt)
                d2xdt2 = (trk[j + 1, 0] - trk[j, 0] + trk[j - 1, 0]) / dt ** 2
                d2ydt2 = (trk[j + 1, 1] - trk[j, 1] + trk[j - 1, 1]) / dt ** 2
                k = (dxdt*d2ydt2 - dydt*d2xdt2) / (dxdt**2 + dydt**2)**(3/2)
                sgn = -1 if trk[j, 0] < xmid else 1
                curvature.append(sgn * k)
    vel = left_vel + right_vel
    dp = float(sz[:-2]) * {'mm': 1e-3, 'um': 1e-6}[sz[-2:]]
    # dp = float(sz[:-6]) * {'mm': 1e-3, 'um': 1e-6}[sz[-6:-4]]
    ws = settling_velocity(dp, 293.15, 0, 2500)
    ws_avg = np.abs(average_velocity(dp))  # average settling velocity over height H = 0.1 m

    # Displacement vectors
    vectors = np.array(vectors)
    plt.figure()
    plt.scatter(vectors[:, 0], vectors[:, 1], 10, vectors[:, 2], cmap='Reds')
    plt.gca().set_aspect('equal')

    # Left/right velocity
    bins = np.arange(0, 2*np.mean(left_vel), np.std(left_vel)/5)
    lh, b = np.histogram(left_vel, bins, density=True)
    rh, b = np.histogram(right_vel, bins, density=True)
    x = (b[1:] + b[:-1])/2
    plt.figure()
    plt.semilogy(x, lh, '-o', markersize=3, label='left')
    plt.semilogy(x, rh, '-o', markersize=3, label='right')
    plt.semilogy([ws, ws], [1e-2, 1e2], '-', color='0.5')
    plt.semilogy([ws_avg, ws_avg], [1e-2, 1e2], '--', color='0.5')
    plt.text(ws*1.03, 1, '$w_s$', fontsize=16, color='0.5')
    plt.ylim([1e-2, 1e2])
    plt.xlim([0, 2*np.mean(left_vel)])
    plt.tick_params(labelsize=12)
    plt.ylabel('PDF($|v|$)  (s/m)', fontsize=16)
    plt.xlabel('$|v|$ (m/s)', fontsize=16)
    plt.tight_layout()
    plt.legend(fontsize=14)

    # All instantaneous velocities
    _, wt = settling_velocity_above_z(dp, 293.15, 0, 2500, -0.1, return_full=True)
    label = '$d_p = {:s}$ {:s}'.format(sz[:-2], sz[-2:])
    bins = np.arange(0, 2 * np.mean(vel), np.std(vel) / 5)
    h, b = np.histogram(vel, bins, density=True)
    x = (b[1:] + b[:-1])/2
    mu, sigma = np.mean(vel), np.std(vel)
    g = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sigma)**2)
    ht, _ = np.histogram(-wt, bins, density=True)
    plt.figure()
    plt.semilogy(x, g, '--k', label='Gaussian')
    plt.semilogy(x, h, '-o', color=colors[sz], markersize=4, lw=2, label=label)
    # plt.semilogy(x, ht, '-', color='0.4', markersize=4, lw=2, label='theoretical')
    plt.semilogy([ws, ws], [1e-2, 1e2], '-', color='0.5')
    plt.semilogy([ws_avg, ws_avg], [1e-2, 1e2], ':', color='0.5')
    # plt.text(ws*.85, .2, '$w_s$', fontsize=16, color='0.5')
    plt.semilogy([mu, mu], [1e-2, 1e2], '-', color=[c/2 for c in colors[sz][:3]])
    # plt.text(mu*1.03, .2, '$\mu$', fontsize=16, color=(.6, .3, .15))
    plt.ylim([1e-2, 1e2])
    plt.xlim([0, 2*mu])
    plt.tick_params(labelsize=12)
    plt.ylabel('PDF($|\mathbf{w}|$)  (s/m)', fontsize=16)
    plt.xlabel('$|\mathbf{w}|$ (m/s)', fontsize=16)
    plt.legend(fontsize=14)
    plt.tight_layout()

    # Curvature
    mu, sigma = np.nanmean(curvature), np.nanstd(curvature)
    bins = np.arange(mu-3*sigma, mu+3*sigma, sigma/20)
    h, b = np.histogram(curvature, bins, density=True)
    x = (b[1:] + b[:-1])/2
    plt.figure()
    plt.semilogy([0, 0], [1e-4, 1], '-k', lw=1)
    plt.semilogy(x, h, '-o', color=colors[sz], markersize=4, lw=2, label=label)
    plt.text(0.02, 0.98, 'away from ice', transform=plt.gca().transAxes, fontsize=16, color='0.3', va='top')
    plt.text(0.98, 0.98, 'toward ice', transform=plt.gca().transAxes, fontsize=16, color='0.3', va='top', ha='right')
    plt.ylim([1e-4, 1])
    plt.xlim([mu-3*sigma, mu+3*sigma])
    plt.tick_params(labelsize=12)
    plt.ylabel('PDF($\kappa$)  (m$^{-1}$)', fontsize=16)
    plt.xlabel('$\kappa$ (m$^{-1}$)', fontsize=16)
    plt.title(label)
    # plt.legend(fontsize=14)
    plt.tight_layout()

    # Acceleration
    plt.figure()
    long_tracks = [trk for trk in tracks if len(trk) > 10]
    for lt in long_tracks:
        vel = []
        for j in range(1, len(lt) - 1):
            dxdt = (lt[j + 1, 0] - lt[j - 1, 0]) / (2 * dt)
            dydt = (lt[j + 1, 1] - lt[j - 1, 1]) / (2 * dt)
            vel.append(np.sqrt(dxdt**2 + dydt**2))
        plt.plot(vel, color=(0, 0, 0.8, 0.01))
    plt.tick_params(labelsize=12)
    plt.ylabel('v', fontsize=16)
    plt.xlabel('t', fontsize=16)
    plt.title(label)
    # plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def add_tracks_to_video(sz):
    video_path = folder + sz + '/' + sz + '.MOV'
    save_path = folder + 'tracking/' + sz + '_tracking.mov'
    cap = cv.VideoCapture(video_path)

    cloud = pd.read_csv('particle_tracks/' + sz + '.csv')
    tracks = get_tracks_xyn(sz)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    ret, frame = cap.read()
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv.CAP_PROP_FPS)

    capSize = (frame.shape[0], frame.shape[1])
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case
    vout = cv.VideoWriter()
    vout.open(save_path, fourcc, fps, capSize, True)

    cmap = plt.get_cmap('Reds_r')
    max_trk_len = 10
    N = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for n in range(N):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cn = cloud[cloud['n'] == n]
        for i, p in cn.iterrows():
            trk = tracks[int(p['track_id'])]
            trk = trk[trk[:, 2] <= n, :]
            for j in range(max(0, len(trk)-max_trk_len-1), len(trk)-1):
                color = [int(255 * val) for val in cmap(min((len(trk)-1 - j)/max_trk_len, 0.99))]
                color = color[:-1][::-1]
                p1, p2 = trk[j, :2].astype(np.int32)[::-1], trk[j+1, :2].astype(np.int32)[::-1],
                cv.line(frame, p1, p2, color, 2)

        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

        # # Display the resulting frame
        # cv.imshow('frame', frame)
        # if cv.waitKey(1) == ord('q'):
        #     break

        vout.write(frame)
        print("\rprogress: {:.1f}%".format(n/N*100), end='')
    vout.release()


def random_particle_image_checks(sz, n):
    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    for i in range(n):
        rframe = np.random.randint(n_frames-1)
        rframe = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, rframe)
        ret, frame = cap.read()

        frame = frame[:, :bottom[sz]]
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        _, binry = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
        binry = 255 - binry

        mask = np.zeros((binry.shape[0] + 2, binry.shape[1] + 2), dtype=np.uint8)
        cv.floodFill(binry, mask, (0, 0), (0,))

        gray[mask[1:-1, 1:-1] == 1] = 255
        circles = find_particle_images_sbd(gray, 5, 10)

        plt.imshow(frame)
        plt.figure()
        plt.imshow(gray)
        plt.show()

        if len(circles) == 0:
            n += 1
            print("no circles found...")
            continue
        x, y, r = circles[np.random.randint(len(circles))]
        rfac = 3
        sample = frame[int(x-rfac*r):int(x+rfac*r), int(y-rfac*r):int(y+rfac*r)]
        plt.imshow(sample)
        plt.title("frame {:d}  |  r = {:.1f} px".format(rframe, r))
        plt.gca().add_artist(plt.Circle((rfac*r, rfac*r), r, edgecolor='r', facecolor='none', linewidth=2))
        plt.show()


def find_particle_images_sbd(img, min_radius, max_radius):
    """ Based on OpenCV's SimpleBlobDetector (https://opencv.org/blog/blob-detection-using-opencv/) """

    params = cv.SimpleBlobDetector_Params()

    # Detect dark (0) / bright (255) blobs
    params.blobColor = 255

    # Thresholds for binarization
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.minArea = int(np.pi * min_radius**2)
    params.maxArea = int(np.pi * max_radius**2)

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    xyr = [[kp.pt[1], kp.pt[0], kp.size/2] for kp in keypoints]
    return xyr


def show_frame(sz: str, n: int):
    path = folder + sz + '/' + sz + '.MOV'
    cap = cv.VideoCapture(path)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    cap.set(cv.CAP_PROP_POS_FRAMES, n)

    _, frame = cap.read()

    plt.imshow(cv.rotate(frame, cv.ROTATE_90_CLOCKWISE))
    plt.title(r"{:.1f} $\mu$m/px".format(ccal[sz]))
    plt.show()


def get_tracks_xyn(sz):
    df = pd.read_csv('particle_tracks/' + sz + '.csv')
    tracks = {}
    for i, row in df.iterrows():
        if row['track_id'] in tracks:
            tracks[row['track_id']].append([row['x'], row['y'], row['n']])
        else:
            tracks[row['track_id']] = [[row['x'], row['y'], row['n']]]
        frac = int(i+1)/len(df)
        nh = int(10*frac)
        print("\r[get_tracks_xyn({:s})]: [".format(sz)+'#'*nh + '-'*(10-nh)+"]", end='')
    print('')
    tracks_lst = []
    for k in tracks:
        tracks_lst.append(np.array(tracks[k]))
    return tracks_lst


def parabolic_peak_fit(x, y):
    """
    Finds optimum by fitting y = ax^2 + bx + c.
    Both <x> and <y> must be a 3-element iterable
    """
    x1, x2, x3 = x
    y1, y2, y3 = y
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
    B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
    C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

    x_max = -B/(2*A)
    y_max = A*x_max**2 + B*x_max + C
    return x_max, y_max


def average_velocity(d):
    def dydt_fc2004(y, t, D):
        """ Ferguson & Church, JSR 2004 """
        R = (2500 - 1000) / 1000
        Rrho = 1 / (R + 1)
        g = 9.81  # m/s^2
        C1 = 18  # -
        C2 = 0.4  # -
        nu = 1e-6  # m^2/s
        Cd = (2 * C1 * nu / np.sqrt(3 * R * g * D ** 3) + np.sqrt(C2)) ** 2
        dydt = np.array([y[1], -0.75 * Rrho * Cd / D * y[1] * np.abs(y[1]) + (Rrho - 1) * g])  # Quadratic drag
        return dydt

    H = 0.1  # m
    dt = 1e-4  # s

    z0 = np.arange(-H, 0, H/100)
    t = np.arange(0, 100, dt)

    v_avg = np.zeros(len(z0))
    for i in range(len(z0)):
        sol = odeint(dydt_fc2004, [z0[i], 0], t, args=(d,))
        sol = sol[sol[:, 0] >= -H, :]
        v = (sol[1:, 1] + sol[:-1, 1]) / 2
        dz = np.diff(sol[:, 0])
        v_avg[i] = 1/(-H - z0[i]) * np.sum(v * dz)

    dz0 = np.diff(z0)
    va = (v_avg[1:] + v_avg[:-1])/2
    return 1/H * np.nansum(va * dz0)


def running_average(arr, n=50, axis=None):
    arr = [np.nanmean(arr[max(0, i - n // 2):min(len(arr), i + n // 2)], axis=axis) for i in range(len(arr))]
    return np.array(arr)


def video_zoomin():
    video_path = '/Users/simenbootsma/Documents/PhD/Work/OperationColdNight/imaging_tests/frozen_tracers/DSC_7239.MOV'
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open file")
        exit()

    N = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    cv.namedWindow('window')
    for n in range(N):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        plt.imshow(frame)
        plt.show()
        frame = frame[1850:2000, 2800:3200]

        cv.imshow('window', frame)
        cv.waitKey(10)
    cv.destroyWindow('window')


if __name__ == '__main__':
    main()

