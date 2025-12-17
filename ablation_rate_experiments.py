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
import cv2 as cv
from PIL import Image
matplotlib.use('Qt5Agg')

# estimated melt rates in mm/s from Ward2024 (preprint)
data_Ward2024 = {
    'd': np.array([0.04, 0.15, 0.5, 2.5, 8, 16]),
    'w10': np.array([0.2097552159211107, 0.2764181435184176, 0.16321301604326208, 0.11701515238853961, 0.09250510070084779, np.nan]),  # W = 10 mm
    'w20': np.array([0.26012808614378413, 0.31543715747439466, 0.17412233383086945, 0.12142247131676466, 0.11321540549425399, 0.09535760401729254,]),  # W = 20 mm
    'w30': np.array([0.2368488985381059, 0.2992087767511801, 0.22825190143916674, 0.1261622704847849, 0.11825780314137402, 0.10445380875960027]),  # W = 30 mm
}


def main():
    # test_process_image_data()
    all_cases = get_all_ids()
    cases = all_cases
    # print("\n".join([str(tup) for tup in list(enumerate(cases))]))
    # return
    for case in cases:
        params = get_param(case)
        if params['images_available'] and (not params['processed']):
            contours, times = process_case(case)

            with open('contours/contours_'+case+'.pkl', 'wb') as f:
                pickle.dump([contours, times], f)

    # process_force_data()
    # plot_results(fresh_water=True)


def animate_contours(case, slow=False, start_n=0):
    with open('contours/contours_'+case+'.pkl', 'rb') as f:
        contours, times = pickle.load(f)

    fig, ax = plt.subplots()
    plt.ion()
    pause_time = 0.5 if slow else 0.01
    for n in range(start_n, len(contours)):
        c = contours[n]
        ax.clear()
        ax.plot(c[:, 0], -c[:, 1], '-k')

        ax.set_aspect('equal')
        ax.set_xlim([0, 3000])
        ax.set_ylim([-3000, 0])
        ax.set_title("{:s}  |  n = {:d}  |  t = {:.0f} s".format(case, n, times[n]-times[0]))
        plt.pause(pause_time)
    plt.ioff()
    plt.show()


def test_process_image_data():
    data_folder = '/Volumes/Melting001/SedimentIce/MSc_NynkeNell/data/Images/'

    folders = sorted(glob(data_folder + '*'))
    folders.remove(data_folder + 'resolution')

    all_cases = get_all_ids()
    all_cases = all_cases
    for case in all_cases:
        params = get_param(case)

        if not params['images_available']:
            continue

        files = sorted(glob(data_folder + params['folder'] + '/*.JPG'))

        img = plt.imread(files[0])
        img = rotate_image(img, params['rotation'])
        gray = np.mean(img, axis=-1)

        # crop
        ymin, ymax = int(params['crop_ymin']), int(params['crop_ymax'])
        xmin, xmax = int(params['crop_xmin']), int(params['crop_xmax'])
        gray = gray[ymin:ymax, xmin:xmax]

        # enhance contrast
        val99 = np.sort(gray.flatten())[int(gray.size*0.99)]
        gray = gray / val99 * 250
        gray[gray > 255] = 255
        gray = gray.astype(np.uint8)

        # binarize
        binry = 255 * np.where(gray < params['bin_thresh'], 1, 0).astype(np.uint8)

        if params['dp'] < 500:
            binry = cv.morphologyEx(binry, cv.MORPH_CLOSE, np.ones((3, 3)))

        if 'clear' in case and binry[0, 0] > 0:
            _, binry, _, _ = cv.floodFill(binry, None, [0, 0], 0)

        edges = find_edges(binry, largest_only=True, remove_outside=True)

        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
        ax[0].imshow(gray)
        ax[1].imshow(binry)
        ax[2].imshow(gray)
        ax[2].plot(edges[:, 0], edges[:, 1], '-r')

        for a in ax:
            a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        plt.suptitle(case)
        plt.show()


def fix_contour(case, n, bt=None):
    data_folder = '/Volumes/Melting001/SedimentIce/MSc_NynkeNell/data/Images/'
    folders = sorted(glob(data_folder + '*'))
    folders.remove(data_folder + 'resolution')
    params = get_param(case)
    files = sorted(glob(data_folder + params['folder'] + '/*.JPG'))
    fn = files[n]

    if bt is not None:
        params['bin_thresh'] = bt

    with open('contours/contours_'+case+'.pkl', 'rb') as f:
        contours, times = pickle.load(f)

    edges = process_file(fn, params, debug=True)

    plt.figure()
    plt.plot(contours[n][:, 0], contours[n][:, 1], label='before')
    plt.plot(edges[:, 0], edges[:, 1], label='after')
    plt.legend()
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    plt.show()

    if input("Save contour? (y/n)").lower() == 'y':
        contours[n] = edges
        with open('contours/contours_' + case + '.pkl', 'wb') as f:
            pickle.dump([contours, times], f)
        print("Saved!")


def process_file(fn, params, debug=False):
    img = plt.imread(fn)
    img = rotate_image(img, params['rotation'])
    gray = np.mean(img, axis=-1)

    # crop
    ymin, ymax = int(params['crop_ymin']), int(params['crop_ymax'])
    xmin, xmax = int(params['crop_xmin']), int(params['crop_xmax'])
    gray = gray[ymin:ymax, xmin:xmax]

    # enhance contrast
    val99 = np.sort(gray.flatten())[int(gray.size * 0.99)]
    gray = gray / val99 * 250
    gray[gray > 255] = 255
    gray = gray.astype(np.uint8)

    # binarize
    binry = 255 * np.where(gray < params['bin_thresh'], 1, 0).astype(np.uint8)

    if params['dp'] < 500:
        binry = cv.morphologyEx(binry, cv.MORPH_CLOSE, np.ones((3, 3)))

    if 'clear' in params.name and binry[0, 0] > 0:
        _, binry, _, _ = cv.floodFill(binry, None, [0, 0], 0)

    edges = find_edges(binry, largest_only=True, remove_outside=True)

    if debug:
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 5))
        ax[0].imshow(gray)
        ax[1].imshow(binry)
        ax[2].imshow(gray)
        ax[2].plot(edges[:, 0], edges[:, 1], '-r')

        for a in ax:
            a.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
        plt.suptitle(params.name + " | bin_thresh = {:d}".format(params['bin_thresh']))
        plt.show()
    return edges


def process_case(case):
    data_folder = '/Volumes/Melting001/SedimentIce/MSc_NynkeNell/data/Images/'

    folders = sorted(glob(data_folder + '*'))
    folders.remove(data_folder + 'resolution')

    params = get_param(case)

    files = sorted(glob(data_folder + params['folder'] + '/*.JPG'))
    contours = []
    times = []
    for fn in files:
        edges = process_file(fn, params)
        contours.append(edges)
        times.append(get_time(fn))

        print("\r[process_case({:s})] {:.1f}%".format(case, len(times)/len(files)*100), end='')
    print(" \033[42mdone\033[0m")
    return contours, times


def process_force_data(pick_start_end=False):
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
    ward_marker = {'markersize': 10, 'linestyle': '', 'mec': 'k', 'mew': 1.5}
    gprime = 9.81 * (2500-1000)/1000
    nu = 1e-6  # kinematic viscosity [m2/s]

    data_means = data.groupby('dp').mean(numeric_only=True)
    data_std = data.groupby('dp').std(numeric_only=True)
    dp = data_means.index.values
    data_means['Ga'] = np.sqrt(gprime * (dp*1e-6)**3) / nu
    Ga_err = np.abs(np.sqrt(gprime * (np.vstack([dp - data_means['dp_err'].values, dp + data_means['dp_err'].values])*1e-6)**3)/nu - data_means['Ga'].values)
    data['Ga'] = np.sqrt(gprime * (data['dp']*1e-6)**3) / nu

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

    Ga_Ward = np.sqrt(gprime * (data_Ward2024['d'] * 1e-3)**3) / nu
    ax.plot(Ga_Ward, data_Ward2024['w10'], 's', color="#EA33F7", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w20'], 'o', color="#0000F5", **ward_marker)
    ax.plot(Ga_Ward, data_Ward2024['w30'], '^', color="#EA3323", **ward_marker)

    ax.plot(data['Ga'], -data['drdt'] * 1e3, **single_marker)
    ax.errorbar(data_means['Ga'], -data_means['drdt'] * 1e3, xerr=Ga_err, **mean_marker)
    ax.plot([1e-2, 1e9], [-1e3*np.mean(drdt_clear), -1e3*np.mean(drdt_clear)], color='#024D80', lw=1.5)
    ax.text(3e-1, 1.1*-1e3*np.mean(drdt_clear), 'clear ice', color='#024D80', fontsize=14)
    plt.xlabel('Ga', fontsize=16)
    plt.ylabel('$\dot{R}$ [mm/s]', fontsize=16)
    # plt.ylim(top=0.5)
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.xlim([2e-1, 1.5e4])
    plt.ylim([0, None])

    plt.show()


def get_param(case, name=None):
    param = pd.read_csv('process_parameters.csv', index_col='ID').loc[case]
    if name is None:
        return param
    return param[name]


def get_all_ids():
    df = pd.read_csv('process_parameters.csv')
    return df['ID'].values


def rotate_image(img, angle):
    # angle (degrees) --- negative: clockwise, positive: counter-clockwise

    rot90 = np.round(angle/90)
    if rot90 != 0:
        rc = cv.ROTATE_90_CLOCKWISE if rot90 < 0 else cv.ROTATE_90_COUNTERCLOCKWISE
        for _ in range(int(abs(rot90))):
            img = cv.rotate(img, rc)
    angle -= rot90 * 90

    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv.warpAffine(img, M, (cols, rows))
    return img


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


def get_time(filepath):
    """ Time in seconds after midnight at which the photo was taken """
    if '.tif' in filepath.lower():
        with Image.open(filepath) as img:
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
            t_str = meta_dict['DateTime'][0].split(' ')[1]
    elif '.nef' in filepath.lower():
        exif = Image.open(filepath).getexif()
        t_str = exif[36867].split(' ')[1]
    elif '.jpg' in filepath.lower() or '.jpeg' in filepath.lower():
        exif = Image.open(filepath)._getexif()
        t_str = exif[36867].split(' ')[1]
    else:
        raise ValueError("[get_time]: Unknown filetype")
    t = np.sum([int(s) * 60 ** (2 - i) for i, s in enumerate(t_str.split(':'))])
    return t


if __name__ == "__main__":
    main()

