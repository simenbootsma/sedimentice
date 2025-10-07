import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv

matplotlib.use('Qt5Agg')


def main():
    play_animation()
    # run_animation()
    # run_simplified()
    # run_fixed_interface()
    # run_moving_interface()


def play_animation():
    filepath = 'heat_sim_Fo0_1_N500_infFoice.npy'
    T = np.load(filepath)

    # Nxy = 500
    # h = 1 / Nxy
    # # dt = 0.5 * h**2 / (max(Fo_sed, Fo_ice) * gamma**2)
    # dt = 5e-7
    # dt_show = 0.01
    # dn_show = int(dt_show/dt)
    # tf = 2
    #
    # d = 16e-3  # particle diameter, m
    # gamma = .5
    # L = d/gamma  # domain size, m
    # alpha_ice = 1.2e-6  # thermal diffusivity, m^2/s
    # alpha_glass = 7.6e-7  # thermal diffusivity, m^2/s
    # alpha_stainless_steel = 5e-6  # thermal diffusivity, m^2/s
    # alpha_fake = 2e-7
    # v = 1e-4  # ablation rate, m/s
    # Fo_ice = alpha_ice / (v * d)
    # Fo_sed = alpha_fake / (v * d)
    #
    # Fo = Fo_ice * np.ones((Nxy, Nxy))
    # mask = circle_mask(Fo.shape, int(0.6*gamma*Nxy), Nxy//2, int(gamma*Nxy/2))
    # Fo[mask == 1] = Fo_sed

    plt.ion()
    fig, ax = plt.subplots()
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for i in range(T.shape[-1]):
        ax.clear()
        # jm = int(t * gamma / h)
        # T[:, :jm, ind] = np.where(mask[:, :jm] == 0, np.nan, T[:, :jm, ind])
        ax.imshow(T[:, :, i], vmin=-1, vmax=0, cmap=plt.get_cmap('coolwarm'))
        ax.set_title("t = {:.2f}".format(2*i/T.shape[-1]))
        plt.savefig('heat_sim_Fo0_1_infFoice/img{:04d}.png'.format(i), dpi=600)
        plt.pause(.01)

    plt.ioff()
    plt.show()


def run_animation():
    # Settings
    d = 16e-3  # particle diameter, m
    gamma = .5
    L = d/gamma  # domain size, m
    alpha_ice = 1.2e-6  # thermal diffusivity, m^2/s
    alpha_ice_eff = 1e-8  # effectice diffusivity ice, as most heat is taken up by latent heat
    alpha_glass = 7.6e-7  # thermal diffusivity, m^2/s
    alpha_stainless_steel = 5e-6  # thermal diffusivity, m^2/s
    alpha_fake = 2e-7
    v = 1e-5  # ablation rate, m/s
    Fo_ice = alpha_ice_eff / (v * d)
    Fo_sed = alpha_fake / (v * d)
    Fo_sed = 1
    print("Fo_ice = {:.1f}".format(Fo_ice))
    print("Fo_sed = {:.1f}".format(Fo_sed))
    Nxy = 500
    h = 1 / Nxy
    # dt = 0.5 * h**2 / (max(Fo_sed, Fo_ice) * gamma**2)
    dt = 4e-6
    dt_show = 0.01
    dn_show = int(dt_show/dt)
    tf = 2

    stab = max(Fo_sed, Fo_ice) * gamma**2 * dt / h**2
    colour = "\033[92m" if stab < 0.5 else "\033[93m"
    print(colour + "stability: {:.2f}".format(stab) + "\033[0m")

    # Setup
    Nt = int(tf/dt)+1
    T_show = np.zeros((Nxy, Nxy, int(tf/dt_show)+1))
    Fo = Fo_ice * np.ones((Nxy, Nxy))

    mask = circle_mask(Fo.shape, int(0.6*gamma*Nxy), Nxy//2, int(gamma*Nxy/2))
    Fo[mask == 1] = Fo_sed

    Fo_ip, Fo_im = (Fo[1:-1, 1:-1] + Fo[1:-1, 2:]) / 2, (Fo[1:-1, :-2] + Fo[1:-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp, Fo_jm = (Fo[1:-1, 1:-1] + Fo[2:, 1:-1]) / 2, (Fo[:-2, 1:-1] + Fo[1:-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_top, Fo_im_top = (Fo[0, 1:-1] + Fo[0, 2:]) / 2, (Fo[0, :-2] + Fo[0, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_top, Fo_jm_top = (Fo[0, 1:-1] + Fo[1, 1:-1]) / 2, (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_bot, Fo_im_bot = (Fo[-1, 1:-1] + Fo[-1, 2:]) / 2, (Fo[-1, :-2] + Fo[-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_bot, Fo_jm_bot = (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2, (Fo[-2, 1:-1] + Fo[-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)

    # Run
    T = np.zeros((Nxy, Nxy))
    T[:, :] = -1  # initial condition
    # T[:, 0, 0] = T[:, 1, 0] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left BC
    T[:, 0] = 0
    T_show[:, :, 0] = T.copy()
    pT = T.copy()  # T of previous time step
    for n in range(1, Nt):
        T[1:-1, 1:-1] = pT[1:-1, 1:-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip * (pT[1:-1, 2:] - pT[1:-1, 1:-1])
                                                       - Fo_im * (pT[1:-1, 1:-1] - pT[1:-1, :-2])
                                                       + Fo_jp * (pT[2:, 1:-1] - pT[1:-1, 1:-1])
                                                       - Fo_jm * (pT[1:-1, 1:-1] - pT[:-2, 1:-1]))

        # BC
        T[0, 1:-1] = pT[0, 1:-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_top * (pT[0, 2:] - pT[0, 1:-1])
                                                    - Fo_im_top * (pT[0, 1:-1] - pT[0, :-2])
                                                    + Fo_jp_top * (pT[1, 1:-1] - pT[0, 1:-1])
                                                    - Fo_jm_top * (pT[0, 1:-1] - pT[-1, 1:-1]))  # Top
        T[-1, 1:-1] = pT[-1, 1:-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_bot * (pT[-1, 2:] - pT[-1, 1:-1])
                                                     - Fo_im_bot * (pT[-1, 1:-1] - pT[-1, :-2])
                                                     + Fo_jp_bot * (pT[0, 1:-1] - pT[-1, 1:-1])
                                                     - Fo_jm_bot * (pT[-1, 1:-1] - pT[-2, 1:-1]))  # Bottom
        # T[:, 0, n] = T[:, 1, n] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left

        # 'melt' ice
        jm = int(n * dt * gamma / h)
        T[:, :jm] = np.where(mask[:, :jm] == 0, 0, T[:, :jm])
        # print(jm)

        # T[:, 0, n] = 0
        T[:, -1] = T[:, -2]  # Right

        if n % dn_show == 0:
            T_show[:, :, n//dn_show] = T.copy()
            T_show[:, :jm, n//dn_show] = np.where(mask[:, :jm] == 0, np.nan, T_show[:, :jm, n//dn_show])

        pT = T.copy()
        print("\rsimulating... {:.0f}%".format(n/(Nt-1)*100), end='')

    np.save('heat_sim_Fo0_1_N500_infFoice.npy', T_show)

    plt.imshow(T, vmin=-1, vmax=1)
    plt.title('Close window to start animation')
    plt.show()

    for n in range(0, T_show.shape[-1], T_show.shape[-1]//10):
        plt.plot(T_show[Nxy//2, :, n])
    plt.show()

    plt.ion()
    fig, ax = plt.subplots()
    for t in np.arange(0, tf, dt_show):
        ind = int(t/dt_show)
        ax.clear()
        jm = int(t * gamma / h)
        Ti = T_show[:, :, ind]
        Ti[:, :jm] = np.where(mask[:, :jm] == 0, np.nan, Ti[:, :jm])
        ax.imshow(Ti, vmin=-1, vmax=0, cmap=plt.get_cmap('coolwarm'))
        ax.set_title("t = {:.2f}".format(t))
        plt.pause(.01)

    plt.ioff()

    # plt.figure()
    # plt.imshow(mask)
    plt.show()



def run_simplified():
    # Settings
    gamma = .3
    alpha_ice = 1.2e-6  # thermal diffusivity, m^2/s
    alpha_glass = 7.6e-7  # thermal diffusivity, m^2/s
    alpha_stainless_steel = 5e-6  # thermal diffusivity, m^2/s
    v = 1e-4
    L = 1
    Fo_ice = alpha_ice / (v * gamma * L)
    Fo_sed = alpha_glass / (v * gamma * L)
    Ste = 5
    h = 5e-3
    dt = 2e-4
    tf = 1
    phi = 0.6
    q = 1e1

    # Setup
    Nx, Ny, Nt = int(1/h)+1, int(1/h)+1, int(tf/dt)+1
    T = np.zeros((Ny, Nx, Nt))
    Fo = Fo_ice * np.ones((Ny, Nx))

    xs, ys = slice(0, int(gamma * Nx)), slice(int((1-gamma)/2 * Ny), int((1+gamma)/2 * Ny))
    Fo[ys, xs] = Fo_sed
    # Fo[mask == 1] = Fo_sed

    Fo_ip, Fo_im = (Fo[1:-1, 1:-1] + Fo[1:-1, 2:]) / 2, (Fo[1:-1, :-2] + Fo[1:-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp, Fo_jm = (Fo[1:-1, 1:-1] + Fo[2:, 1:-1]) / 2, (Fo[:-2, 1:-1] + Fo[1:-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_top, Fo_im_top = (Fo[0, 1:-1] + Fo[0, 2:]) / 2, (Fo[0, :-2] + Fo[0, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_top, Fo_jm_top = (Fo[0, 1:-1] + Fo[1, 1:-1]) / 2, (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_bot, Fo_im_bot = (Fo[-1, 1:-1] + Fo[-1, 2:]) / 2, (Fo[-1, :-2] + Fo[-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_bot, Fo_jm_bot = (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2, (Fo[-2, 1:-1] + Fo[-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)

    # Run
    T[:, :, 0] = -1  # initial condition
    # T[:, 0, 0] = T[:, 1, 0] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left BC
    T[:, 0, 0] = 0
    for n in range(1, Nt):

        T[1:-1, 1:-1, n] = T[1:-1, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip * (T[1:-1, 2:, n - 1] - T[1:-1, 1:-1, n - 1])
                                                       - Fo_im * (T[1:-1, 1:-1, n - 1] - T[1:-1, :-2, n - 1])
                                                       + Fo_jp * (T[2:, 1:-1, n - 1] - T[1:-1, 1:-1, n - 1])
                                                       - Fo_jm * (T[1:-1, 1:-1, n - 1] - T[:-2, 1:-1, n - 1]))

        # BC
        T[0, 1:-1, n] = T[0, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_top * (T[0, 2:, n - 1] - T[0, 1:-1, n - 1])
                                                    - Fo_im_top * (T[0, 1:-1, n - 1] - T[0, :-2, n - 1])
                                                    + Fo_jp_top * (T[1, 1:-1, n - 1] - T[0, 1:-1, n - 1])
                                                    - Fo_jm_top * (T[0, 1:-1, n - 1] - T[-1, 1:-1, n - 1]))  # Top
        T[-1, 1:-1, n] = T[-1, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_bot * (T[-1, 2:, n - 1] - T[-1, 1:-1, n - 1])
                                                     - Fo_im_bot * (T[-1, 1:-1, n - 1] - T[-1, :-2, n - 1])
                                                     + Fo_jp_bot * (T[0, 1:-1, n - 1] - T[-1, 1:-1, n - 1])
                                                     - Fo_jm_bot * (T[-1, 1:-1, n - 1] - T[-2, 1:-1, n - 1]))  # Bottom
        # T[:, 0, n] = T[:, 1, n] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left
        T[:, 0, n] = 0
        T[:, -1, n] = T[:, -2, n]  # Right
        print("\rsimulating... {:.0f}%".format(n/(Nt-1)*100), end='')

    plt.imshow(T[:, :, 0], vmin=-1, vmax=1)
    plt.title('Close window to start animation')
    plt.show()

    for n in range(0, Nt, Nt//10):
        plt.plot(T[Ny//2, :, n])
    plt.show()

    plt.ion()
    fig, ax = plt.subplots()
    dt_show = 0.01
    for t in np.arange(0, tf, dt_show):
        ind = int(t/dt)
        ax.clear()
        ax.imshow(T[:, :, ind], vmin=-1, vmax=0, cmap=plt.get_cmap('coolwarm'))
        ax.set_title("t = {:.2f}".format(t))
        plt.pause(.01)

    plt.ioff()

    # plt.figure()
    # plt.imshow(mask)
    plt.show()


def run_fixed_interface():
    # Settings
    gamma = .1
    alpha_ice = 1.2e-6  # thermal diffusivity, m^2/s
    alpha_glass = 7.6e-7  # thermal diffusivity, m^2/s
    alpha_stainless_steel = 5e-6  # thermal diffusivity, m^2/s
    v = 1e-4
    L = 1
    Fo_ice = alpha_ice / (v * gamma * L)
    Fo_sed = alpha_stainless_steel / (v * gamma * L)
    Ste = 5
    h = 5e-3
    dt = 2e-4
    tf = 1
    phi = 0.6
    q = 1e1

    # Setup
    Nx, Ny, Nt = int(1/h)+1, int(1/h)+1, int(tf/dt)+1
    T = np.zeros((Ny, Nx, Nt))
    Fo = Fo_ice * np.ones((Ny, Nx))

    # Fo[Ny//3:2*Ny//3, Nx//10:3*Nx//10] = Fo_sed
    mask = circles_mask((Ny, Nx), gamma / h / 2, phi)
    Fo[mask == 1] = Fo_sed

    Fo_ip, Fo_im = (Fo[1:-1, 1:-1] + Fo[1:-1, 2:]) / 2, (Fo[1:-1, :-2] + Fo[1:-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp, Fo_jm = (Fo[1:-1, 1:-1] + Fo[2:, 1:-1]) / 2, (Fo[:-2, 1:-1] + Fo[1:-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_top, Fo_im_top = (Fo[0, 1:-1] + Fo[0, 2:]) / 2, (Fo[0, :-2] + Fo[0, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_top, Fo_jm_top = (Fo[0, 1:-1] + Fo[1, 1:-1]) / 2, (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_bot, Fo_im_bot = (Fo[-1, 1:-1] + Fo[-1, 2:]) / 2, (Fo[-1, :-2] + Fo[-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_bot, Fo_jm_bot = (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2, (Fo[-2, 1:-1] + Fo[-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)

    # Run
    T[:, :, 0] = -1  # initial condition
    # T[:, 0, 0] = T[:, 1, 0] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left BC
    T[:, 0, 0] = 0
    for n in range(1, Nt):

        T[1:-1, 1:-1, n] = T[1:-1, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip * (T[1:-1, 2:, n - 1] - T[1:-1, 1:-1, n - 1])
                                                       - Fo_im * (T[1:-1, 1:-1, n - 1] - T[1:-1, :-2, n - 1])
                                                       + Fo_jp * (T[2:, 1:-1, n - 1] - T[1:-1, 1:-1, n - 1])
                                                       - Fo_jm * (T[1:-1, 1:-1, n - 1] - T[:-2, 1:-1, n - 1]))

        # BC
        T[0, 1:-1, n] = T[0, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_top * (T[0, 2:, n - 1] - T[0, 1:-1, n - 1])
                                                    - Fo_im_top * (T[0, 1:-1, n - 1] - T[0, :-2, n - 1])
                                                    + Fo_jp_top * (T[1, 1:-1, n - 1] - T[0, 1:-1, n - 1])
                                                    - Fo_jm_top * (T[0, 1:-1, n - 1] - T[-1, 1:-1, n - 1]))  # Top
        T[-1, 1:-1, n] = T[-1, 1:-1, n-1] + gamma ** 2 * dt / h ** 2 * (Fo_ip_bot * (T[-1, 2:, n - 1] - T[-1, 1:-1, n - 1])
                                                     - Fo_im_bot * (T[-1, 1:-1, n - 1] - T[-1, :-2, n - 1])
                                                     + Fo_jp_bot * (T[0, 1:-1, n - 1] - T[-1, 1:-1, n - 1])
                                                     - Fo_jm_bot * (T[-1, 1:-1, n - 1] - T[-2, 1:-1, n - 1]))  # Bottom
        # T[:, 0, n] = T[:, 1, n] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left
        T[:, 0, n] = 0
        T[:, -1, n] = T[:, -2, n]  # Right
        print("\rsimulating... {:.0f}%".format(n/(Nt-1)*100), end='')

    plt.imshow(T[:, :, 0], vmin=-1, vmax=1)
    plt.title('Close window to start animation')
    plt.show()

    for n in range(0, Nt, Nt//10):
        plt.plot(T[Ny//2, :, n])
    plt.show()

    plt.ion()
    fig, ax = plt.subplots()
    dt_show = 0.01
    for t in np.arange(0, tf, dt_show):
        ind = int(t/dt)
        ax.clear()
        ax.imshow(T[:, :, ind], vmin=-1, vmax=0, cmap=plt.get_cmap('coolwarm'))
        ax.set_title("t = {:.2f}".format(t))
        plt.pause(.01)

    plt.ioff()

    plt.figure()
    plt.imshow(mask)
    plt.show()


def run_moving_interface():
    # Settings
    gamma = .1
    alpha_ice = 1.2e-6  # thermal diffusivity, m^2/s
    alpha_glass = 7.6e-7  # thermal diffusivity, m^2/s
    alpha_stainless_steel = 5e-6  # thermal diffusivity, m^2/s
    v = 1e-4
    Fo_ice = alpha_ice / (v * gamma)
    Fo_sed = alpha_stainless_steel / (v * gamma)
    Ste = 5
    h = 5e-3
    dt = 1e-4
    tf = 2
    phi = 0
    q = 1e1

    CFL = max(Fo_sed, Fo_ice) * gamma ** 2 * dt / h ** 2
    if CFL > 0.5: print("warning: CFL larger than 0.5 ({:.2e})".format(CFL))

    # Setup
    Nx, Ny, Nt = int(1/h)+1, int(1/h)+1, int(tf/dt)+1
    T = np.zeros((Ny, Nx, Nt))
    Fo = Fo_ice * np.ones((Ny, Nx))

    # Fo[Ny//3:2*Ny//3, Nx//10:3*Nx//10] = Fo_sed
    Fo[circle_mask(Fo.shape, int(0.1/h), int(0.5/h), 0.05/h) == 1] = Fo_sed

    Fo_ip, Fo_im = (Fo[1:-1, 1:-1] + Fo[1:-1, 2:]) / 2, (Fo[1:-1, :-2] + Fo[1:-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp, Fo_jm = (Fo[1:-1, 1:-1] + Fo[2:, 1:-1]) / 2, (Fo[:-2, 1:-1] + Fo[1:-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_top, Fo_im_top = (Fo[0, 1:-1] + Fo[0, 2:]) / 2, (Fo[0, :-2] + Fo[0, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_top, Fo_jm_top = (Fo[0, 1:-1] + Fo[1, 1:-1]) / 2, (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)
    Fo_ip_bot, Fo_im_bot = (Fo[-1, 1:-1] + Fo[-1, 2:]) / 2, (Fo[-1, :-2] + Fo[-1, 1:-1]) / 2  # Fo_(i+1/2), Fo_(i-1/2)
    Fo_jp_bot, Fo_jm_bot = (Fo[-1, 1:-1] + Fo[0, 1:-1]) / 2, (Fo[-2, 1:-1] + Fo[-1, 1:-1]) / 2  # Fo_(j+1/2), Fo_(j-1/2)

    # Run
    T[:, :, 0] = -1  # initial condition
    T[:, 0, 0] = T[:, 1, 0] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left BC
    # T[:, 0, 0] = 0
    for n in range(1, Nt):
        i_int = int(gamma * n*dt / h)  # interface location
        T[1:-1, i_int+1:-1, n] = (T[1:-1, i_int+1:-1, n-1] + gamma ** 2 * dt / h ** 2 *
                                  (Fo_ip[:, i_int:] * (T[1:-1, i_int+2:, n - 1] - T[1:-1, i_int+1:-1, n - 1])
                                    - Fo_im[:, i_int:] * (T[1:-1, i_int+1:-1, n - 1] - T[1:-1, i_int:-2, n - 1])
                                                       + Fo_jp[:, i_int:] * (T[2:, i_int+1:-1, n - 1] - T[1:-1, i_int+1:-1, n - 1])
                                                       - Fo_jm[:, i_int:] * (T[1:-1, i_int+1:-1, n - 1] - T[:-2, i_int+1:-1, n - 1])))

        # BC
        T[0, i_int+1:-1, n] = (T[0, i_int+1:-1, n-1] + gamma ** 2 * dt / h ** 2 *
                         (Fo_ip_top[i_int:] * (T[0, i_int+2:, n - 1] - T[0, i_int+1:-1, n - 1])
                                                    - Fo_im_top[i_int:] * (T[0, i_int+1:-1, n - 1] - T[0, i_int:-2, n - 1])
                                                    + Fo_jp_top[i_int:] * (T[1, i_int+1:-1, n - 1] - T[0, i_int+1:-1, n - 1])
                                                    - Fo_jm_top[i_int:] * (T[0, i_int+1:-1, n - 1] - T[-1, i_int+1:-1, n - 1])))  # Top
        T[-1, i_int+1:-1, n] = (T[-1, i_int+1:-1, n-1] + gamma ** 2 * dt / h ** 2 *
                          (Fo_ip_bot[i_int:] * (T[-1, i_int+2:, n - 1] - T[-1, i_int+1:-1, n - 1])
                                                     - Fo_im_bot[i_int:] * (T[-1, i_int+1:-1, n - 1] - T[-1, i_int:-2, n - 1])
                                                     + Fo_jp_bot[i_int:] * (T[0, i_int+1:-1, n - 1] - T[-1, i_int+1:-1, n - 1])
                                                     - Fo_jm_bot[i_int:] * (T[-1, i_int+1:-1, n - 1] - T[-2, i_int+1:-1, n - 1])))  # Bottom
        T[:, i_int, n] = T[:, 1, n] + (1-phi)/(gamma**2 * Fo_ice * Ste) * h  # Left
        # T[:, i_int, n] = 0
        T[:, -1, n] = T[:, -2, n]  # Right
        T[:, :i_int, n] = np.nan  # water
        print("\rsimulating... {:.0f}%".format(n/(Nt-1)*100), end='')

    plt.imshow(T[:, :, 0], vmin=-1, vmax=1)
    plt.title('Close window to start animation')
    plt.show()

    for n in range(0, Nt, Nt//10):
        plt.plot(T[Ny//2, :, n])
    plt.show()

    plt.ion()
    fig, ax = plt.subplots()
    dt_show = 0.01
    for t in np.arange(0, tf, dt_show):
        ind = int(t/dt)
        ax.clear()
        ax.imshow(T[:, :, ind], vmin=-1, vmax=0, cmap=plt.get_cmap('coolwarm'))
        ax.set_title("t = {:.2f}".format(t))
        plt.pause(.01)

    plt.ioff()
    plt.show()


def circle_mask(shape, i, j, r):
    mask = np.zeros(shape, dtype=np.uint8)
    for ii in range(shape[1]):
        for jj in range(shape[0]):
            if (ii - i)**2 + (jj - j)**2 <= r**2:
                mask[jj, ii] = 1
    return mask


def circles_mask(shape, r, phi):
    n_circ = np.prod(shape) * phi / (np.pi * r**2)
    a = 2/np.sqrt(3)
    nx = 1/(4*a)*(1 + a + np.sqrt(a**2 + (8*n_circ - 6)*a + 1))
    ny = int(a * nx)
    nx = int(nx)

    dx = shape[1] / nx
    dy = shape[0] / ny

    grid1 = [[(i+0.5)*dx, (j+0.5)*dy] for i in np.arange(nx) for j in np.arange(ny)]
    grid2 = [[(i + 1) * dx, (j + 1) * dy] for i in np.arange(nx-1) for j in np.arange(ny-1)]

    # points = np.array([[x, y + r*(i%2)] for i, x in enumerate(np.arange(np.sqrt(3)*r, shape[1], np.sqrt(3)*r))
    #                                     for y in np.arange(r, shape[0]-r, 2*r)])
    points = np.array(grid1 + grid2)
    mask = np.zeros(shape, dtype=np.uint8)
    for ii in range(shape[1]):
        for jj in range(shape[0]):
            if np.any((ii - points[:, 0])**2 + (jj - points[:, 1])**2 <= r**2):
                mask[jj, ii] = 1
    mask[:, :-17] = mask[:, 17:]
    mask[:, -17:] = 0
    return mask


if __name__ == "__main__":
    main()

