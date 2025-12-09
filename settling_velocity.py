import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib
import time

matplotlib.use('Qt5Agg')


def main():
    d_arr = np.logspace(-6, -1, 50)
    z_arr = -np.logspace(-6, -1, 50)

    num_m = np.zeros((len(d_arr), len(z_arr)))
    ana_m = np.zeros((len(d_arr), len(z_arr)))

    for i in range(len(d_arr)):
        for j in range(len(z_arr)):
            num_m[i, j] = -settling_velocity_at_z(d_arr[i], 2500, z_arr[j])
            ana_m[i, j] = -analytical_avg_vel(d_arr[i], 2500, z_arr[j])

    plt.imshow((ana_m-num_m)/num_m*100, cmap='coolwarm', vmin=-1, vmax=1)
    cb = plt.colorbar()
    cb.set_label('error (%)')
    plt.show()

    analytical_test()

    diameter = np.logspace(-5, -1, 100)
    yterm_quad = [distance_until_terminal_velocity(d, dydt_quad) for d in diameter]
    yterm_stokes = [distance_until_terminal_velocity(d, dydt_stokes) for d in diameter]
    tterm_quad = [time_until_terminal_velocity(d, dydt_quad) for d in diameter]
    tterm_stokes = [time_until_terminal_velocity(d, dydt_stokes) for d in diameter]

    sizes = [55e-6, 90e-6, 120e-6, 200e-6, 250e-6, 375e-6, 875e-6, 1e-3, 2e-3, 3e-3, 4e-3, 8e-3, 16e-3]

    plt.figure()
    plt.loglog(diameter, tterm_quad, label='quad')
    plt.loglog(diameter, tterm_stokes, label='stokes')

    for s in sizes:
        ind = np.argmin(np.abs(diameter - s))
        plt.plot(s, min(tterm_quad[ind], tterm_stokes[ind]), 'ok')

    plt.ylabel('Time to terminal velocity (s)')
    plt.xlabel('Diameter (m)')
    plt.ylim([1e-3, 1e0])
    plt.legend()

    plt.figure()
    plt.loglog(diameter, yterm_quad, label='quad')
    plt.loglog(diameter, yterm_stokes, label='stokes')

    yterms = []
    for s in sizes:
        ind = np.argmin(np.abs(diameter - s))
        yterms.append(min(yterm_quad[ind], yterm_stokes[ind]))
        plt.plot(s, yterms[-1], 'ok')

    plt.ylabel('Distance to terminal velocity (m)')
    plt.xlabel('Diameter (m)')
    plt.ylim([1e-6, 1e3])
    plt.legend()

    plt.figure()
    plt.plot(sizes, np.array(yterms), 'o')
    plt.plot(diameter, 0.1 * np.ones(diameter.size), '--k')
    plt.xlabel('Particle diameter (m)', fontsize=14)
    plt.ylabel(r'Distance to $v=0.95 v_t$ (m)', fontsize=14)
    plt.text(1.5e-5, .108, 'Cylinder height', fontsize=12)
    plt.xscale('log')
    plt.xlim([np.min(diameter), np.max(diameter)])
    plt.ylim([0, .2])
    plt.yticks([0, .05, .1, .15, .2])
    plt.tick_params(labelsize=12)

    plt.figure()
    va = [average_velocity(d, dydt_fc2004) for d in diameter]
    vt = [terminal_velocity(d, dydt_fc2004) for d in diameter]

    plt.plot(diameter, np.abs(va), label='average over H')
    plt.plot(diameter, np.abs(vt), label='terminal')
    plt.xscale('log')

    plt.show()


def analytical_test():
    y0 = np.array([0, 0])
    t = np.linspace(0, 10, 1000)
    d = 1e-3
    sol = odeint(dydt_fc2004, y0, t, args=(d,))
    v_ana = analytical_solution_fc2004(d, t)

    plt.plot(t, sol[:, 1])
    plt.plot(t, v_ana)
    plt.show()


def analytical_solution_fc2004(dp, t, v0=0):
    R = (2500 - 1000) / 1000
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*dp**3) + np.sqrt(C2))**2
    a = 3*Cd/(4*(R+1)*dp)
    b = R/(R+1)*g
    c = np.arctanh(-v0*np.sqrt(a/b))
    return -np.sqrt(b/a) * np.tanh(np.sqrt(a*b)*(t + c))


def terminal_velocity(d, func):
    t = np.linspace(0, 10, 1000)
    sol = odeint(func, [0, 0], t, args=(d,))
    return sol[-1, 1]


def average_velocity(d, func):
    H = 0.1  # m
    dt = 1e-4  # s

    z0 = np.arange(-H, 0, H/100)
    t = np.arange(0, 100, dt)

    v_avg = np.zeros(len(z0))
    for i in range(len(z0)):
        sol = odeint(func, [z0[i], 0], t, args=(d,))
        sol = sol[sol[:, 0] >= -H, :]
        v = (sol[1:, 1] + sol[:-1, 1]) / 2
        dz = np.diff(sol[:, 0])
        v_avg[i] = 1/(-H - z0[i]) * np.sum(v * dz)
    #     plt.plot(t[:len(sol)], sol[:, 0])
    # plt.show()

    dz0 = np.diff(z0)
    va = (v_avg[1:] + v_avg[:-1])/2
    return 1/H * np.nansum(va * dz0)


def time_until_terminal_velocity(d, func):
    dt = 1e-4
    t = np.arange(0, 100, dt)
    sol = odeint(func, [0, 0], t, args=(d,))
    v = sol[:, 1]
    return t[v < .95 * np.min(v)][0]


def distance_until_terminal_velocity(d, func):
    dt = 1e-4
    t = np.arange(0, 100, dt)
    sol = odeint(func, [0, 0], t, args=(d,))
    v = sol[:, 1]
    return np.abs(np.sum(v[v > .95 * np.min(v)]) * dt)


def dydt_quad(y, t, D):
    # Rrho = 1000 / 2200
    Rrho = 1000 / 2500
    Cd = 0.44
    g = 9.81
    dydt = np.array([y[1], -0.75*Rrho*Cd/D * y[1] * np.abs(y[1]) + (Rrho - 1)*g])  # Quadratic drag
    return dydt


def dydt_stokes(y, t, D):
    Rrho = 1000 / 2500
    g = 9.81
    nu = 1e-6
    dydt = np.array([y[1], -18 * Rrho * nu / D**2 * y[1] + (Rrho - 1) * g])  # Stokes' drag
    return dydt


def dydt_fc2004(y, t, D):
    R = (2500 - 1000) / 1000
    Rrho = 1 / (R + 1)
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*D**3) + np.sqrt(C2))**2
    dydt = np.array([y[1], -0.75*Rrho*Cd/D * y[1] * np.abs(y[1]) + (Rrho - 1)*g])  # Quadratic drag
    return dydt


def settling_velocity_at_z(d_p, rho_p, z, v0=0.):
    """ Here we assume the amount of particles falling to be the same at any height z0"""
    z = np.abs(z)
    z0 = np.arange(-z, z/1000, z/1000)
    t = np.logspace(-7, 4, 10000)
    dt = np.hstack((t[0], np.diff(t)))
    rho_f = 1000

    # v = velocity_fc2004(t, d_p, rho_f, rho_p, v0)
    sol = odeint(dydt_fc2004, [0, 0], t, args=(d_p,))
    v = sol[:, 1]

    v_avg = np.zeros(len(z0))
    for i in range(len(z0)):
        # I = np.cumsum(v*dt) >= (-z-z0[i])
        I = sol[:, 0] >= (-z-z0[i])
        if np.any(I):
            v_avg[i] = v[I][-1]
    v_avg[np.isinf(v_avg)] = np.nan

    dz0 = np.diff(z0)
    va = (v_avg[1:] + v_avg[:-1])/2
    return 1/z * np.nansum(va * dz0)


def velocity_fc2004(t, dp, rho_f, rho_p, v0=0.):
    R = (rho_p - rho_f) / rho_f
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*dp**3) + np.sqrt(C2))**2
    a = 3*Cd/(4*(R+1)*dp)
    b = R/(R+1)*g
    arg = -v0*np.sqrt(a/b)
    if arg < -1 or arg > 1:
        return np.nan
    c = np.arctanh(arg) / np.sqrt(a*b)
    return -np.sqrt(b/a) * np.tanh(np.sqrt(a*b)*(t + c))


def analytical_avg_vel(dp, rho_p, z, v0=0.):
    z = -np.abs(z)
    rho_f = 1000
    R = (rho_p - rho_f) / rho_f
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*dp**3) + np.sqrt(C2))**2
    a = 3*Cd/(4*(R+1)*dp)
    b = R/(R+1)*g
    arg = -v0*np.sqrt(a/b)
    if arg < -1 or arg > 1:
        return np.nan
    c = np.arctanh(arg) / np.sqrt(a*b)
    d = c*np.sqrt(a*b)
    if -a*z > 50:
        # prevent overflow in exponent
        ach = -a*z + np.log(2*np.cosh(d))
    else:
        ach = np.arccosh(np.exp(-a*z)*np.cosh(d))
    wbar = np.sqrt(b/a)/(a*z) * (ach - np.tanh(ach) - np.tanh(d) - d)
    return wbar


if __name__ == '__main__':
    main()


