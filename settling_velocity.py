import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def main():
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

    plt.show()


def test():
    y0 = np.array([0, 0])
    t = np.linspace(0, 1, 100)
    d = 1e-3
    sol = odeint(dydt_quad, y0, t, args=(d,))

    plt.plot(t, sol[:, 1])
    plt.show()


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
    Rrho = 1 / 2500
    Cd = 0.44
    g = 9.81
    dydt = np.array([y[1], -0.75*Rrho*Cd/D * y[1] * np.abs(y[1]) + (Rrho - 1)*g])  # Quadratic drag
    return dydt


def dydt_stokes(y, t, D):
    Rrho = 1000 / 2200
    g = 9.81
    nu = 1e-6
    dydt = np.array([y[1], -18 * Rrho * nu / D**2 * y[1] + (Rrho - 1) * g])  # Stokes' drag
    return dydt


if __name__ == '__main__':
    main()


