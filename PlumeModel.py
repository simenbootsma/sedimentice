import numpy as np
from scipy.integrate import odeint


class PlumeModel:
    def __init__(self, T_inf, S_inf, b0, w0, phi0, alpha, w_a, dRdt, d_p, max_z, z0, n_points):
        # Fixed parameters
        self.rho_s = 910  # [kg/m3] density of ice
        self.rho_p = 2500  # [kg/m3] density of particles
        self.phi_s = 0.6  # [-] volume fraction of particles in ice
        self.g = 9.81  # [m/s2] gravitation acceleration
        self.T_inf = T_inf + 273.15  # [K] ambient temperature
        self.T_s = 0 + 273.15  # [K] ice temperature
        self.T_i = 0 + 273.15  # [K] interface temperature
        self.T_out = self.T_inf  # [K] temperature just outside the plume
        self.S_inf = S_inf  # [g/kg] ambient salinity
        self.S_s = 0  # [g/kg] ice salinity
        self.S_i = 0  # interface salinity
        self.S_out = self.S_inf  # [g/kg] salinity just outside the plume
        self.c_p_p = 840  # [J/kg/K] specific heat of glass
        self.c_p_f = 4200  # [J/kg/K] specific heat of water
        self.lat_heat = 334000  # [J/kg] latent heat of fusion of water
        self.Pr = 7     # [-] Prandtl number
        self.nu = 1e-6  # [m^2/s] kinematic viscosity of water
        self.k = 0.6  # [W/m/K] thermal conductivity of water

        self.rho_inf = rho_water(self.T_inf, self.S_inf)
        self.rho_i = rho_water(self.T_i, self.S_i)
        self.rho_out = rho_water(self.T_out, self.S_out)

        self.alpha = alpha
        self.w_a = w_a
        self.dRdt = dRdt
        self.d_p = d_p

        self.constant_ws = False

        # Initial conditions
        self.b0 = b0
        self.w0 = w0
        self.phi0 = phi0
        self.T0 = self.T_i
        self.S0 = self.S_i

        # Grid
        self.n_points = n_points
        self.z0 = z0
        self.H = max_z
        self.z = -np.logspace(np.log10(self.z0), np.log10(self.H), self.n_points)

        self.parameter_names = ['alpha', 'w0', 'b0', 'phi0', 'w_a', 'dRdt', 'z0', 'max_z', 'n_points', 'phi_s']

    def set_parameter(self, param: str, value):
        assert param in self.parameter_names

        if param == 'alpha':
            self.alpha = value
        elif param == 'w0':
            self.w0 = value
        elif param == 'b0':
            self.b0 = value
        elif param == 'phi0':
            self.phi0 = value
        elif param == 'phi_s':
            self.phi_s = value
        elif param == 'w_a':
            self.w_a = value
        elif param == 'dRdt':
            self.dRdt = value
        elif param == 'z0':
            self.z0 = value
        elif param == 'max_z':
            self.H = value
        elif param == 'n_points':
            self.n_points = int(value)

        if param in ['z0', 'max_z', 'n_points']:
            self.z = -np.logspace(np.log10(self.z0), np.log10(self.H), self.n_points)

    def integrate(self, return_fluxes=False):
        # Initialize variables
        b = np.nan * np.zeros(self.z.size - 1)
        T = np.nan * np.zeros(self.z.size - 1)
        S = np.nan * np.zeros(self.z.size - 1)
        w = np.nan * np.zeros(self.z.size - 1)
        phi = np.nan * np.zeros(self.z.size - 1)
        rho = np.nan * np.zeros(self.z.size - 1)

        # Initialize fluxes
        F_f = np.zeros(self.z.size - 1)
        F_p = np.zeros(self.z.size - 1)
        F_M = np.zeros(self.z.size - 1)
        F_T = np.zeros(self.z.size - 1)
        F_S = np.zeros(self.z.size - 1)

        # Set initial conditions
        b[0] = self.b0
        T[0] = self.T0
        S[0] = self.S0
        w[0] = self.w0
        phi[0] = self.phi0
        rho[0] = rho_water(self.T0, self.S0)

        ws_const = settling_velocity(self.d_p, self.T0, self.S0, self.rho_p)
        ws0 = ws_const if self.constant_ws else 0
        F_f[0] = self.b0 * self.w0 * (1 - self.phi0)
        F_p[0] = self.b0 * (self.w0 + ws0) * self.phi0
        F_M[0] = self.b0 * self.w0 ** 2 * (1 - self.phi0) + self.b0 * (self.w0 + ws0) ** 2 * self.phi0 * self.rho_p / self.rho_inf
        F_T[0] = self.b0 * self.w0 * (1 - self.phi0) * self.T0 + self.b0 * (self.w0 + ws0) * self.phi0 * self.rho_p / self.rho_inf * self.c_p_p / self.c_p_f * self.T0
        F_S[0] = self.b0 * self.w0 * (1 - self.phi0) * self.S0

        # Start integration
        for i in range(self.n_points - 2):
            # Compute derivatives of fluxes
            dF_vf_dz = self.alpha * (w[i] - self.w_a) * self.rho_out / self.rho_inf - self.dRdt * (1 - self.phi_s) * self.rho_i / self.rho_inf
            dF_vp_dz = -self.phi_s * self.dRdt
            dF_M_dz = self.alpha * (w[i] - self.w_a) * self.rho_out / self.rho_inf * self.w_a + b[i] * self.g * (
                    (1 - phi[i]) * (rho[i] / self.rho_inf - 1) + phi[i] * (self.rho_p / self.rho_inf - 1))
            dF_T_dz = self.alpha * (w[i] - self.w_a) * self.rho_out / self.rho_inf * self.T_out - self.dRdt * (
                    1 - self.phi_s) * self.rho_i / self.rho_inf * self.T_i - self.dRdt * self.phi_s * self.rho_p / self.rho_inf * self.c_p_p / self.c_p_f * self.T_i
            dF_S_dz = self.alpha * (w[i] - self.w_a) * self.rho_out / self.rho_inf * self.S_out - self.dRdt * (1 - self.phi_s) * self.rho_i / self.rho_inf * self.S_i

            # Update fluxes
            dz = abs(self.z[i + 1] - self.z[i])
            F_f[i + 1] = F_f[i] + dF_vf_dz * dz
            F_p[i + 1] = F_p[i] + dF_vp_dz * dz
            F_M[i + 1] = F_M[i] + dF_M_dz * dz
            F_T[i + 1] = F_T[i] + dF_T_dz * dz
            F_S[i + 1] = F_S[i] + dF_S_dz * dz

            # Update variables
            ws = ws_const if self.constant_ws else -settling_velocity_at_z(self.d_p, T[i], S[i], self.rho_p, self.z[i])
            S[i + 1] = F_S[i + 1] / F_f[i + 1]
            T[i + 1] = F_T[i + 1] / (F_f[i + 1] + F_p[i + 1] * self.rho_p / self.rho_inf * self.c_p_p / self.c_p_f)
            w[i + 1] = (F_M[i + 1] - F_p[i + 1] * self.rho_p / self.rho_inf * ws) / (
                        F_f[i + 1] + F_p[i + 1] * self.rho_p / self.rho_inf)
            phi[i + 1] = 1 / (1 + F_f[i + 1] / F_p[i + 1] * (1 + ws / w[i + 1]))
            b[i + 1] = F_f[i + 1] / (1 - phi[i + 1]) / w[i + 1]
            rho[i + 1] = rho_water(T[i + 1], S[i + 1])

            if phi[i + 1] < 0:
                raise ArithmeticError("phi < 0")
        if return_fluxes:
            return F_f, F_p, F_M, F_T, F_S
        return b, w, phi, T - 273.15, S

    def converge_melt_rate(self, dRdt0=-1e-5, eps=1e-3, natural=False):
        prev_dRdt = 0
        dRdt = dRdt0
        cnt = 0
        while abs((dRdt - prev_dRdt)/dRdt0) > eps:
            prev_dRdt = dRdt
            self.dRdt = dRdt
            _, w, phi, T, S = self.integrate()

            # # Based on average Nu
            # prefac = -0.332 * self.Pr**(1/3) * self.k * (self.T_inf - self.T_i) / ((1-self.phi_s) * self.rho_s * self.lat_heat * self.H**3 * np.sqrt(self.nu))
            # z = (self.z[1:] + self.z[:-1]) / 2
            # dRdt = prefac * np.sum(np.sqrt(np.abs(z) * w) * np.abs(np.diff(self.z)))

            z = (self.z[1:] + self.z[:-1]) / 2
            if natural:
                # Bejan equation 7.51 (page 347)
                rho_f = rho_water(T+273.15, S)
                rho_plume = rho_f * (1-phi) + self.rho_p * phi
                Ra = self.g * np.abs(self.rho_inf - rho_plume)/self.rho_inf * np.abs(z)**3 / (self.nu**2) * self.Pr
                prefac = -0.503*(self.Pr/(self.Pr+0.986*self.Pr**(1/2) + 0.492))**(1/4) * self.k * (T+273.15 - self.T_i) / ((1-self.phi_s) * self.rho_s * self.lat_heat * self.H)
                dRdt = np.sum(prefac * Ra**(1/4) * np.abs(np.diff(self.z)))
            else:
                # Bejan equation 5.83 (page 244)
                # Based on local Nu
                # prefac = -0.332 * self.Pr**(1/3) * self.k * (T+273.15 - self.T_i) / ((1-self.phi_s) * self.rho_s * self.lat_heat * self.H * np.sqrt(self.nu))
                prefac = -0.332 * self.Pr ** (1 / 3) * self.k * (self.T_inf - self.T_i) / (
                            (1 - self.phi_s) * self.rho_s * self.lat_heat * self.H * np.sqrt(self.nu))  # Based on ambient temperature, same as Bejan
                dRdt = np.sum(prefac * np.sqrt(w/np.abs(z)) * np.abs(np.diff(self.z)))
            if cnt > 20:
                raise ConvergenceError("[PlumeModel.converge_melt_rate] Could not converge")
            cnt += 1
        return dRdt

    def compute_quantities(self, q=None):
        qnames = ['b', 'w', 'phi', 'T', 'S']
        result = self.integrate()
        if q is None:
            return result
        if type(q) is str:
            assert q in qnames, "invalid quantity name, possible: " + ", ".join(qnames)
            return result[qnames.index(q)]
        if type(q) is list:
            assert all([qq in qnames for qq in q]), "invalid quantity name, possible: " + ", ".join(qnames)
            return [result[qnames.index(qq)] for qq in q]

    def compute_fluxes(self, f=None):
        fnames = ['Fp', 'Ff', 'Fm', 'FT', 'FS']
        result = self.integrate(return_fluxes=True)
        if f is None:
            return result
        if type(f) is str:
            assert f in fnames, "invalid flux name, possible: "+", ".join(fnames)
            return result[fnames.index(f)]
        if type(f) is list:
            assert all([ff in fnames for ff in f]), "invalid flux name, possible: "+", ".join(fnames)
            return [result[fnames.index(ff)] for ff in f]

class ConvergenceError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


def settling_velocity(d_p, temp, sal, rho_p):
    # Ferguson & Church, J. Sed. Res. (2004)
    rho_f = rho_water(temp, sal)
    R = (rho_p-rho_f)/rho_f
    nu_f = 1e-6
    g = 9.81
    C1 = 18
    C2 = 0.4
    Vst = (R*g*d_p**2)/(C1*nu_f+np.sqrt(0.75*C2*R*g*d_p**3))
    return Vst


def settling_velocity_at_z(d_p, temp, sal, rho_p, z, v0=0.):
    """ Here we assume the amount of particles falling to be the same at any height z"""
    z = -np.abs(z)
    rho_f = rho_water(temp, sal)
    R = (rho_p - rho_f) / rho_f
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*d_p**3) + np.sqrt(C2))**2
    a = 3*Cd/(4*(R+1)*d_p)
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
    wz = np.sqrt(b/a)/(a*z) * (ach - np.tanh(ach) - np.tanh(d) - d)
    return wz


def velocity_fc2004(t, dp, rho_f, rho_p, v0=0):
    R = (rho_p - rho_f) / rho_f
    Rrho = 1 / (R + 1)
    g = 9.81  # m/s^2
    C1 = 18  # -
    C2 = 0.4  # -
    nu = 1e-6  # m^2/s
    Cd = (2*C1*nu/np.sqrt(3*R*g*dp**3) + np.sqrt(C2))**2
    a = 3/4*Rrho*Cd/dp
    b = -(Rrho-1)*g
    c = np.arctanh(-v0*np.sqrt(a/b))
    return -np.sqrt(b/a) * np.tanh(np.sqrt(a*b)*(t + c))


def rho_water(temp, sal):
    """
    Computes density of seawater in kg/m^3.
    Function taken from Eq. 6 in Sharqawy2010.
    Valid in the range 0 < t < 40 degC and 0.5 < sal < 43 g/kg.
    Accuracy: 0.01%
    """
    t68 = (temp - 273.15) / (1 - 2.5e-4)  # inverse of Eq. 4 in Sharqawy2010
    sp = sal / 1.00472  # inverse of Eq. 3 in Sharqawy2010

    rho_0 = 999.842594 + 6.793952e-2 * t68 - 9.095290e-3 * t68 ** 2 + 1.001685e-4 * t68 ** 3 - 1.120083e-6 * t68 ** 4 + 6.536336e-9 * t68 ** 5
    A = 8.24493e-1 - 4.0899e-3 * t68 + 7.6438e-5 * t68 ** 2 - 8.2467e-7 * t68 ** 3 + 5.3875e-9 * t68 ** 4
    B = -5.72466e-3 + 1.0227e-4 * t68 - 1.6546e-6 * t68 ** 2
    C = 4.8314e-4
    rho_sw = rho_0 + A * sp + B * sp ** (3 / 2) + C * sp ** 2
    return rho_sw


if __name__ == '__main__':
    pass

