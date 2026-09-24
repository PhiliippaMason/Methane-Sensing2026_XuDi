# Forward model workflow:
# 1. Build vertical atmospheric layers from surface pressure
# 2. Compute gas optical depths from layer subcolumns and absorption LUTs
# 3. Compute aerosol optical properties from aerosol LUTs and vertical distribution
# 4. Solve radiative transfer to obtain high-resolution TOA radiance
# 5. Add surface reflection and fluorescence contribution
# 6. Convolve with instrument spectral response function
# 7. Output simulated spectrum

import numpy as np

# =========================
# 1. Data structure definition
# =========================


class StateVector:
    """
    State vector to be inverted x
    """

    def __init__(
        self,
        ch4_subcolumns,  # shape: (L,)
        co_subcolumns,  # shape: (L,)
        h2o_subcolumns,  # shape: (L,)
        o2_subcolumns,  # shape: (L,)
        aer_column,  # N_aer
        aer_size_param,  # alpha
        aer_height,  # z_aer
        aer_width,  # w0
        albedo_coeffs,  # [a0, a1, a2]
        fluo_coeffs,
    ):  # [f0, f1]
        self.ch4_subcolumns = ch4_subcolumns
        self.co_subcolumns = co_subcolumns
        self.h2o_subcolumns = h2o_subcolumns
        self.o2_subcolumns = o2_subcolumns

        self.aer_column = aer_column
        self.aer_size_param = aer_size_param
        self.aer_height = aer_height
        self.aer_width = aer_width

        self.albedo_coeffs = albedo_coeffs
        self.fluo_coeffs = fluo_coeffs


class AuxiliaryData:
    """
    Auxiliary input b
    """

    def __init__(
        self,
        surface_pressure,  # p_s
        pressure_edges,  # shape: (L+1,)
        temperature_profile,  # shape: (L,)
        height_edges,  # shape: (L+1,)
        sza,  # solar zenith angle, degree
        vza,  # viewing zenith angle, degree
        raa,  # relative azimuth angle, degree
        solar_spectrum_hi,  # shape: (N_nu,)
        nu_hi,  # high-res spectral grid
        nu_inst,  # instrument spectral grid
        isrf_matrix,  # shape: (N_inst, N_hi)
        refr_index_real,  # m_r
        refr_index_imag,
    ):  # m_i
        self.surface_pressure = surface_pressure
        self.pressure_edges = pressure_edges
        self.temperature_profile = temperature_profile
        self.height_edges = height_edges

        self.sza = sza
        self.vza = vza
        self.raa = raa

        self.solar_spectrum_hi = solar_spectrum_hi
        self.nu_hi = nu_hi
        self.nu_inst = nu_inst
        self.isrf_matrix = isrf_matrix

        self.refr_index_real = refr_index_real
        self.refr_index_imag = refr_index_imag


class LUT:
    """
    Pre-computed lookup table
    """

    def __init__(
        self,
        gas_absorption_lut,  # Lookup function/object for each gas absorption cross section
        aerosol_optics_lut,
    ):  # erosol Scattering Property Lookup Function/Object
        self.gas_absorption_lut = gas_absorption_lut
        self.aerosol_optics_lut = aerosol_optics_lut


class GasAbsorptionLUT:
    """
    Minimal gas absorption LUT example
    """

    def interpolate(self, gas_name, nu, pressure, temperature):
        # Placeholder: return a fake absorption cross section
        # shape: (N_nu,)
        scale = {
            "CH4": 1.0e-21,
            "CO": 5.0e-22,
            "H2O": 2.0e-21,
            "O2": 8.0e-22,
        }.get(gas_name, 1.0e-22)

        return scale * np.ones_like(nu)


class AerosolOpticsLUT:
    """
    Minimal aerosol optics LUT example
    """

    def interpolate(self, nu, size_param, refr_real, refr_imag):
        n = len(nu)

        extinction = 1.0e-3 * np.ones(n)
        ssa = 0.95 * np.ones(n)

        return {
            "extinction": extinction,
            "ssa": ssa,
            "phase": None,
        }


# =========================
# 2. utility functions
# =========================


def cosd(angle_deg):
    return np.cos(np.deg2rad(angle_deg))


def compute_layer_mid_pressure(pressure_edges):
    """
    The pressure at the midpoint of the layer is obtained from the layer boundary pressure.
    pressure_edges: (L+1,)
    return: (L,)
    """
    return 0.5 * (pressure_edges[:-1] + pressure_edges[1:])


def surface_albedo_model(nu, coeffs, nu0):
    """
    Surface albedo polynomial
    A_s(nu) = a0 + a1*(nu - nu0) + a2*(nu - nu0)^2
    """
    a0, a1, a2 = coeffs
    return a0 + a1 * (nu - nu0) + a2 * (nu - nu0) ** 2


def fluorescence_model(nu, coeffs, nu_ref):
    """
    Fluorescent items
    F_fluo(nu) = f0 + f1*(nu - nu_ref)
    """
    f0, f1 = coeffs
    return f0 + f1 * (nu - nu_ref)


def gaussian_aerosol_profile(z_edges, N_aer, z_center, width):
    """
    The aerosol column mass was distributed to each layer according to a Gaussian distribution.
    z_edges: (L+1,)
    return layer_aer_column: (L,)
    """
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = np.diff(z_edges)

    profile = np.exp(-0.5 * ((z_mid - z_center) / width) ** 2)
    profile = profile / np.sum(
        profile * dz
    )  # Normalize to make the integral equal to 1

    layer_aer_column = N_aer * profile * dz
    return layer_aer_column


# =========================
# 3. LUT Lookup
# =========================


def lookup_gas_cross_section(gas_name, nu_hi, p_layer, T_layer, gas_absorption_lut):
    """
    Interpolation from the gas absorption LUT yields:
    sigma_g(nu, p, T)
    return shape: (N_nu,)
    """
    # Pseudocode: In a real implementation, it should be written as 3D interpolation.
    sigma = gas_absorption_lut.interpolate(
        gas_name=gas_name, nu=nu_hi, pressure=p_layer, temperature=T_layer
    )
    return sigma


def lookup_aerosol_properties(nu_hi, alpha, mr, mi, aerosol_optics_lut):
    """
    Obtained from aerosol LUT:
    - extinction cross section
    - single scattering albedo
    - phase function / moments
    """
    props = aerosol_optics_lut.interpolate(
        nu=nu_hi, size_param=alpha, refr_real=mr, refr_imag=mi
    )
    # props example:
    # props["extinction"] -> (N_nu,)
    # props["ssa"]        -> (N_nu,)
    # props["phase"]      -> ...
    return props


# =========================
# 4. Optical thickness calculation
# =========================


def compute_gas_optical_depths(x, b, lut):
    """
    Calculate the total gas absorption optical thickness at each wavenumber for each layer
    return tau_gas_layers: shape (L, N_nu)
    """
    nu_hi = b.nu_hi
    p_mid = compute_layer_mid_pressure(b.pressure_edges)
    T = b.temperature_profile
    L = len(T)
    N_nu = len(nu_hi)

    tau_gas_layers = np.zeros((L, N_nu))

    gas_dict = {
        "CH4": x.ch4_subcolumns,
        "CO": x.co_subcolumns,
        "H2O": x.h2o_subcolumns,
        "O2": x.o2_subcolumns,
    }

    for a in range(L):
        p_l = p_mid[a]
        T_l = T[a]

        for gas_name, subcolumns in gas_dict.items():
            sigma = lookup_gas_cross_section(
                gas_name=gas_name,
                nu_hi=nu_hi,
                p_layer=p_l,
                T_layer=T_l,
                gas_absorption_lut=lut.gas_absorption_lut,
            )
            tau_gas_layers[a, :] += sigma * subcolumns[a]

    return tau_gas_layers


def compute_aerosol_optical_depths(x, b, lut):
    """
    Calculate the optical thickness and scattering properties of each aerosol layer.
    return:
        tau_aer_layers: (L, N_nu)
        ssa_layers:     (L, N_nu)
        phase_layers:   optional
    """
    nu_hi = b.nu_hi
    z_edges = b.height_edges
    L = len(z_edges) - 1
    N_nu = len(nu_hi)

    # 1) Vertical distribution
    aer_layer_column = gaussian_aerosol_profile(
        z_edges=z_edges, N_aer=x.aer_column, z_center=x.aer_height, width=x.aer_width
    )

    # 2) LUT scattering properties
    aer_props = lookup_aerosol_properties(
        nu_hi=nu_hi,
        alpha=x.aer_size_param,
        mr=b.refr_index_real,
        mi=b.refr_index_imag,
        aerosol_optics_lut=lut.aerosol_optics_lut,
    )

    ext = aer_props["extinction"]  # (N_nu,)
    ssa = aer_props["ssa"]  # (N_nu,)
    phase = aer_props.get("phase", None)

    tau_aer_layers = np.zeros((L, N_nu))
    ssa_layers = np.zeros((L, N_nu))

    for b in range(L):
        tau_aer_layers[b, :] = ext * aer_layer_column[b]
        ssa_layers[b, :] = ssa

    return tau_aer_layers, ssa_layers, phase


# =========================
# 5. RT Solver Interface
# =========================


def radiative_transfer_solver(
    tau_total_layers, tau_aer_layers, ssa_layers, phase_layers, sza, vza, raa
):
    """
    RTM Solver Interface
      - DISORT
      - VLIDORT
      - doubling-adding
      - Own approximate solver

    返回:
      T_down  : Downward transmission rate, shape (N_nu,)
      T_up    : Upward transmittance, shape (N_nu,)
      S_atm   : Large balloon albedo, shape (N_nu,)
      I_path  : Path radiation, shape (N_nu,)
    """
    # N_nu = tau_total_layers.shape[1]

    # -------- A very simplified approximation --------
    mu0 = cosd(sza)
    mu = cosd(vza)

    tau_col = np.sum(tau_total_layers, axis=0)

    T_down = np.exp(-tau_col / mu0)
    T_up = np.exp(-tau_col / mu)

    # Simplified path scattering term
    tau_aer_col = np.sum(tau_aer_layers, axis=0)
    mean_ssa = np.mean(ssa_layers, axis=0)
    I_path = 0.05 * tau_aer_col * mean_ssa

    # Simplified spherical albedo
    S_atm = 0.02 * tau_aer_col

    return T_down, T_up, S_atm, I_path


# =========================
# 6. Assemble high-resolution TOA spectrum
# =========================


def build_high_res_toa_radiance(x, b, lut):
    """
    Generate high-resolution TOA radiance
    return I_toa_hi: shape (N_nu,)
    """
    nu_hi = b.nu_hi
    E0 = b.solar_spectrum_hi
    mu0 = cosd(b.sza)

    # 1) Calculate the optical thickness of each layer
    tau_gas_layers = compute_gas_optical_depths(x, b, lut)
    tau_aer_layers, ssa_layers, phase_layers = compute_aerosol_optical_depths(x, b, lut)

    tau_total_layers = tau_gas_layers + tau_aer_layers

    # 2) RT solution
    T_down, T_up, S_atm, I_path = radiative_transfer_solver(
        tau_total_layers=tau_total_layers,
        tau_aer_layers=tau_aer_layers,
        ssa_layers=ssa_layers,
        phase_layers=phase_layers,
        sza=b.sza,
        vza=b.vza,
        raa=b.raa,
    )

    # 3) Surface albedo
    nu0 = np.mean(nu_hi)
    A_s = surface_albedo_model(nu_hi, x.albedo_coeffs, nu0)

    # multiple surface-atmosphere coupling corrections
    denom = 1.0 - A_s * S_atm
    denom = np.maximum(denom, 1e-6)  # Preventing division by zero

    I_surface = (mu0 * E0 / np.pi) * A_s * T_down * T_up / denom

    # 4) Fluorescent items
    nu_f = np.mean(nu_hi)
    F_fluo = fluorescence_model(nu_hi, x.fluo_coeffs, nu_f)
    I_fluo = (F_fluo / np.pi) * T_up

    # 5) TOA High resolution radiance
    I_toa_hi = I_surface + I_path + I_fluo

    return I_toa_hi


# =========================
# 7. Instrument convolution
# =========================


def convolve_with_instrument(I_toa_hi, isrf_matrix):
    """
    Convolution with instrument response function
    I_toa_hi:   (N_hi,)
    isrf_matrix:(N_inst, N_hi)
    return y_sim:(N_inst,)
    """
    return isrf_matrix @ I_toa_hi


# =========================
# 8. Overall forward model
# =========================


def forward_model(x, b, lut):
    """
    Final forward model:
    y_sim = F(x, b)
    """
    # Step 1: High-resolution TOA radiance
    I_toa_hi = build_high_res_toa_radiance(x, b, lut)

    # Step 2: Instrument convolution
    y_sim = convolve_with_instrument(I_toa_hi, b.isrf_matrix)

    return y_sim


# =========================
# 9. Main program
# =========================


def main():
    # -------------------------
    # Example
    # -------------------------

    L = 12
    N_hi = 5000
    N_inst = 400

    x = StateVector(
        ch4_subcolumns=np.ones(L) * 1e-3,
        co_subcolumns=np.ones(L) * 1e-4,
        h2o_subcolumns=np.ones(L) * 1e-2,
        o2_subcolumns=np.ones(L) * 1e-1,
        aer_column=0.1,
        aer_size_param=1.5,
        aer_height=2.0,  # km
        aer_width=1.0,  # km
        albedo_coeffs=[0.2, 0.0, 0.0],
        fluo_coeffs=[0.01, 0.0],
    )

    b = AuxiliaryData(
        surface_pressure=1013.25,
        pressure_edges=np.linspace(1013.25, 1.0, L + 1),
        temperature_profile=np.linspace(290, 220, L),
        height_edges=np.linspace(0, 12, L + 1),  # km
        sza=30.0,
        vza=10.0,
        raa=50.0,
        solar_spectrum_hi=np.ones(N_hi),
        nu_hi=np.linspace(12900, 13200, N_hi),
        nu_inst=np.linspace(12900, 13200, N_inst),
        isrf_matrix=np.random.rand(N_inst, N_hi),  # Normalization
        refr_index_real=1.4,
        refr_index_imag=0.01,
    )

    # Assume the LUT object has an interpolate method
    lut = LUT(
        gas_absorption_lut=GasAbsorptionLUT(),
        aerosol_optics_lut=AerosolOpticsLUT(),
    )

    y_sim = forward_model(x, b, lut)

    return y_sim
