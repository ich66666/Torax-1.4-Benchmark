"""IterHybrid-XHARD: aggressive flat-top initial state for adversarial control
benchmark. Skips the gymtorax ramp-up; the plasma starts in a stressed
high-beta high-temperature state and the policy has to hold it there under
the hell_disturbance perturbation cascade.

Not a fidelity simulation. Designed to break.
"""
import numpy as np


_NBI_W_TO_MA = 1.0 / 16.0e6

_nbi_times  = (0.0, 150.0)
_nbi_powers = (33.0e6, 33.0e6)
_nbi_cd     = tuple(p * _NBI_W_TO_MA for p in _nbi_powers)

_eccd_power = {0.0: 20.0e6, 150.0: 20.0e6}

_r_nbi = 0.25
_w_nbi = 0.25
_el_heat_fraction = 0.66


CONFIG = {
    "plasma_composition": {
        "main_ion":  {"D": 0.5, "T": 0.5},
        "impurity":  "Ne",
        "Z_eff":     {0.0: {0.0: 2.0, 1.0: 2.0}},
    },
    "profile_conditions": {
        "Ip":              {0.0: 12.5e6, 150.0: 12.5e6},
        "T_i":             {0.0: {0.0: 15.0, 1.0: 0.4}},
        "T_i_right_bc":    0.4,
        "T_e":             {0.0: {0.0: 15.0, 1.0: 0.4}},
        "T_e_right_bc":    0.4,
        "n_e_right_bc_is_fGW": True,
        "n_e_right_bc":    {0.0: 0.40, 150.0: 0.40},
        "nbar":            0.85,
        "n_e":             {0.0: {0.0: 1.3, 1.0: 1.0}},
        "normalize_n_e_to_nbar": True,
        "n_e_nbar_is_fGW": True,
        "initial_psi_from_j":            True,
        "initial_j_is_total_current":    True,
        "current_profile_nu":            2,
    },
    "numerics": {
        "t_final":              150.0,
        "fixed_dt":             1.0,
        "adaptive_dt":          False,
        "evolve_ion_heat":      True,
        "evolve_electron_heat": True,
        "evolve_current":       True,
        "evolve_density":       True,
    },
    "geometry": {
        "geometry_type":         "chease",
        "geometry_file":         "iterhybrid.mat2cols",
        "Ip_from_parameters":    True,
        "R_major":               6.2,
        "a_minor":               2.0,
        "B_0":                   5.3,
    },
    "sources": {
        "ecrh": {
            "gaussian_width":    0.05,
            "gaussian_location": 0.35,
            "P_total":           _eccd_power,
        },
        "generic_heat": {
            "gaussian_location":      _r_nbi,
            "gaussian_width":         _w_nbi,
            "P_total":                (_nbi_times, _nbi_powers),
            "electron_heat_fraction": _el_heat_fraction,
        },
        "generic_current": {
            "use_absolute_current":   True,
            "gaussian_width":         _w_nbi,
            "gaussian_location":      _r_nbi,
            "I_generic":              (_nbi_times, _nbi_cd),
        },
        "gas_puff": {
            "puff_decay_length": 0.3,
            "S_total":           1.0e20,
        },
        "fusion":              {},
        "ei_exchange":         {},
        "ohmic":               {},
        "cyclotron_radiation": {},
        "impurity_radiation":  {
            "model_name":           "mavrin_fit",
            "radiation_multiplier": 0.0,
        },
    },
    "neoclassical": {
        "bootstrap_current": {"bootstrap_multiplier": 1.0},
    },
    "pedestal": {
        "model_name":           "set_T_ped_n_ped",
        "set_pedestal":         True,
        "T_i_ped":              {0.0: 2.0, 150.0: 2.0},
        "T_e_ped":              {0.0: 2.0, 150.0: 2.0},
        "n_e_ped_is_fGW":       True,
        "n_e_ped":              0.85,
        "rho_norm_ped_top":     0.95,
    },
    "transport": {
        "model_name":        "qlknn",
        "apply_inner_patch": True,
        "D_e_inner":         0.15,
        "V_e_inner":         0.0,
        "chi_i_inner":       0.3,
        "chi_e_inner":       0.3,
        "rho_inner":         0.1,
        "apply_outer_patch": True,
        "D_e_outer":         0.1,
        "V_e_outer":         0.0,
        "chi_i_outer":       2.0,
        "chi_e_outer":       2.0,
        "rho_outer":         0.95,
        "chi_min":           0.05,
        "chi_max":           100.0,
        "D_e_min":           0.05,
        "D_e_max":           50.0,
        "V_e_min":          -10.0,
        "V_e_max":           10.0,
        "smoothing_width":   0.1,
        "DV_effective":      True,
        "include_ITG":       True,
        "include_TEM":       True,
        "include_ETG":       True,
        "avoid_big_negative_s": False,
    },
    "solver": {
        "solver_type":             "linear",
        "use_predictor_corrector": True,
        "n_corrector_steps":       10,
        "chi_pereverzev":          30.0,
        "D_pereverzev":            15.0,
        "use_pereverzev":          True,
    },
    "time_step_calculator": {
        "calculator_type": "fixed",
    },
}


def get_config():
    import copy
    return copy.deepcopy(CONFIG)
