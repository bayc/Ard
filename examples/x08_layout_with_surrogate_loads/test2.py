from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import floris

# data = scipy.io.loadmat('OUTPUT_FASTFarmCurl_v2_AllCombined_OnlySeedsAveraged.mat')
# SurrogateDataset = data['SurrogateModel']


config_floris = Path.cwd() / "batch.yaml"
fmodel = floris.FlorisModel(config_floris)
fmodel.set_operation_model("simple-derating")

def floris_get_case(
    WD_val=4.0,
    WS_val=8.0,
    TI_val=0.10,
    VS_val=0.10,
    yaw_val=[0.0, 0.0, 0.0],
    spacing_val=4.0,
    orientation_base=270.0,
    power_setpoint_val=np.array([1., 1., 1.]),
    p_rated=3400000,
    D_rotor=130.0,
    h_hub=110.0,
):
    # xi_turbines = spacing_val*D_rotor*np.array([-1, 0, 1])
    # eta_turbines = np.array([0.0, 0.0, 0.0])
    # x_turbines = xi_turbines*np.cos(np.radians(WD_val)) + eta_turbines*np.sin(np.radians(WD_val))
    # y_turbines = -xi_turbines*np.sin(np.radians(WD_val)) + eta_turbines*np.cos(np.radians(WD_val))
    x_turbines = np.array([0.0, spacing_val * D_rotor, 2 * spacing_val * D_rotor])
    y_turbines = np.array([0.0, 0.0, 0.0])
    power_setpoints = np.array(power_setpoint_val) * p_rated

    fmodel.set(
        layout_x=x_turbines,
        layout_y=y_turbines,
        yaw_angles=np.array([yaw_val]),
        # wind_directions=[orientation_base],
        wind_directions=[orientation_base + WD_val],
        wind_speeds=[WS_val],
        turbulence_intensities=[TI_val],
        wind_shear=VS_val,
        power_setpoints=np.array([power_setpoints]),
    )
    fmodel.run()
    SAWS = fmodel.get_turbine_SAWS()#.squeeze()
    SATI = fmodel.get_turbine_SATI()#.squeeze()

    return SAWS, SATI

# floris_get_case()

wind_speeds = np.array([6, 10, 14])
TIs = np.array([.1, .2])
shear_coeff = np.array([0.1, 0.2])
turbine_spacing = np.array([4, 6, 8])
wind_directions = np.array([-8., 0., 8.])
yaw_T1 = [-30., 0., 30.]
yaw_T2 = [-20., 0., 20.]
power_setpoints_T1_T2 = [.6, 1.]

floris_surrogate_inputs = pd.DataFrame()

for ws in wind_speeds:
    for TI in TIs:
        for shear in shear_coeff:
            for turb_spacing in turbine_spacing:
                for wd in wind_directions:
                    for yawT1 in yaw_T1:
                        for yawT2 in yaw_T2:
                            for power_setpoint in power_setpoints_T1_T2:
                                SAWS, SATI = floris_get_case(
                                    WD_val=wd,
                                    WS_val=ws,
                                    TI_val=TI,
                                    VS_val=shear,
                                    yaw_val=[yawT1, yawT2, 0.0],
                                    spacing_val=turb_spacing,
                                    power_setpoint_val=[power_setpoint, power_setpoint, 1.],
                                    p_rated=3.4e6,
                                    orientation_base=270.0,
                                    D_rotor=130.0,
                                    h_hub=110.0,
                                )
                                floris_inputs = {
                                    "spacing_D": turb_spacing,
                                    "wd_deg": wd,
                                    "ws_m_per_s": ws,
                                    "TI_per": TI,
                                    "VS": shear,
                                    "yaw_deg": [yawT1, yawT2, 0.0],
                                    "power_demand_per": [power_setpoint, power_setpoint, 1.],
                                    "SAWS": SAWS.flatten(),
                                    "SATI": SATI.flatten(),
                                }
                                floris_surrogate_input = pd.DataFrame.from_dict(
                                    floris_inputs,
                                    orient='index',
                                )
                                floris_surrogate_inputs = pd.concat(
                                    [floris_surrogate_inputs, floris_surrogate_input],
                                    axis=1,
                                    ignore_index=False,
                                )

                                # if wd == 0.0:
                                #     break

wind_speeds = np.array([8])
TIs = np.array([.1])
shear_coeff = np.array([0.1])
turbine_spacing = np.array([6])
wind_directions = np.array([-8., -4., 0., 4., 8.])
yaw_T1 = [-30., -20., -10., 0., 10., 20., 30.]
yaw_T2 = [-20., -10., 0., 10., 20.]
power_setpoints_T1_T2 = [.6, .7, .8, .9, 1.]

for ws in wind_speeds:
    for TI in TIs:
        for shear in shear_coeff:
            for turb_spacing in turbine_spacing:
                for wd in wind_directions:
                    for yawT1 in yaw_T1:
                        for yawT2 in yaw_T2:
                            for power_setpoint in power_setpoints_T1_T2:
                                SAWS, SATI = floris_get_case(
                                    WD_val=wd,
                                    WS_val=ws,
                                    TI_val=TI,
                                    VS_val=shear,
                                    yaw_val=[yawT1, yawT2, 0.0],
                                    spacing_val=turb_spacing,
                                    power_setpoint_val=[power_setpoint, power_setpoint, 1.],
                                    p_rated=3.4e6,
                                    orientation_base=270.0,
                                    D_rotor=130.0,
                                    h_hub=110.0,
                                )
                                floris_inputs = {
                                    "spacing_D": turb_spacing,
                                    "wd_deg": wd,
                                    "ws_m_per_s": ws,
                                    "TI_per": TI,
                                    "VS": shear,
                                    "yaw_deg": [yawT1, yawT2, 0.0],
                                    "power_demand_per": [power_setpoint, power_setpoint, 1.],
                                    "SAWS": SAWS.flatten(),
                                    "SATI": SATI.flatten(),
                                }
                                floris_surrogate_input = pd.DataFrame.from_dict(
                                    floris_inputs,
                                    orient='index',
                                )
                                floris_surrogate_inputs = pd.concat(
                                    [floris_surrogate_inputs, floris_surrogate_input],
                                    axis=1,
                                    ignore_index=False,
                                )

floris_surrogate_inputs = floris_surrogate_inputs.transpose()
floris_surrogate_inputs.to_csv("floris_surrogate_inputs.csv", index=False)
floris_surrogate_inputs.to_excel("floris_surrogate_inputs.xlsx", sheet_name="Sheet1", index=False)
