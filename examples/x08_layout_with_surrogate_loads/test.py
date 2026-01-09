from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import floris
import floris.layout_visualization as layoutviz
from floris.flow_visualization import visualize_cut_plane

from ard.farm_aero.floris import create_FLORIS_turbine_from_windIO
from ard.farm_aero.templates import create_windresource_from_windIO
from ard.utils.io import load_yaml
from ard.farm_loads.surrogate_load_functions import (
    ANN_DEL_BladeRoot,
    ANN_DEL_Shaft,
    ANN_DEL_TowerBase,
    ANN_DEL_YawBearings,
)


path_inputs = Path.cwd().absolute() / "inputs"
input_dict = load_yaml(path_inputs / "ard_system.yaml")

layout_x1 = np.array([0.2609205761552146, -0.46762644489752747, 0.1747859928911326, -0.2425751876015167, -1.0640161942497564, 1.4964998343327909, -1.343894289929609, 0.5489693863223485, -0.8685254814874988]
) * 1000
layout_y1 = np.array([1.148559316675545, 1.3477908392468698, -0.9781924513066624, -0.8478603656150479, -1.1144653020696271, -0.5052355698017017, -1.3792816597481272, -0.6516496606531748, -0.10268051065013134]
) * 1000

layout_x2 = np.array([0.14877266664536104, 0.779021708317688, 0.5952853442924542, -0.32002358936340836, 0.9379001368888673, 1.4812382026973454, -1.2711618577686026, 1.2187497095500315, -0.8473650428371415]
) * 1000
layout_y2 = np.array([-0.987512305220853, -0.2808748367158094, -1.0677143230357622, -1.300023281858448, -0.9386484488479849, -0.4990797664466369, -1.4987540873336875, -0.7585678621719356, -1.41999293641547]
) * 1000

layout_x3 = np.array([0.07693080572097805, 0.5708255598237529, 0.4684593594834569, -0.3162116861018182, 0.9709762879825411, 1.5000000000000002, -1.3859539278330402, 1.2055927072711055, -0.9708018033472681]
) * 1000
layout_y3 = np.array([-0.9880131250094761, -0.7874106978692765, -1.0686706279522908, -1.405282847639196, -0.8414652194986524, -0.4216782506166495, -1.4727846025496827, -0.6643757671307712, -1.4152826150006597]
) * 1000

D = 130.
layout_x = np.array([])
layout_y = np.array([])

layouts = [[layout_x1, layout_y1], [layout_x2, layout_y2], [layout_x3, layout_y3]]
layout_names = ['Layout 1', 'Layout 2', 'Layout 3']

# layouts = [[np.array([0., 6*D, 12*D]), np.array([0., 0., 0.])]]

dels_bladeroot = []
dels_shaft = []
dels_towerbase = []
dels_yawbearings = []

# fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(10,8), sharey=True)

for i, layout in enumerate(layouts):

    input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["x"] = layout[0]
    input_dict["modeling_options"]["windIO_plant"]["wind_farm"]["layouts"]["coordinates"]["y"] = layout[1]

    modeling_options = input_dict["modeling_options"]
    windIO = modeling_options["windIO_plant"]

    wind_query = create_windresource_from_windIO(windIO, "probability")

    # wind_query.upsample(wd_step=1.0, inplace=True)

    fmodel = floris.FlorisModel("defaults")

    fmodel.set(
        turbine_type=[create_FLORIS_turbine_from_windIO(windIO, modeling_options)],
        wind_shear=windIO["site"]["energy_resource"]["wind_resource"].get("shear"),
        reference_wind_height=getattr(wind_query, "reference_height", None),
        solver_settings=(modeling_options.get("floris", {}).get("solver_settings")),
        layout_x=layout[0],
        layout_y=layout[1],
        wind_data=wind_query,
    )
    # print(fmodel.layout_x)

    fmodel.run()

    # ax = axarr[i]
    # layoutviz.plot_turbine_points(fmodel, ax=ax)

    # fmodel.core.to_file(Path("batch.yaml"))

    SATI = fmodel.get_turbine_SATI() * 100
    SAWS = fmodel.get_turbine_SAWS()
    turbine_powers_percent = fmodel.get_turbine_powers_percent().flatten()
    turbine_powers_percent = np.ones_like(turbine_powers_percent) * 100.

    SATI_collapsed = SATI.reshape(-1, SATI.shape[-1])
    SAWS_collapsed = SAWS.reshape(-1, SAWS.shape[-1])

    Yaw = fmodel.core.farm.yaw_angles.flatten()

    # [SAWSup SAWSright SAWSdown SAWSleft SATIup SATIright SATIdown SATIleft Yaw PowerDemand[%]]
    input_data = np.concatenate(
        (
            SAWS_collapsed,
            SATI_collapsed,
            Yaw[:, None],
            turbine_powers_percent[:, None],
        ),
        axis=1,
    )
    # print(input_data)

    # input_data = np.array([
    #     8.22342659,	7.977170676, 7.633753041, 7.972027965, 8.560708045,	8.693631852, 9.258131985,	9.096162388, 0.0, 100.
    # ])

    # input_data = np.array([
    #     8.11, 7.91, 7.60, 7.91, 7.96, 7.56, 8.09, 8.29, 0.0, 100.
    # ])

    # Make predictions using ANN surrogates
    del_bladeroot_ann = ANN_DEL_BladeRoot(input_data)
    del_shaft_ann = ANN_DEL_Shaft(input_data)
    del_towerbase_ann = ANN_DEL_TowerBase(input_data)
    del_yawbearings_ann = ANN_DEL_YawBearings(input_data)

    # print(del_towerbase_ann)
    # lkj

    dels_bladeroot.append(del_bladeroot_ann)
    dels_shaft.append(del_shaft_ann)
    dels_towerbase.append(del_towerbase_ann)
    dels_yawbearings.append(del_yawbearings_ann)

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    # ax[0, 0].hist(del_bladeroot_ann, bins=30)
    # ax[0, 0].set_title('Bladeroot DEL')
    # # ax[0, 0].set_ylim(0, 1000)
    # ax[0, 0].set_yscale('log')
    # ax[0, 0].set_ylabel('Frequency (log scale)')
    # ax[0, 0].set_xlabel('Value [kNm]')
    # ax[0, 1].hist(del_shaft_ann, bins=30)
    # ax[0, 1].set_title('Shaft DEL')
    # # ax[0, 1].set_ylim(0, 200)
    # ax[1, 0].set_yscale('log')
    # ax[1, 0].set_ylabel('Frequency (log scale)')
    # ax[1, 0].set_xlabel('Value [kNm]')
    # ax[1, 0].hist(del_towerbase_ann, bins=30)
    # ax[1, 0].set_title('Tower Base DEL')
    # # ax[1, 0].set_ylim(0, 200)
    # ax[0, 1].set_yscale('log')
    # ax[0, 1].set_ylabel('Frequency (log scale)')
    # ax[0, 1].set_xlabel('Value [kNm]')
    # ax[1, 1].hist(del_yawbearings_ann, bins=30)
    # ax[1, 1].set_title('Yaw Bearings DEL')
    # # ax[1, 1].set_ylim(0, 1000)
    # ax[1, 1].set_yscale('log')
    # ax[1, 1].set_ylabel('Frequency (log scale)')
    # ax[1, 1].set_xlabel('Value [kNm]')

    # fig.suptitle(layout_names[i], fontsize=16)
    # plt.savefig('layout_' + str(i) + '_upsampledWD.png')
    # plt.close()

# plt.savefig("turbine_layouts.png")
# plt.close()

fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=((12,4)), constrained_layout=True)

for i, layout in enumerate(layouts):
    fmodel.set(
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.1],
        layout_x=layout[0],
        layout_y=layout[1],
    )
    fmodel.run()

    horizontal_plane = fmodel.calculate_horizontal_plane(
        x_resolution=250,
        y_resolution=250,
        x_bounds=(-2500., 2500.),
        y_bounds=(-2500., 2500.),
        height=110.0,
    )

    # Plot the flow field with rotors
    visualize_cut_plane(
        horizontal_plane,
        ax=axarr[i],
        label_contours=False,
        title="Layout " + str(i+1),
    )

    # Plot the turbine rotors
    layoutviz.plot_turbine_rotors(fmodel, ax=axarr[i])
    layoutviz.plot_turbine_labels(fmodel, ax=axarr[i])

    axarr[i].set_xlim((-2000., 2000.))
    axarr[i].set_ylim((-2000., 2000.))

    square_bottom_left_x = -1500.
    square_bottom_left_y = -1500.
    square_width = 3000.
    square_height = 3000.

    # 4. Add the square patch
    # Set a higher zorder (e.g., 3) to ensure it appears on top of the contour lines
    square = patches.Rectangle(
        (square_bottom_left_x, square_bottom_left_y), 
        square_width, 
        square_height, 
        linewidth=1, 
        edgecolor='black', 
        facecolor='none', 
        zorder=4
    )
    axarr[i].add_patch(square)

    if i == 0:
        axarr[i].set_ylabel("y-coordinate [m]")
    axarr[i].set_xlabel("x-coordinate [m]")

plt.savefig("turbine_layouts.png")
plt.close()

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10,8), sharex='col', sharey=True)

ax[0, 0].set_title('Bladeroot DEL')
ax[2, 0].set_xlabel('Value [kNm]')
for i, dels in enumerate(dels_bladeroot):
    ax[i, 0].hist(dels, bins=30)
    ax[i, 0].set_yscale('log')
    ax[i, 0].set_ylabel('Frequency (log scale)')
    # ax[i, 0].set_xlabel('Value [kNm]')

ax[0, 1].set_title('Shaft DEL')
ax[2, 1].set_xlabel('Value [kNm]')
for i, dels in enumerate(dels_shaft):
    ax[i, 1].hist(dels, bins=30)
    ax[i, 1].set_yscale('log')
    # ax[i, 1].set_ylabel('Frequency (log scale)')
    # ax[i, 1].set_xlabel('Value [kNm]')

ax[0, 2].set_title('Tower Base DEL')
ax[2, 2].set_xlabel('Value [kNm]')
for i, dels in enumerate(dels_towerbase):
    ax[i, 2].hist(dels, bins=30)
    ax[i, 2].set_yscale('log')
    # ax[i, 2].set_ylabel('Frequency (log scale)')
    # ax[i, 2].set_xlabel('Value [kNm]')

ax[0, 3].set_title('Yaw Bearings DEL')
ax[2, 3].set_xlabel('Value [kNm]')
for i, dels in enumerate(dels_yawbearings):
    ax[i, 3].hist(dels, bins=30)
    ax[i, 3].set_yscale('log')
    # ax[i, 3].set_ylabel('Frequency (log scale)')
    # ax[i, 3].set_xlabel('Value [kNm]')

fig.text(0.025, 0.78, 'Layout 1', va='center', rotation='vertical', fontsize=14)
fig.text(0.025, 0.5,  'Layout 2', va='center', rotation='vertical', fontsize=14)
fig.text(0.025, 0.22, 'Layout 3', va='center', rotation='vertical', fontsize=14)

plt.savefig('dels_dist.png')
plt.close()
