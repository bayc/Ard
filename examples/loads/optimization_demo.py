import os
from pathlib import Path

import pprint as pp

import numpy as np
import matplotlib.pyplot as plt

import floris
from floris.wind_data import WindRose
import openmdao.api as om

import optiwindnet.plotting
import ard
import ard.layout.spacing
import ard.layout.gridfarm
import ard.farm_aero
import ard.utils.io
from ard.cost.wisdem_wrap import LandBOSSEWithSpacingApproximations


def run_example():

    layout_type = "gridfarm"
    nturbs = 9
    cable_max_tot_length = 7500
    maxiters = 200
    optimizer = "COBYLA"
    objective = "weighted_loads_by_freq"
    case_title = layout_type + "_" + str(nturbs) + "turbs_" + str(cable_max_tot_length) + "maxcablelength_" + str(maxiters) + "iters_" + optimizer + "_" + objective

    # create the wind query
    wind_rose_wrg = floris.wind_data.WindRoseWRG(
        Path(ard.__file__).parents[1] / "examples" / "data" / "wrg_example.wrg"
    )

    # Get the interpolated data
    sector_freq = wind_rose_wrg._interpolate_data(0, 0, wind_rose_wrg.interpolant_sector_freq)
    weibull_A = wind_rose_wrg._interpolate_data(0, 0, wind_rose_wrg.interpolant_weibull_A)
    weibull_k = wind_rose_wrg._interpolate_data(0, 0, wind_rose_wrg.interpolant_weibull_k)

    # Initialize the freq_table
    freq_table = np.zeros((wind_rose_wrg.n_sectors, len(wind_rose_wrg.wind_speeds)))

    # First fill in the rows of the table using the weibull distributions,
    # weighted by the sector freq
    for sector in range(wind_rose_wrg.n_sectors):
        wind_speeds, freq = wind_rose_wrg._generate_wind_speed_frequencies_from_weibull(
            weibull_A[sector], weibull_k[sector], wind_speeds=wind_rose_wrg.wind_speeds
        )
        freq_table[sector, :] = sector_freq[sector] * freq

    # Normalize the table
    freq_table = freq_table / freq_table.sum()

    wind_rose = WindRose(
        wind_directions=wind_rose_wrg.wind_directions,
        wind_speeds=wind_speeds,
        freq_table=freq_table,
        ti_table=wind_rose_wrg.ti_table,
        compute_zero_freq_occurrence=True,
    )
    wind_rose.plot()
    # plt.show()
    # lkj
    wind_rose_wrg.set_wd_step(90.0)
    wind_rose_wrg.set_wind_speeds(np.array([5.0, 10.0, 15.0, 20.0]))
    wind_rose = wind_rose_wrg.get_wind_rose_at_point(0.0, 0.0)
    wind_query = ard.wind_query.WindQuery.from_FLORIS_WindData(wind_rose)

    # specify the configuration/specification files to use
    filename_turbine_spec = (
        Path(ard.__file__).parents[1]
        / "examples"
        / "data"
        / "turbine_spec_IEA-3p4-130-RWT.yaml"
    )  # toolset generalized turbine specification
    data_turbine_spec = ard.utils.io.load_turbine_spec(filename_turbine_spec)

    # set up the modeling options
    modeling_options = {
        "farm": {
            "N_turbines": nturbs,
            "N_substations": 1,
        },
        "turbine": data_turbine_spec,
        "collection": {
            "max_turbines_per_string": 8,
            "solver_name": "appsi_highs",
            "solver_options": dict(
                time_limit=60,
                mip_rel_gap=0.005,  # TODO ???
            ),
        },
    }

    # create the OpenMDAO model
    model = om.Group()
    group_layout2aep = om.Group()

    # first the layout
    if layout_type == "gridfarm":
        group_layout2aep.add_subsystem(  # layout component
            "layout",
            ard.layout.gridfarm.GridFarmLayout(modeling_options=modeling_options),
            promotes=["*"],
        )
        layout_global_input_promotes = [
            "angle_orientation",
            "angle_skew",
            "spacing_primary",
            "spacing_secondary",
        ]
    elif layout_type == "sunflower":
        group_layout2aep.add_subsystem(  # layout component
            "layout",
            ard.layout.sunflower.SunflowerFarmLayout(modeling_options=modeling_options),
            promotes=["*"],
        )
        layout_global_input_promotes = ["spacing_target"]
    elif layout_type == "fullfarm":
        group_layout2aep.add_subsystem(  # layout component
            "layout",
            ard.layout.fullfarm.FullFarmLanduse(modeling_options=modeling_options),
            promotes=["*"],
        )
        layout_global_input_promotes = ["x_turbines", "y_turbines"]
    else:
        raise KeyError("you shouldn't be able to get here.")
    layout_global_output_promotes = [
        "spacing_effective_primary",
        "spacing_effective_secondary",
    ]  # all layouts have this

    # group_layout2aep.add_subsystem(  # FLORIS AEP component
    #     "aepPlaceholder",
    #     ard.farm_aero.placeholder.PlaceholderAEP(
    #         modeling_options=modeling_options,
    #         wind_rose=wind_rose,
    #     ),
    #     # promotes=["AEP_farm"],
    #     promotes=["x_turbines", "y_turbines", "AEP_farm"],
    # )
    group_layout2aep.add_subsystem(  # FLORIS AEP component
        "aepFLORIS",
        ard.farm_aero.floris.FLORISTowerBaseLoad(
            modeling_options=modeling_options,
            wind_rose=wind_rose,
            case_title=case_title,
        ),
        # promotes=["AEP_farm"],
        promotes=["x_turbines", "y_turbines", "AEP_farm", "tower_base_load"],
    )
    farmaero_global_output_promotes = ["AEP_farm", "tower_base_load"]

    group_layout2aep.approx_totals(
        method="fd", step=1e-3, form="central", step_calc="rel_avg"
    )
    model.add_subsystem(
        "layout2aep",
        group_layout2aep,
        promotes_inputs=[
            *layout_global_input_promotes,
        ],
        promotes_outputs=[
            *layout_global_output_promotes,
            *farmaero_global_output_promotes,
        ],
    )

    if layout_type == "gridfarm":
        model.add_subsystem(  # landuse component
            "landuse",
            ard.layout.gridfarm.GridFarmLanduse(modeling_options=modeling_options),
            promotes_inputs=layout_global_input_promotes,
        )
    elif layout_type == "sunflower":
        model.add_subsystem(  # landuse component
            "landuse",
            ard.layout.sunflower.SunflowerFarmLanduse(
                modeling_options=modeling_options
            ),
        )
        model.connect("layout2aep.x_turbines", "landuse.x_turbines")
        model.connect("layout2aep.y_turbines", "landuse.y_turbines")
    else:
        raise KeyError("you shouldn't be able to get here.")

    model.add_subsystem(  # collection component
        "optiwindnet_coll",
        ard.collection.optiwindnetCollection(
            modeling_options=modeling_options,
            case_title=case_title,
        ),
    )
    model.connect("layout2aep.x_turbines", "optiwindnet_coll.x_turbines")
    model.connect("layout2aep.y_turbines", "optiwindnet_coll.y_turbines")

    # model.add_subsystem(  # tower base load component
    #     "farmload",
    #     ard.farm_aero.floris.FLORISTowerBaseLoad(
    #         modeling_options=modeling_options,
    #         wind_rose=wind_rose,
    #         case_title='loads',
    #     )
    # )

    model.add_subsystem(  # constraints for turbine proximity
        "spacing_constraint",
        ard.layout.spacing.TurbineSpacing(
            modeling_options=modeling_options,
        ),
    )
    model.connect("layout2aep.x_turbines", "spacing_constraint.x_turbines")
    model.connect("layout2aep.y_turbines", "spacing_constraint.y_turbines")

    model.add_subsystem(  # turbine capital costs component
        "tcc",
        ard.cost.wisdem_wrap.TurbineCapitalCosts(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
        ],
    )

    model.add_subsystem(  # LandBOSSE component
        "landbosse",
        # ard.cost.wisdem_wrap.LandBOSSE(),
        LandBOSSEWithSpacingApproximations(modeling_options=modeling_options),
    )
    model.connect(  # effective primary spacing for BOS
        "optiwindnet_coll.total_length_cables",
        "landbosse.total_length_cables",
    )

    model.add_subsystem(  # operational expenditures component
        "opex",
        ard.cost.wisdem_wrap.OperatingExpenses(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "opex_per_kW",
        ],
    )

    model.add_subsystem(  # cost metrics component
        "financese",
        ard.cost.wisdem_wrap.PlantFinance(),
        promotes_inputs=[
            "turbine_number",
            "machine_rating",
            "tcc_per_kW",
            "offset_tcc_per_kW",
            "opex_per_kW",
        ],
    )
    model.connect("AEP_farm", "financese.plant_aep_in")
    model.connect("landbosse.total_capex_kW", "financese.bos_per_kW")

    # build out the problem based on this model
    prob = om.Problem(model)
    prob.setup()

    ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
    ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

    # set up the working/design variables
    prob.set_val("spacing_primary", 7.0)
    prob.set_val("spacing_secondary", 7.0)
    prob.set_val("angle_orientation", 0.0)

    prob.set_val("optiwindnet_coll.x_substations", [100.0])
    prob.set_val("optiwindnet_coll.y_substations", [100.0])

    # run the model
    prob.run_model()

    # om.n2(prob)

    # collapse the test result data
    test_data = {
        "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
        "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
        "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
        "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
        "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
        "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
        "coll_length": float(
            prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
        ),
        "turbine_spacing": float(
            np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
        ),
    }

    print("\n\nRESULTS:\n")
    pp.pprint(test_data)
    print("\n\n")

    optimize = True  # set to False to skip optimization

    if optimize:
        # now set up an optimization driver

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = optimizer
        prob.driver.options["maxiter"] = maxiters
        # prob.driver.options["debug_print"] = ['desvars', 'nl_cons', 'ln_cons', 'objs', 'totals']

        prob.driver.options["debug_print"] = ['objs']

        # prob.driver = om.DifferentialEvolutionDriver()
        # prob.driver.options["max_gen"] = 30  # DEBUG!!!!! short
        # prob.driver.options["pop_size"] = 15  # DEBUG!!!!! short
        # # prob.driver.options["Pc"] = 0.5
        # # prob.driver.options["F"] = 0.5
        # prob.driver.options["run_parallel"] = False
        # prob.driver.options["debug_print"] = ["desvars", "nl_cons", "ln_cons", "objs"]

        prob.model.add_design_var("spacing_primary", lower=3.0, upper=10.0)
        prob.model.add_design_var("spacing_secondary", lower=3.0, upper=10.0)
        prob.model.add_design_var("angle_orientation", lower=-180.0, upper=180.0)
        prob.model.add_design_var("angle_skew", lower=-75.0, upper=75.0)
        prob.model.add_constraint(
            "spacing_constraint.turbine_spacing", units="m", lower=284.0 * 3.0
        )
        prob.model.add_constraint("optiwindnet_coll.total_length_cables", units="m", upper=cable_max_tot_length)
        # prob.model.add_constraint("landuse.area_tight", units="km**2", lower=50.0)
        # prob.model.add_objective("optiwindnet_coll.total_length_cables")
        prob.model.add_objective("tower_base_load", scaler=1e4)
        # prob.model.add_objective("AEP_farm", scaler=1e11)

        # create a recorder
        os.makedirs('optimization_demo_out/' + case_title, exist_ok=True)
        recorder = om.SqliteRecorder('optimization_demo_out/' + case_title + "/opt_results.sql")

        # add the recorder to the problem
        prob.add_recorder(recorder)
        # add the recorder to the driver
        prob.driver.add_recorder(recorder)

        # prob.driver.recording_options['includes'] = ['FLORISTowerBaseLoad.AEP_farm']

        model.layout2aep.add_recorder(recorder)
        model.optiwindnet_coll.add_recorder(recorder)

        # set up the problem
        prob.setup()

        ard.cost.wisdem_wrap.LandBOSSE_setup_latents(prob, modeling_options)
        ard.cost.wisdem_wrap.FinanceSE_setup_latents(prob, modeling_options)

        # set up the working/design variables initial conditions
        prob.set_val("spacing_primary", 7.0)
        prob.set_val("spacing_secondary", 7.0)
        prob.set_val("angle_orientation", 0.0)
        prob.set_val("angle_skew", 0.0)

        prob.set_val("optiwindnet_coll.x_substations", [100.0])
        prob.set_val("optiwindnet_coll.y_substations", [100.0])

        # run the optimization
        prob.run_driver()

        # collapse the test result data
        test_data = {
            "Tower_base_load_val": float(prob.get_val("tower_base_load", units="kN*m")[0]),
            "AEP_val": float(prob.get_val("AEP_farm", units="GW*h")[0]),
            "CapEx_val": float(prob.get_val("tcc.tcc", units="MUSD")[0]),
            "BOS_val": float(prob.get_val("landbosse.total_capex", units="MUSD")[0]),
            "OpEx_val": float(prob.get_val("opex.opex", units="MUSD/yr")[0]),
            "LCOE_val": float(prob.get_val("financese.lcoe", units="USD/MW/h")[0]),
            "area_tight": float(prob.get_val("landuse.area_tight", units="km**2")[0]),
            "coll_length": float(
                prob.get_val("optiwindnet_coll.total_length_cables", units="km")[0]
            ),
            "turbine_spacing": float(
                np.min(prob.get_val("spacing_constraint.turbine_spacing", units="km"))
            ),
        }

        # clean up the recorder
        prob.cleanup()

        # print the results
        print("\n\nRESULTS (opt):\n")
        pp.pprint(test_data)
        print("\n\n")

    optiwindnet.plotting.gplot(prob.model.optiwindnet_coll.graph)

    plt.show()


if __name__ == "__main__":

    run_example()
