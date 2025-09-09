import numpy as np
import openmdao.api as om

from shapely import length
import shapely.geometry as sg

import ard.layout.templates as templates


class FullFarmLayout(templates.LayoutTemplate):
    """
    """

    def initialize(self):
        """Initialization of OM component."""
        super().initialize()

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # add four-parameter grid farm layout DVs
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default complex step for the layout tools, since they're often algebraic
        self.declare_partials("*", "*", method="cs")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
        lengthscale_spacing_streamwise = inputs["spacing_primary"] * D_rotor
        lengthscale_spacing_spanwise = inputs["spacing_secondary"] * D_rotor

        N_square = int(np.sqrt(self.N_turbines))  # floors

        count_y, count_x = np.meshgrid(
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
            np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2)),
        )

        if self.N_turbines == N_square**2:
            pass
        elif self.N_turbines <= N_square * (N_square + 1):
            # grid farm is a little bit above the last square... add a trailing
            # row.
            count_x = np.vstack([count_x, ((N_square + 1) / 2) * np.ones((N_square,))])
            count_y = np.vstack(
                [count_y, np.arange(-((N_square - 1) / 2), ((N_square + 1) / 2))]
            )
            count_x = count_x.flatten()
            count_y = count_y.flatten()
        else:
            # grid farm is nearly the next square... oversize and cut the last
            count_y, count_x = np.meshgrid(
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
                np.arange(-((N_square) / 2), ((N_square + 2) / 2)),
            )
        count_x = count_x.flatten()[: self.N_turbines]
        count_y = count_y.flatten()[: self.N_turbines]

        angle_skew = -np.pi / 180.0 * inputs["angle_skew"]
        Bmtx = np.array(
            [
                [1.0, 0.0],
                [np.tan(float(angle_skew[0])), 1.0],
            ]
        ).squeeze()

        xi_positions = count_x * lengthscale_spacing_spanwise
        yi_positions = count_y * lengthscale_spacing_streamwise

        angle_orientation = np.pi / 180.0 * inputs["angle_orientation"]
        Amtx = np.array(
            [
                [np.cos(angle_orientation), np.sin(angle_orientation)],
                [-np.sin(angle_orientation), np.cos(angle_orientation)],
            ]
        ).squeeze()
        xyp = Amtx @ (Bmtx @ np.vstack([xi_positions, yi_positions]))

        outputs["x_turbines"] = xyp[0, :].tolist()
        outputs["y_turbines"] = xyp[1, :].tolist()

        # outputs["spacing_effective_primary"] = inputs["spacing_primary"]
        # outputs["spacing_effective_secondary"] = np.sqrt(
        #     inputs["spacing_secondary"] ** 2.0 / np.cos(angle_skew) ** 2.0
        # )


class FullFarmLanduse(templates.LanduseTemplate):
    """
    Landuse class for full Cartesian grid farm layout.

    This is a class to compute the landuse area of a fully specified Cartesian
    grid farm layout.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from
        `templates.LayoutTemplate`)
    N_turbines : int
        the number of turbines that should be in the farm layout (inherited from
        `templates.LayoutTemplate`)

    Inputs
    ------
    x_turbines : np.ndarray
        a 1-D numpy array that represents that x (i.e. Easting) coordinate of
        the location of each of the turbines in the farm in meters
    y_turbines : np.ndarray
        a 1-D numpy array that represents that y (i.e. Northing) coordinate of
        the location of each of the turbines in the farm in meters

    Outputs
    -------
    area_tight : float
        the area in square kilometers that the farm occupies based on the
        circumscribing geometry with a specified (default zero) layback buffer
        (inherited from `templates.LayoutTemplate`)
    """

    def setup(self):
        """Setup of OM component."""
        super().setup()

        # add the full layout inputs
        self.add_input(
            "x_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in x-direction",
        )
        self.add_input(
            "y_turbines",
            np.zeros((self.N_turbines,)),
            units="m",
            desc="turbine location in y-direction",
        )

    def setup_partials(self):
        """Derivative setup for OM component."""

        # default complex step for the layout-landuse tools, since they're often algebraic
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computation for the OM component."""

        # extract the points from the inputs
        points = list(
            zip(
                list(inputs["x_turbines"]),
                list(inputs["y_turbines"]),
            )
        )

        # create a multi-point object
        mp = sg.MultiPoint(points)

        # create a laybacked geometry
        D_rotor = self.modeling_options["turbine"]["geometry"]["diameter_rotor"]
        lengthscale_layback = float(inputs["distance_layback_diameters"][0] * D_rotor)

        # area tight is equal to the convex hull area for the points in sq. km.
        outputs["area_tight"] = (
            mp.convex_hull.buffer(lengthscale_layback).area / 1000**2
        )
