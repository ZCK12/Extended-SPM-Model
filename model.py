import math
import numpy as np
from numpy.typing import ArrayLike
from typing import NewType, Tuple
from matplotlib.path import Path

# Polygon object used to define boundary of the model surface
# The polygon object should be an array of tuples shape (N,2) with
# coordinates that define a closed polygon in R2 space.
Polygon = NewType("Polygon", ArrayLike[int])


class spm_model:
    """
    Class for simulating the Sand Pile Model (SPM) on various 2D surfaces.

    The Sand Pile Model (also known as the Abelian Sandpile Model or the Bak-Tang-Wiesenfeld Model)
    is a mathematical model used to study self-organized criticality. Grains of sand are dropped onto
    a finite grid. When any cell in the grid accumulates more sand than a specific threshold, it "topples"
    and distributes its sand to its neighbors. This might lead to subsequent toppling in a chain reaction.
    The model is interesting because it leads to complex, large-scale patterns from simple rules.

    This class allows for simulations of the SPM on different boundary shapes like circles, ellipses, squares,
    and rectangles. Custom shapes can also be provided as 2D arrays of coordinates. The simulation space is
    rasterized based on the specified boundary, creating a boolean matrix indicating the valid cells within the
    boundary. The `cellular_matrix` then represents the pile height in each cell of the simulation.

    Attributes:
    - polygons: Predefined polygonal boundaries.
    - width: Width of the simulation grid.
    - height: Height of the simulation grid.
    - raster_matrix: A boolean matrix indicating whether a cell is within the boundary or not.
    - cellular_matrix: Matrix representing the pile height in each cell.

    Methods:
    - __transform_polygon: Transforms a polygon to fit the desired simulation space.
    - __rasterize_path: Rasterizes the polygonal path to produce the `raster_matrix`.

    Parameters:
    - surface_boundary (Polygon | str): The boundary shape of the model. Can be one of the predefined shapes
      or a custom 2D array of coordinates.
    - width (int): Width of the simulation grid.
    - height (int): Height of the simulation grid.
    """

    polygons: dict[str, np.ndarray] = {
        "circle":       np.array([[math.cos(x), math.sin(x)] for x in np.linspace(0,2*np.pi,20,endpoint=False)]),
        "square":       np.array([(1,1), (-1,1), (-1,-1), (1,-1)])
        }


    def __init__(self, surface_boundary: Polygon | str, width: int, height: int):
        self.width = width
        self.height = height

        # Type checking the surface_boundary variable
        if not isinstance(surface_boundary, (Polygon, str)):
            raise TypeError("Surface boundary argument must be of type Polygon or String")

        # If the provided surface boundary is a string, then it should reference
        # a premade polygonal boundary.
        if type(surface_boundary) == str:
            surface_boundary = surface_boundary.lower()
            if surface_boundary in spm_model.polygons.keys():
                surface_boundary = spm_model.polygons[surface_boundary]
            else:
                e_str = f"Unknown polygon surface_boundary, must be of type Polygon or one of {spm_model.polygons.keys()}"
                raise ValueError(e_str)

        # Various checks for the validity of the custom polygonal boundary
        else:
            if len(surface_boundary.shape) != 2 or surface_boundary.shape[1] != 2:
                raise ValueError("Polygon must be a 2D array with shape (N, 2).")

            if not issubclass(surface_boundary.dtype.type, (np.integer, np.floating)):
                raise TypeError("Polygon must contain integer or float coordinates.")

            if surface_boundary.shape[0] < 3:
                raise ValueError("Polygon must have at least 3 vertices.")

        # We first transform the polygon to scale it up to our desired width and height. The scaled
        # polygon will be 2 units short in each dimension to allow for a buffer zone in our rasterised matrix.
        surface_boundary = self.__transform_polygon(surface_boundary, self.width-2, self.height-2)

        # We use this new scaled polygon to define a closed path for rasterising with.
        surface_path = Path(surface_boundary, closed=True)

        # Now we can finally rasterise our simulation surface using the polygon path (with a 1 unit buffer zone)]
        # The spm_matrix is an ndarray of booleans that indicate whether a given cell within our simulation surface or not.
        self.raster_matrix = self.__rasterize_path(surface_path, self.width, self.height)

        # The cellular_matrix is used for running the Sand Pile Model, and contains the pile height in each cell.
        self.cellular_matrix = np.zeros((width, height), dtype=np.uint8)
        self.condition_matrix = None # TODO implement condition_matrix and interactions


    @staticmethod
    def __transform_polygon(polygon: np.ndarray, desired_width: int, desired_height: int) -> np.ndarray:
        """
        Transforms a polygon to ensure all its vertices are in the first quadrant, and it matches the
        desired height and width of the simulation space.

        :param polygon: ndarray of shape (N, 2) representing the polygon.
        :param desired_width: Desired width of the scaled polygon.
        :param desired_height: Desired height of the scaled polygon.
        :return: Transformed polygon as ndarray.
        """
        min_x = np.min(polygon[:, 0])
        min_y = np.min(polygon[:, 1])

        translated_polygon = polygon.copy()
        translated_polygon[:, 0] -= min_x
        translated_polygon[:, 1] -= min_y

        current_width = np.max(translated_polygon[:, 0]) - np.min(translated_polygon[:, 0])
        current_height = np.max(translated_polygon[:, 1]) - np.min(translated_polygon[:, 1])

        # Calculate scaling factors for x and y axes
        x_scale = desired_width / current_width
        y_scale = desired_height / current_height

        # Apply the scaling factors
        scaled_polygon = translated_polygon.copy()
        scaled_polygon[:, 0] *= x_scale
        scaled_polygon[:, 1] *= y_scale

        return scaled_polygon


    @staticmethod
    def __rasterize_path(poly_path: Path, width: int, height: int) -> np.ndarray:
        """
        Convert a polygon into a 2D array of boolean values. The boolean
        indicates whether that cell in within our surface boundary or not.

        :param polygon: List of coordinate tuples representing the polygon.
        :param width: Width of the output array.
        :param height: Height of the output array.
        :return: 2D numpy array of boolean values.
        """
        # Create an empty 2D array of the desired size
        raster = np.zeros((width, height), dtype=bool)

        # Loop through each cell in the array, with 1 index offsets
        # to ensure a buffer of 1 cell on each edge of the array.
        for x in range(width-1):
            for y in range(height-1):
                # Check if the cell's center is inside the path
                if poly_path.contains_point((x + 0.5, y + 0.5)):
                    raster[x+1, y+1] = True

        return raster
