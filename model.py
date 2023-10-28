import math
import numpy as np
import matplotlib.pyplot as plt
import random

from numpy.typing import ArrayLike
from typing import NewType, Tuple
from matplotlib.path import Path



# Polygon object used to define boundary of the model surface
# The polygon object should be an array of tuples shape (N,2) with
# coordinates that define a closed polygon in R2 space.
Polygon = NewType("Polygon", np.ndarray[np.uint16])


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
        "circle":       np.array([[math.cos(x), math.sin(x)] for x in np.linspace(0,2*np.pi,65,endpoint=True)]),
        "square":       np.array([(1,1), (-1,1), (-1,-1), (1,-1)], dtype=np.float64)
        }


    def __init__(self, surface_boundary: Polygon | str, width: int, height: int, topple_height: int=4):
        self.width = width
        self.height = height
        self.topple_height = topple_height

        # Type checking the surface_boundary variable
        if not isinstance(surface_boundary, (np.ndarray, str)):
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
        surface_path = Path(surface_boundary, closed=False)

        # Now we can finally rasterise our simulation surface using the polygon path (with a 1 unit buffer zone)]
        # The spm_matrix is an ndarray of booleans that indicate whether a given cell within our simulation surface or not.
        self.raster_matrix = self.__rasterize_path(surface_path, self.width, self.height)

        # The cellular_matrix is used for running the Sand Pile Model, and contains the pile height in each cell.
        self.cellular_matrix = np.zeros((width, height), dtype=np.uint16)
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


    def directional_size(self, azimuth: float):
        # Convert azimuth to radians
        azimuth_rad = np.deg2rad((450 - azimuth) % 360)

        # Initialize min and max projections
        min_proj = np.inf
        max_proj = -np.inf

        # Loop through each point in the array
        for y in range(self.height):
            for x in range(self.width):
                if self.raster_matrix[x, y]:
                    # Calculate projection of point (x, y) on line with azimuth angle
                    proj = x * np.cos(azimuth_rad) + y * np.sin(azimuth_rad)

                    # Update min and max projections
                    min_proj = min(min_proj, proj)
                    max_proj = max(max_proj, proj)

        # Length of the shape in the azimuth direction is the difference between max and min projections
        return (max_proj - min_proj) + 1 # +1 to account for measurements in the centre of tiles, rather than edges.


    def resolve_topples(self, topple_cell_set: set[tuple[int, int]]=None):
        """
        Process the cellular_matrix to resolve any unstable cells, where instability is defined by a cell's
        value being greater than or equal to the topple_height. An unstable cell will distribute its
        excess value to its four neighboring cells (up, down, left, and right). This process is repeated
        iteratively until all cells in the cellular_matrix are stable.
        """
        # Create a set (unordered, unique) of all the cells that are required to be toppled initially
        if topple_cell_set == None:
            topple_cell_set = set()
            for x_pos in range(self.width):
                for y_pos in range(self.height):
                    # Only cells that are inside our boundary are considered for toppling.
                    if (self.cellular_matrix[x_pos, y_pos] >= self.topple_height) and self.raster_matrix[x_pos, y_pos]:
                        topple_cell_set.add((x_pos, y_pos))
        else:
            # If a set is provided, then it must not be empty.
            assert len(topple_cell_set) > 0

        # Topple the matrix cell by cell, pushing new toppled cells to the stack
        while len(topple_cell_set) > 0:
            coord = topple_cell_set.pop()

            # Loops for as long as the current cell is unstable
            while self.cellular_matrix[coord] >= self.topple_height:
                self.cellular_matrix[coord] -= 4
                # Computes adjacent cell by adding an X and Y offset
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    adj_coord = (coord[0] + dx, coord[1] + dy)
                    self.cellular_matrix[adj_coord] += 1
                    # If the cell in question is unstable and is a valid interior cell, then it is added to the topple set.
                    if self.cellular_matrix[adj_coord] >= self.topple_height and self.raster_matrix[adj_coord]:
                        topple_cell_set.add(adj_coord)

        # If the set is empty, then the configuration is stable
        return


    def add_sand(self, coord: tuple[int, int], amount=1) -> bool:
        """
        Add a specified amount of sand to the cell at the given coordinates and resolve any resulting toppling. If the
        target cell is outside the boundary area, an IndexError is raised. If the sand addition results in the cell becoming
        unstable, the resolve_topples method is called to stabilize the cellular_matrix.

        Parameters:
            coord (tuple[int, int]): The (x, y) coordinates of the target cell in the cellular_matrix.
            amount (int, optional): The amount of sand to add to the target cell. Default is 1.

        Returns:
            bool: A boolean representing whether or not cells toppled (and were resolved) as a result of the addition.
        """
        if not self.raster_matrix[coord]:
            raise IndexError(f"Provided coordinate {coord} is outside of the boundary area")

        self.cellular_matrix[coord] += amount
        if self.cellular_matrix[coord] >= self.topple_height:
            self.resolve_topples({coord})
            return True

        return False


my_spm = spm_model("circle", 8, 8)
print(my_spm.directional_size(0))
for i in range(100000):
    try:
        my_spm.add_sand((random.randrange(1,200),random.randrange(1,200)), 1)
    except:
        pass

    if i%5000 == 0:
        fig, ax = plt.subplots()
        cax = ax.imshow(my_spm.raster_matrix, cmap='viridis')

        # Add a color bar
        cbar = fig.colorbar(cax)

        # Show the plot
        plt.show()
        if input() == "stop":
            pass
            break

