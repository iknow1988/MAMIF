import numpy as np
from typing import List, Tuple
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from .TreeGrid2D import TreeGrid2D
def mapping_from_point_to_cell(x_axis_used:List[float] , y_axis_used:List[float]\
                               , x:float,y:float)->Tuple[int,int]:

    try:
        x_cell_axis = x_axis_used.tolist().index(x)
    except:
        x_cell_axis = int(x_axis_used[np.abs(x_axis_used - x).argmin()])
    try:
        y_cell_axis = y_axis_used.tolist().index(y)
    except:
        y_cell_axis = int(y_axis_used[np.abs(y_axis_used - y).argmin()])

    return (x_cell_axis, y_cell_axis)


def make_polygon(minx, miny, maxx, maxy) -> Polygon:

    return Polygon([[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]])

def _generate_x_y_axis_sequence(minx:float , maxx:float ,\
                                      miny:float,maxy:float, width:float)->Tuple:
    """
    Return
        Tuple of np.array
    """

    minx = np.floor(minx)
    miny = np.floor(miny)
    maxx = np.ceil(maxx)
    maxy = np.ceil(maxy)

    x_axis = np.arange(minx, maxx + width, width)

    y_axis = np.arange(miny, maxy + width, width)

    return (x_axis, y_axis)

def _generate_list_of_points_in_square(minx:float , maxx:float ,\
                                      miny:float,maxy:float,width:float)->List[Point]:
    """
    Return
        List of points inside the square specified by the four points.
        (Note: the points on boundaries of maxx and maxy are included)
    """
    x_axis, y_axis = _generate_x_y_axis_sequence(minx, maxx, miny, maxy, width)

    # All combinations of x_axis and y_axis values
    point_grid = np.array(np.meshgrid(x_axis, y_axis)).T.reshape(-1, 2)

    assert (point_grid.shape[0] == len(x_axis) * len(y_axis))

    point_list = [Point(d) for d in point_grid]
    return point_list


def generate_list_of_blocked_points(polygon: Polygon, x_axis_used: np.array,
                                    y_axis_used: np.array,
                                    width: float) -> List[Point]:
    """
    """
    # Adjust the value for bounds.
    minx, miny, maxx, maxy = polygon.bounds
    minx = x_axis_used[np.abs(x_axis_used - minx).argmin()]
    maxx = x_axis_used[np.abs(x_axis_used - maxx).argmin()]
    miny = y_axis_used[np.abs(y_axis_used - miny).argmin()]
    maxy = y_axis_used[np.abs(y_axis_used - maxy).argmin()]

    # polygon = make_polygon(minx, miny, maxx, maxy)
    # Each point (x,y), x should be in x_axis_used and y should be in y_axis_used.
    points_in_square = _generate_list_of_points_in_square(
        minx, maxx, miny, maxy, width)

    points_in_polygon = [p for p in points_in_square if polygon.contains(p)]

    return points_in_polygon


def matrixrize_polygon( minx:float, miny:float, maxx:float, maxy:float,\
                      tree_grid:TreeGrid2D, width:float)->List[List[int]]:

    # Step 1
    x_axis, y_axis = _generate_x_y_axis_sequence(minx, maxx, miny, maxy, width)
    # Step 2
    matrix_cells = [[0 for i in range(len(x_axis))]
                    for j in range(len(y_axis))]

    for p in tree_grid.polygon_list:

        blocked_points = generate_list_of_blocked_points(p , x_axis_used=x_axis,\
                                                        y_axis_used=y_axis,width=width)

        K, J = len(x_axis) - 1, len(y_axis) - 1
        for p in blocked_points:

            x, y = mapping_from_point_to_cell(x_axis, y_axis, p.x, p.y)
            matrix_cells[J - y][x] = 1
    return matrix_cells
