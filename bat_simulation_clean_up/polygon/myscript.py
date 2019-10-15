import os
import pickle
import pprint

import numpy as np

from polygon_tools import TreeGrid2D, matrixrize_polygon


def processing_shape_file(dir_name: str, file_name: str):

    g = TreeGrid2D(dir_name + '/' + fname)

    distance = 30  # distance to a side of boundary
    width = 0.05
    # g.plot(distance_from_center=distance)

    minx = -distance
    miny = -distance
    maxx = distance
    maxy = distance

    cells = matrixrize_polygon(minx, miny, maxx, maxy, g, width)
    cells_np = np.array(cells)
    print("Shape : {}".format(cells_np.shape))
    print("Number of 1's {}".format(np.sum(cells_np)))
    # pp = pprint.PrettyPrinter(width=len(cells[0]) * 4, compact=True)
    # pp.pprint(cells)

    output_file = file_name.replace('.shp', '')
    with open(output_file, 'wb') as f:
        pickle.dump(cells, f)
    #


if __name__ == "__main__":
    dir_name = 'shape_files'
    directory = os.listdir(dir_name)
    # for fname in directory:
    #     if fname.endswith('.shp'):
    #         processing_shape_file(dir_name, fname)
    for fname in directory:
        if fname == 'carpark.shp':
            processing_shape_file(dir_name, fname)
