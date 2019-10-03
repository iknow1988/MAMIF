import pickle
import pprint

import numpy as np

from polygon_tools import TreeGrid2D, matrixrize_polygon

if __name__ == "__main__":
    g = TreeGrid2D('./test_concave_hulls.shp')

    distance = 50
    width = 0.2
    # g.plot(distance_from_center=distance)

    minx = -distance
    miny = -distance
    maxx = distance
    maxy = distance

    cells = matrixrize_polygon(minx, miny, maxx, maxy, g, width)
    cells_np = np.array(cells)
    print("Shape : {}".format(cells_np.shape))
    # pp = pprint.PrettyPrinter(width=len(cells[0]) * 4, compact=True)
    # pp.pprint(cells)
    with open("shape", 'wb') as f:
        pickle.dump(cells, f)
