from polygon_tools import TreeGrid2D, matrixrize_polygon
import pprint
if __name__ == "__main__":
    g = TreeGrid2D('./test_concave_hulls.shp')

    distance = 10
    width = 0.5
    g.plot(distance_from_center=distance)

    minx = -distance
    miny = -distance
    maxx = distance
    maxy = distance

    cells = matrixrize_polygon(minx, miny, maxx, maxy, g, width)

    pp = pprint.PrettyPrinter(width=len(cells[0]) * 4, compact=True)
    pp.pprint(cells)
