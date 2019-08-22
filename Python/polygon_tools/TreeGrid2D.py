import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pltPolygon
from matplotlib.collections import PatchCollection
import shapefile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from typing import List, Tuple
import numpy as np
from pathlib import Path


class TreeGrid2D:
    def __init__(self, shape_file_name: str):

        self.file_name = shape_file_name

        self.raw_polygon_data = self._extract_polygon_data()

        self.polygon_list = self._extract_polygon()

    def _extract_polygon_data(self) -> List[List[Tuple]]:
        """The only one time open this shapefile.
        
        Read and store the list of polygon in "raw_polygon_data"
        
        Return: 
            list of polygon, where each polygon is a list of tuple,(x:float,y:float)
        """
        polygon_data = []
        print(self.file_name)
        with shapefile.Reader(self.file_name) as shp:

            for s in shp.shapeRecords():
                polygon = s.shape.points
                polygon_data.append(polygon)

        return polygon_data

    def _extract_polygon(self) -> List[Polygon]:
        """
        Use the data from 'raw_polygon_data' to constuct
            a list of shapely.geometry.polygon.Polygon object.
            Which is useful to check in a point is contain inside 
            a given polygon.
            
        Return:
            list of "shapely.geometry.polygon.Polygon" objects.
        """
        polygon_list = []

        for polygon_data in self.raw_polygon_data:
            polygon_list.append(Polygon(polygon_data))

        return polygon_list

    def plot(self, distance_from_center: int = 40,
             fig_size: tuple = (10, 10)) -> None:
        """
        Ploting the polygons
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim([-distance_from_center, distance_from_center])
        ax.set_ylim([-distance_from_center, distance_from_center])
        patches = []

        for i in range(len(self.raw_polygon_data)):
            #     polygon = pltPolygon(np.random.rand(num_sides ,2), True)
            polygon = pltPolygon(np.array(self.raw_polygon_data[i]), True)
            patches.append(polygon)

        p = PatchCollection(patches, cmap=plt.cm.jet, alpha=0.4)

        # colors = 100*np.random.rand(len(patches))
        # p.set_array(np.array(colors))

        ax.add_collection(p)

        plt.show()


if __name__ == "__main__":

    # path_to_test_shape_files = Path('.')
    # shape_file_name = 'test_concave_hulls.shp'
    # full_path = path_to_test_shape_files / shape_file_name
    # print(full_path)
    g = TreeGrid2D('./test_concave_hulls.shp')

    g.plot(distance_from_center=20)