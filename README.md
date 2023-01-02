## Flood Emergency Planning	

![](http://rnk0xriaw.hn-bkt.clouddn.com/0096.png)

### Geometry

```python
# geometry.py

from shapely.geometry import Point
from constant import speed


class PlainPoint:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y
        self.__geometry = Point(x, y)

    def get_geometry(self):
        return self.__geometry


class PointWithHeight(PlainPoint):
    def __init__(self, x, y, height):
        super().__init__(x, y)
        self.__height = height

    def get_height(self):
        return self.__height

    def set_height(self, new_height):
        self.__height = new_height


class ItnPoint(PointWithHeight):
    def __init__(self, x, y, height, fid):
        super().__init__(x, y, height)
        self.__fid = fid

    def get_fid(self):
        return self.__fid


class PlainLine:
    def __init__(self, p1, p2):
        self.__p1 = p1
        self.__p2 = p2
        self.__geometry = [p1, p2]

    def get_geometry(self):
        return self.__geometry


class DirectedLine(PlainLine):
    def __init__(self, start_node, end_node):
        super().__init__(start_node, end_node)
        self.__start_node = self.get_geometry()[0]
        self.__end_node = self.get_geometry()[1]

    def get_start_node(self):
        return self.__start_node

    def get_end_node(self):
        return self.__end_node


class Edge(DirectedLine):
    def __init__(self, fid, start_node, end_node, length):
        super().__init__(start_node, end_node)
        self.__fid = fid
        self.__length = length
        self.__height_diff = end_node.get_height() - start_node.get_height()

    def get_fid(self):
        return self.__fid

    def get_length(self):
        return self.__length

    def get_height_diff(self):
        return self.__height_diff

    def add_weight(self):
        base_weight = self.__length / speed

        # if height_diff is positive, it's ascending -> base_weight+height_weight
        # if height_diff is negative, it's descending -> base_weight-abs(height_weight)
        height_weight = (self.__height_diff / 10)

        if self.__height_diff == 0:
            return base_weight
        else:
            return base_weight + height_weight
```

### Task 1: User Input

```python
# task_1.py

import geopandas as gpd
from shapely.geometry import Point, Polygon
from constant import map_x_min, map_x_max, map_y_min, map_y_max

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')


def generate_box(x_min, y_min, x_max, y_max):
    return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))


def user_input():
    try:
        x = float(input('please input a coordinates x: '))
        y = float(input('please input a coordinates y: '))

        box = generate_box(x_min=map_x_min, x_max=map_x_max, y_min=map_y_min,
                           y_max=map_y_max)
        input_point = Point(x, y)

        # in isle area
        if isle_of_wight.contains(input_point).iloc[0] or isle_of_wight.touches(input_point).iloc[0]:
            # in bounding box
            if box.contains(input_point) or box.touches(input_point):
                print(
                    f'>>>>>>>>>>>>> Input Point({input_point.x},{input_point.y}) is inside given bound area in TASK_1')
                return input_point, 1

            else:
                print(f'>>>>>>>>>>>>> Input Point({input_point.x},{input_point.y}) is inside given isle area in TASK_1')
                return input_point, 2
        else:
            raise Exception(f'Input Point({input_point.x},{input_point.y}) is outside given area')
    except Exception as error:
        raise Exception(f'{error} ==> TASK_1')
```

1. Generate the maximum area
2. Identify whether user_input_point is in the given area via `GeoDataframe.contains&GeoDataframe.touches`

### Task 2: Highest Point Identification

```python
# task_2.py

from shapely.geometry import Point, Polygon
import rasterio
from rasterio import mask
import numpy as np
import time

from geometry import PointWithHeight

elevation = rasterio.open('Material/elevation/SZ.asc')


def get_xy_from_height(height_value, height_data, transformer=None):
    highest_point_xy = np.where(height_data == height_value)
    row = highest_point_xy[1][0]
    col = highest_point_xy[2][0]

    if transformer is not None:
        transformed_x, transformed_y = rasterio.transform.xy(transformer, row, col)

        return transformed_x, transformed_y

    return height_data.xy(row, col)


def get_height_from_xy(x, y, height_data=elevation, hash_table=None):
    (row, col) = height_data.index(x, y)  # fast

    if hash_table is not None:
        return hash_table[(row, col)]

    point_height = height_data.read(1)[row, col]  # slow
    return point_height


# create a hash table to store all the height info of (row,col) in elevation file
def define_hash():
    start = time.perf_counter()

    hash_table = {}
    elevation_matrix = elevation.read(1)  # 5000,9000

    print('>>>>>>>>>>>>> Generating row_col_height_hash_table in TASK_2')
    for row in range(len(elevation_matrix)):
        for col in range(len(elevation_matrix[0])):
            hash_table[(row, col)] = elevation_matrix[row][col]

    spend_time = time.perf_counter() - start
    print(f'>>>>>>>>>>>>> Generated row_col_height_hash_table! Spent {spend_time.__format__(".2f")} s')

    return hash_table


def highest_point_identify(input_point, buffer_radius):
    """
    :param input_point: shapely.geometry.Point,user input point form task_1
    :param buffer_radius: radius of buffer, default is 5km
    :return: (PointWithHeight,PointWithHeight,{}),
                first is the point user input with its height info,
                second is the highest point within given radius,
                third is a hash table storing (row,col)=>height
    """

    try:
        print('>>>>>>>>>>>>> Progressing highest_point_identify in TASK_2')
        buffer = input_point.buffer(buffer_radius)

        elev_boundary_right_top = (elevation.bounds[2], elevation.bounds[3])
        elev_boundary_left_top = (elevation.bounds[0], elevation.bounds[3])
        elev_boundary_left_bottom = (elevation.bounds[0], elevation.bounds[1])
        elev_boundary_right_bottom = (elevation.bounds[2], elevation.bounds[1])

        elevation_mbr = Polygon(
            [elev_boundary_right_top, elev_boundary_left_top, elev_boundary_left_bottom, elev_boundary_right_bottom])

        row_col_height_hash_table = define_hash()

        input_point_height = get_height_from_xy(input_point.x, input_point.y, hash_table=row_col_height_hash_table)
        point_input_with_height = PointWithHeight(input_point.x, input_point.y, input_point_height)

        # Prevent buffer from exceeding the elevation range
        mask_cropped_by_mbr = buffer.intersection(elevation_mbr)
        masked_elevation_raster, transformer = rasterio.mask.mask(dataset=elevation, shapes=[mask_cropped_by_mbr],
                                                                  crop=True, nodata=0, filled=False)

        highest_value = np.max(masked_elevation_raster)
        highest_point_x, highest_point_y = get_xy_from_height(height_value=highest_value,
                                                              height_data=masked_elevation_raster,
                                                              transformer=transformer)

        highest_point = PointWithHeight(highest_point_x, highest_point_y, highest_value)

        return point_input_with_height, highest_point, row_col_height_hash_table
    except Exception as error:
        raise Exception(f'{error} ==> TASK_2')
  
```

1. `get_height_from_xy()`: 
   - This step provides a function to get elevation information through coordinates.
   - Since the operation Raster.index(x, y) is relatively faster than Raster.read(1)[row,col], we stores all the height information in a "Hash Table" via a dictionary, whichgreatly improve the operation efficiency when obtaining elevation information at each ITN-point in Task 4, from 10 minutes to nearly 20 seconds.
2. `numpy.max`：Returns the maximum value in the array, which can be used when getting the max height in elevation.
3. `rasterio.mask.mask`：By establishing a 5km buffer area as the mask from the center point, the height equal to maximum value is where the highest point is.

### Task 3: Nearest Integrated Transport Network

```python
# task_3.py

from rtree import index
from shapely.geometry import Point
import json
from geometry import ItnPoint
from task_2 import highest_point_identify, get_height_from_xy


def nearest_itn_node(input_point, rc_height_hash_table=None):
    """
    :param input_point: Point(x,y) which is shapely.geometry.Point
    :param rc_height_hash_table: hash table with (row,col):height
    :return ItnPoint，nearest itn to input_node with height info
    """

    try:
        print('>>>>>>>>>>>>> Progressing nearest_itn_node in TASK_3')

        # read itn info
        with open('Material/itn/solent_itn.json') as file:
            itn_json = json.load(file)

        itn_rtree_index = index.Index()
        itn_road_nodes = itn_json['roadnodes']

        for itn_node_index, itn_node_info in enumerate(itn_road_nodes.items()):
            node_coordinates = (itn_node_info[1]['coords'][0], itn_node_info[1]['coords'][1])
            itn_rtree_index.insert(id=itn_node_index, coordinates=node_coordinates, obj=itn_node_info[0])

        # find nearest itn to input_node via rtree
        nearest_itn_to_input = None

        for fid in itn_rtree_index.nearest(coordinates=(input_point.x, input_point.y), num_results=1, objects='raw'):
            coordinates = itn_road_nodes[fid]['coords']
            x, y = coordinates

            nearest_itn_to_input = ItnPoint(x, y, get_height_from_xy(x, y, hash_table=rc_height_hash_table), fid)

        return nearest_itn_to_input
    except Exception as error:
        raise Exception(f'{error} ==> TASK_3')

```

1. `index.Index.nearest(coordinates, num_results)`：
   - coordinates：The point input by user, which is, need to obtain the nearest point to that.
   - num_results：The number of returned results can be set to 1.
   - Return: generator

### Task 4: Shortest Path

```python
# task_4.py

import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx
from geometry import ItnPoint, Edge
import json
from task_2 import highest_point_identify, get_height_from_xy
from task_3 import nearest_itn_node
from constant import CRS_BNG


# generate a GeoDataframe of given path of several LineString
def generate_dataframe(graph, path, road_links):
    short_distance_path_fids = []
    short_distance_path_geometry = []
    for i in range(len(path) - 1):
        pre_node = path[i]
        next_node = path[i + 1]
        edge_fid = graph.edges[pre_node, next_node]['fid']
        short_distance_path_fids.append(edge_fid)
        road = LineString(road_links[edge_fid]['coords'])
        short_distance_path_geometry.append(road)

    result_df = gpd.GeoDataFrame({'fid': short_distance_path_fids, 'geometry': short_distance_path_geometry})
    result_df.crs = CRS_BNG

    return result_df


def shortest_path(point_start, point_end, rc_height_hash_table=None):
    """
    :param point_start: ItnPoint
    :param point_end: ItnPoint
    :param rc_height_hash_table: hash table with (row,col):height
    :return short_distance_path and short_time_path，GeoDataframe whose geometry columns are all LineString
    """

    try:
        print('>>>>>>>>>>>>> Progressing shortest_path in TASK_4')

        with open('Material/itn/solent_itn.json') as file:
            itn_json = json.load(file)

        itn_road_nodes = itn_json['roadnodes']
        itn_road_links = itn_json['roadlinks']

        edge_list = []

        for itn_link_index, itn_link_info in itn_road_links.items():
            start_x = itn_road_nodes[itn_link_info['start']]['coords'][0]
            start_y = itn_road_nodes[itn_link_info['start']]['coords'][1]
            end_x = itn_road_nodes[itn_link_info['end']]['coords'][0]
            end_y = itn_road_nodes[itn_link_info['end']]['coords'][1]
            start_h = get_height_from_xy(start_x, start_y, hash_table=rc_height_hash_table)
            end_h = get_height_from_xy(end_x, end_y, hash_table=rc_height_hash_table)

            start_node = ItnPoint(start_x, start_y, start_h, itn_link_info['start'])
            end_node = ItnPoint(end_x, end_y, end_h, itn_link_info['end'])

            edge_list.append(
                Edge(fid=itn_link_index, start_node=start_node, end_node=end_node, length=itn_link_info['length']))

        # work out the weight of edge and get shortest path via networkx
        graph = networkx.DiGraph()
        for edge in edge_list:
            weight = edge.add_weight()

            edge_start_node = edge.get_geometry()[0]
            edge_end_node = edge.get_geometry()[1]

            graph.add_edge(edge_start_node.get_fid(), edge_end_node.get_fid(), fid=edge.get_fid(),
                           length=edge.get_length(),
                           weight=weight)
            graph.add_edge(edge_end_node.get_fid(), edge_start_node.get_fid(), fid=edge.get_fid(),
                           length=edge.get_length(),
                           weight=weight)

        short_distance_path = networkx.dijkstra_path(G=graph, source=point_start.get_fid(), target=point_end.get_fid(),
                                                     weight='length')
        short_time_path = networkx.dijkstra_path(G=graph, source=point_start.get_fid(), target=point_end.get_fid(),
                                                 weight='time')

        short_distance_path_df = generate_dataframe(graph, short_distance_path, itn_road_links)
        short_time_path_df = generate_dataframe(graph, short_time_path, itn_road_links)

        return short_distance_path_df, short_time_path_df
    except Exception as error:
        raise Exception(f'{error} ==> TASK_4')

```

1. `Edge class`: Store the final weight of path
2. `weight`：weight_base + weight_height
   - weight_base：length / speed which is the basic time. (metre/per minute)
   - weight_height：cost of height ascending or descending = height_difference / 10  (metre/per minute)
3. `networkx.dijkstra_path(G, source, target, weight)`：
4. For each point, the process of obtaining its altitude information is relatively slow as mentioned in TASK_2

   - Reason for slowness: it is required to excute `elevation.read(1)[row,col]` every time when calculate the height of each given ITN-point

   - Optimization: Storing all `(row,col)->height` information via a hash table when first reading all height data in TASK_2.
     - Cost of searching data in a hash table=O(1)

### Task 5: Map Plotting

```python
# task_5.py

import geopandas as gpd
import rasterio
from plotter import Plotter
from constant import CRS_BNG, buffer_radius, map_view_range, map_x_min, map_x_max, map_y_min, map_y_max
from geometry import PlainPoint
import matplotlib.patches as mpatches

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')

background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')


def plot_result(start_point, end_point):
    try:
        plotter = Plotter(CRS_BNG)
        fig, ax = plotter.get_figure()

        buffer_circle = mpatches.Circle((start_point.get_geometry().x, start_point.get_geometry().y), buffer_radius,
                                        alpha=.3)
        plotter.add_artist(buffer_circle)

        start_x = start_point.get_geometry().x
        start_y = start_point.get_geometry().y

        # set range 20km*20km
        x_min = max(map_x_min, start_x - map_view_range / 2)
        x_max = min(map_x_max, start_x + map_view_range / 2)
        y_min = max(map_y_min, start_y - map_view_range / 2)
        y_max = min(map_y_max, start_y + map_view_range / 2)

        # reset map view to 20*20 km2 if out of range
        if x_max - x_min != map_view_range:
            if x_min == map_x_min:
                x_max += map_x_min - (start_x - map_view_range / 2)
            if x_max == map_x_max:
                x_min -= (start_x + map_view_range / 2) - map_x_max
        if y_max - y_min != map_view_range:
            if y_min == map_y_min:
                y_max += map_y_min - (start_y - map_view_range / 2)
            if y_max == map_y_max:
                y_min -= (start_y + map_view_range / 2) - map_y_max
        plotter.set_xy_lim(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # add background of isle of wight
        plotter.add_raster(background, cmap='terrain')

        # mark start and end points on map
        plotter.add_point(start_point.get_geometry().x, start_point.get_geometry().y, color='red', markersize=4)
        plotter.add_point(end_point.get_geometry().x, end_point.get_geometry().y, color='green', markersize=4)

        # add legend to plotter
        plotter.add_legend(legend_type='point', legend_label='start point', color='red', markersize=4)
        plotter.add_legend(legend_type='point', legend_label='end point', color='green', markersize=4)
        plotter.add_legend(legend_type='line', legend_label='road', color='brown')
        plotter.add_legend(legend_type='line', legend_label='shortest distance path', color='blue', linewidth=3)
        plotter.add_legend(legend_type='line', legend_label='shortest time path', color='red', linewidth=3)
        plotter.add_legend(legend_type='line', legend_label='if paths coincide', color='purple', linewidth=3)

        return plotter
    except Exception as error:
        raise Exception(f'{error} ==> TASK_5')
          
```

1. Plot given vector or raster data via class Plotter.
2. Also implement plotting compass, scale bar, legend.

### Task 6: Extend the Region

```python
# task_6.py
```

### main

```python
# main.py
		
from task_1 import user_input
from task_2 import highest_point_identify
from task_3 import nearest_itn_node
from task_4 import shortest_path
from task_5 import plot_result
from constant import buffer_radius
from exception_handler import ExceptionHandler


def main():
    try:
        # task 1
        user_input_p, state = user_input()

        # task 2
        input_p, highest_p, rc_height_hash_table = highest_point_identify(user_input_p, buffer_radius=buffer_radius)

        # task 3
        input_nearest_itn = nearest_itn_node(input_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)
        highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)

        # task 4
        short_distance_path, short_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                             rc_height_hash_table=rc_height_hash_table)

        # task 5
        plotter = plot_result(start_point=input_p, end_point=highest_p)

        plotter.add_vector(short_distance_path, linewidth=3, color='red', alpha=.7)
        plotter.add_vector(short_time_path, linewidth=3, color='blue', alpha=.6)

        plotter.show()
    except Exception as error:
        exc = ExceptionHandler(error)
        exc.log()


if __name__ == '__main__':
    main()
```

### Creativity

1. Using a hash table to store all the records which improves the efficiency.

   ```python
   def define_hash():
       hash_table = {}
       elevation_matrix = elevation.read(1)
   
       for row in range(len(elevation_matrix)):
           for col in range(len(elevation_matrix[0])):
               hash_table[(row, col)] = elevation_matrix[row][col]
   
       return hash_table
   ```

   

2. When plotting the background map 20km x 20km of the surrounding area, if the start point is in an area that is too marginal, it may result in less than 10 km in a certain direction(x or y). At this time, the display range in the opposite direction will be flexibly extended, so that the entire area is still 20km x 20km, so as to ensure that data can be displayed in the same range.

   ```python
           if x_max - x_min != map_view_range:
               if x_min == map_x_min:
                   x_max += map_x_min - (start_x - map_view_range / 2)
               if x_max == map_x_max:
                   x_min -= (start_x + map_view_range / 2) - map_x_max
           if y_max - y_min != map_view_range:
               if y_min == map_y_min:
                   y_max += map_y_min - (start_y - map_view_range / 2)
               if y_max == map_y_max:
                   y_min -= (start_y + map_view_range / 2) - map_y_max
           plotter.set_xy_lim(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
   ```

3. constant.py

   By storing a series of constants that may affect the running results of the software, such as the range data of the map, we can quickly make changes according to areas or situations .

   ```python
   from pyproj import CRS
   
   map_x_min = 430000
   map_x_max = 465000
   map_y_min = 80000
   map_y_max = 95000
   
   CRS_BNG = CRS('epsg:27700')
   
   buffer_radius = 5 * 1000
   speed = buffer_radius / 1 * 60
   map_view_range = 20 * 1000
   ```

4. weight in decent

   According to the Naismith’s rule, the uphill route will increase the weight of the route. Therefore, we add the corresponding rules in descending paths which can reduce the weight slightly.

   ```python
       def add_weight(self):
           base_weight = self.__length / speed
   
           # if height_diff is positive, it's ascending -> base_weight+height_weight
           # if height_diff is negative, it's descending -> base_weight-abs(height_weight)
           height_weight = (self.__height_diff / 10)
   
           if self.__height_diff == 0:
               return base_weight
           else:
               return base_weight + height_weight
   ```

### SO FAR

1. Tasks
   1. User Input. 
   2. Highest Point Identification. 
   3. Nearest Integrated Transport Network. 
   4. Shortest Path. 
   5. Map Plotting 
   6. Extend the Region. 
2. Basic marks
   - OOP: see the Class of all geometry, Class of Plotter, Class of Exception(Error Handler)
   - Regular commits in Git
   - Proper distribution of work
   - PEP8 style
3. Creativity marks
   - Using a hash table to store all the records which improves the efficiency.
   - When plotting the background map 20km x 20km of the surrounding area, if the start point is in an area that is too marginal, it may result in less than 10 km in a certain direction(x or y). At this time, the display range in the opposite direction will be flexibly extended, so that the entire area is still 20km x 20km, so as to ensure that data can be displayed in the same range.





