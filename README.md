## Flood Emergency Planning
## Flood Emergency Planning

```python
# main.py
# requirement
import geopandas
import rasterio
import matplotlib.pyplot as plt
import shapely
import rtree
import networkx

# Material
# vector
isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')
# raster
background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')
```

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

from shapely.geometry import Point, Polygon


def generate_box(x_min, y_min, x_max, y_max):
    return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

def user_input():
    x = float(input('please input a coordinates x: '))
    y = float(input('please input a coordinates y: '))

    box = generate_box(x_min=430000, x_max=465000, y_min=80000, y_max=95000)
    input_point = Point(x, y)

    return input_point, box.contains(input_point) or box.touches(input_point)
```

1. Generate the maximum area
2. Identify whether user_input_point is in the given area via `GeoDataframe.contains&GeoDataframe.touches`

### Task 2: Highest Point Identification

```python
# task_2.py
from shapely.geometry import Point, Polygon
import geopandas as gpd
import rasterio
from rasterio import mask
import numpy as np
import time

from geometry import PointWithHeight

elevation = rasterio.open('Material/elevation/SZ.asc')
isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')


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


def highest_point_identify(input_point, buffer_radius=5 * 1000):
    """
    :param input_point: shapely.geometry.Point,user input point form task_1
    :param buffer_radius: radius of buffer, default is 5km
    :return: (PointWithHeight,PointWithHeight,{}),
                first is the point user input with its height info,
                second is the highest point within given radius,
                third is a hash table storing (row,col)=>height
    """

    buffer = input_point.buffer(buffer_radius)

    elev_boundary_right_top = (elevation.bounds[2], elevation.bounds[3])
    elev_boundary_left_top = (elevation.bounds[0], elevation.bounds[3])
    elev_boundary_left_bottom = (elevation.bounds[0], elevation.bounds[1])
    elev_boundary_right_bottom = (elevation.bounds[2], elevation.bounds[1])

    elevation_mbr = Polygon(
        [elev_boundary_right_top, elev_boundary_left_top, elev_boundary_left_bottom, elev_boundary_right_bottom])

    input_point_height = get_height_from_xy(input_point.x, input_point.y, height_data=elevation)
    point_input_with_height = PointWithHeight(input_point.x, input_point.y, input_point_height)

    # Prevent buffer from exceeding the elevation range
    mask_cropped_by_mbr = buffer.intersection(elevation_mbr)
    masked_elevation_raster, transformer = rasterio.mask.mask(dataset=elevation, shapes=[mask_cropped_by_mbr],
                                                              crop=True, nodata=0, filled=False)

    row_col_height_hash_table = define_hash()

    highest_value = np.max(masked_elevation_raster)
    highest_point_x, highest_point_y = get_xy_from_height(height_value=highest_value,
                                                          height_data=masked_elevation_raster,
                                                          transformer=transformer)

    highest_point = PointWithHeight(highest_point_x, highest_point_y, highest_value)

    return point_input_with_height, highest_point, row_col_height_hash_table
  
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
```

1. `Edge class`: 存放两点所组成的路径信息，来计算最终自身的weight
2. `weight`：weight_base + weight_height
   - weight_base：length / speed which is the basic time. (metre/per minute)
   - weight_height：cost of height ascending or descending = height_difference / 10  (metre/per minute)
3. `networkx.dijkstra_path(G, source, target, weight)`：
4. 测试时该过程较慢，因此在此加了一个进度条

   - Reason for slowness: it is required to excute `elevation.read(1)[row,col]` every time when calculate the height of each given ITN-point

   - Optimization: Storing all `(row,col)->height` information via a hash table when first reading all height data in TASK_2.
     - Cost of searching data in a hash table=O(1)

### Task 5: Map Plotting

```python
# plotter.py

import matplotlib.pyplot as plt
from rasterio import plot as raster_plot
import matplotlib.patches as mpatches
from shapely.geometry import Point
import geopandas as gpd
from constant import CRS_BNG

def distance(p1, p2, crs=CRS_BNG):
    pnt1 = Point(p1[0], p1[1])
    pnt2 = Point(p2[0], p2[1])
    points_df = gpd.GeoDataFrame({'geometry': [pnt1, pnt2]}, crs=crs)
    points_df2 = points_df.shift()

    distance_df = points_df.distance(points_df2)
    distance_df = distance_df.reset_index()
    distance_df.columns = ['index', 'distance']
    distance_df.index = ['d1', 'd2']

    return distance_df.loc['d2', 'distance']


class Plotter:
    def __init__(self, crs):
        self.__base_figure = plt.figure(figsize=(9, 6), dpi=100)
        self.__ax = self.__base_figure.add_subplot(111)
        self.__ax.set_axis_off()

        self.crs = crs

    def get_figure(self):
        return self.__base_figure, self.__ax

    def add_vector(self, vector, **kwargs):
        vector.to_crs(self.crs).plot(ax=self.__ax, **kwargs)

    def add_raster(self, raster, **kwargs):
        raster_plot.show(raster, ax=self.__ax, **kwargs)

    def add_north(self, label_size=10, loc_x=0.95, loc_y=0.95, width=0.05, height=0.1, pad=0.1):
        """
        Add a north arrow with 'N' text note
        :param label_size: size of 'N'
        :param loc_x: Horizontal proportion of the lower part of the text to ax
        :param loc_y: Vertical proportion of the lower part of the text to ax
        :param width: Proportion width of north arrow in ax
        :param height: Proportion height of north arrow in ax
        :param pad: Proportion clearance of 'N' in ax
        :return: None
        """
        ax = self.__ax
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ylen = maxy - miny
        xlen = maxx - minx
        left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
        right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
        top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
        center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
        triangle = mpatches.Polygon([left, top, right, center], color='k')
        ax.text(s='N',
                x=minx + xlen * loc_x,
                y=miny + ylen * (loc_y - pad + height),
                fontsize=label_size,
                horizontalalignment='center',
                verticalalignment='bottom')
        ax.add_patch(triangle)

    def add_scale_bar(self, lon0, lat0, length=2000, size=200):
        """
        lon0: longitude
        lat0: latitude
        length: length
        size: size
        """
        ax = self.__ax
        ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length, colors="black", ls="-", lw=1, label=f'{length} km')
        ax.vlines(x=lon0, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)
        ax.vlines(x=lon0 + length, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)

        scale_distance = distance([lon0, lat0], [lon0 + length, lat0])

        ax.text(lon0 + length / 2, lat0 + 2 * size, f'{int(scale_distance / 1000)} km', horizontalalignment='center')

    # TODO: add legend
    def add_legend(self):
        ax = self.__ax
        isle_of_wight = mpatches.Patch(color='white', label='isle of wight',alpha=.5)

        road, = ax.plot([], label="road", color='black')
        shortest_distance_path, = ax.plot([], label="shortest distance path", color='red', linewidth=3)
        shortest_time_path, = ax.plot([], label="shortest time path", color='blue', linewidth=3)

        ax.legend(handles=[isle_of_wight, road, shortest_distance_path, shortest_time_path], loc='lower left',
                  fontsize='small')

    def show(self):
        self.add_north()
        self.add_scale_bar(lon0=462500, lat0=77000)
        self.add_legend()

        plt.show()
          
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


def main():
    # TODO：use user input from task_1
    user_input_p, state = user_input()

    # input_p, highest_p, rc_height_hash_table = highest_point_identify(Point(450000, 85000))
    input_p, highest_p, rc_height_hash_table = highest_point_identify(user_input_p)

    input_nearest_itn = nearest_itn_node(input_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)

    short_distance_path, short_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                         rc_height_hash_table=rc_height_hash_table)

    plotter = plot_result()
    fig, ax = plotter.get_figure()

    plotter.add_vector(short_distance_path, linewidth=3, color='red', alpha=.7)
    plotter.add_vector(short_time_path, linewidth=3, color='blue', alpha=.6)

    ax.scatter([input_p.get_geometry().x, highest_p.get_geometry().x],
               [input_p.get_geometry().y, highest_p.get_geometry().y], s=15, color='black')

    ax.annotate('start', xy=(input_p.get_geometry().x, input_p.get_geometry().y), color='blue')
    ax.annotate('end', xy=(highest_p.get_geometry().x, highest_p.get_geometry().y), color='green')

    plotter.show()


```

### TODO

- [ ] Task6
- [x] Too slow
- [x] Plotter class
- [ ] Error class
- [x] Task 4 how to add weight







```python
# main.py
# requirement
import geopandas
import rasterio
import matplotlib.pyplot as plt
import shapely
import rtree
import networkx

# Material
# vector
isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')
# raster
background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')
```

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

from shapely.geometry import Point, Polygon


def generate_box(x_min, y_min, x_max, y_max):
    return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))

def user_input():
    x = float(input('please input a coordinates x: '))
    y = float(input('please input a coordinates y: '))

    box = generate_box(x_min=430000, x_max=465000, y_min=80000, y_max=95000)
    input_point = Point(x, y)

    return input_point, box.contains(input_point) or box.touches(input_point)
```

1. Generate the maximum area
2. Identify whether user_input_point is in the given area via `GeoDataframe.contains&GeoDataframe.touches`

### Task 2: Highest Point Identification

```python
# task_2.py
from shapely.geometry import Point, Polygon
import geopandas as gpd
import rasterio
from rasterio import mask
import numpy as np
import time

from geometry import PointWithHeight

elevation = rasterio.open('Material/elevation/SZ.asc')
isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')


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


def highest_point_identify(input_point, buffer_radius=5 * 1000):
    """
    :param input_point: shapely.geometry.Point,user input point form task_1
    :param buffer_radius: radius of buffer, default is 5km
    :return: (PointWithHeight,PointWithHeight,{}),
                first is the point user input with its height info,
                second is the highest point within given radius,
                third is a hash table storing (row,col)=>height
    """

    buffer = input_point.buffer(buffer_radius)

    elev_boundary_right_top = (elevation.bounds[2], elevation.bounds[3])
    elev_boundary_left_top = (elevation.bounds[0], elevation.bounds[3])
    elev_boundary_left_bottom = (elevation.bounds[0], elevation.bounds[1])
    elev_boundary_right_bottom = (elevation.bounds[2], elevation.bounds[1])

    elevation_mbr = Polygon(
        [elev_boundary_right_top, elev_boundary_left_top, elev_boundary_left_bottom, elev_boundary_right_bottom])

    input_point_height = get_height_from_xy(input_point.x, input_point.y, height_data=elevation)
    point_input_with_height = PointWithHeight(input_point.x, input_point.y, input_point_height)

    # Prevent buffer from exceeding the elevation range
    mask_cropped_by_mbr = buffer.intersection(elevation_mbr)
    masked_elevation_raster, transformer = rasterio.mask.mask(dataset=elevation, shapes=[mask_cropped_by_mbr],
                                                              crop=True, nodata=0, filled=False)

    row_col_height_hash_table = define_hash()

    highest_value = np.max(masked_elevation_raster)
    highest_point_x, highest_point_y = get_xy_from_height(height_value=highest_value,
                                                          height_data=masked_elevation_raster,
                                                          transformer=transformer)

    highest_point = PointWithHeight(highest_point_x, highest_point_y, highest_value)

    return point_input_with_height, highest_point, row_col_height_hash_table
  
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
```

1. `Edge class`: 存放两点所组成的路径信息，来计算最终自身的weight
2. `weight`：weight_base + weight_height
   - weight_base：length / speed which is the basic time. (metre/per minute)
   - weight_height：cost of height ascending or descending = height_difference / 10  (metre/per minute)
3. `networkx.dijkstra_path(G, source, target, weight)`：
4. 测试时该过程较慢，因此在此加了一个进度条

   - Reason for slowness: it is required to excute `elevation.read(1)[row,col]` every time when calculate the height of each given ITN-point

   - Optimization: Storing all `(row,col)->height` information via a hash table when first reading all height data in TASK_2.
     - Cost of searching data in a hash table=O(1)

### Task 5: Map Plotting

```python
# plotter.py

import matplotlib.pyplot as plt
from rasterio import plot as raster_plot
import matplotlib.patches as mpatches
from shapely.geometry import Point
import geopandas as gpd
from constant import CRS_BNG

def distance(p1, p2, crs=CRS_BNG):
    pnt1 = Point(p1[0], p1[1])
    pnt2 = Point(p2[0], p2[1])
    points_df = gpd.GeoDataFrame({'geometry': [pnt1, pnt2]}, crs=crs)
    points_df2 = points_df.shift()

    distance_df = points_df.distance(points_df2)
    distance_df = distance_df.reset_index()
    distance_df.columns = ['index', 'distance']
    distance_df.index = ['d1', 'd2']

    return distance_df.loc['d2', 'distance']


class Plotter:
    def __init__(self, crs):
        self.__base_figure = plt.figure(figsize=(9, 6), dpi=100)
        self.__ax = self.__base_figure.add_subplot(111)
        self.__ax.set_axis_off()

        self.crs = crs

    def get_figure(self):
        return self.__base_figure, self.__ax

    def add_vector(self, vector, **kwargs):
        vector.to_crs(self.crs).plot(ax=self.__ax, **kwargs)

    def add_raster(self, raster, **kwargs):
        raster_plot.show(raster, ax=self.__ax, **kwargs)

    def add_north(self, label_size=10, loc_x=0.95, loc_y=0.95, width=0.05, height=0.1, pad=0.1):
        """
        Add a north arrow with 'N' text note
        :param label_size: size of 'N'
        :param loc_x: Horizontal proportion of the lower part of the text to ax
        :param loc_y: Vertical proportion of the lower part of the text to ax
        :param width: Proportion width of north arrow in ax
        :param height: Proportion height of north arrow in ax
        :param pad: Proportion clearance of 'N' in ax
        :return: None
        """
        ax = self.__ax
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ylen = maxy - miny
        xlen = maxx - minx
        left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
        right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
        top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
        center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
        triangle = mpatches.Polygon([left, top, right, center], color='k')
        ax.text(s='N',
                x=minx + xlen * loc_x,
                y=miny + ylen * (loc_y - pad + height),
                fontsize=label_size,
                horizontalalignment='center',
                verticalalignment='bottom')
        ax.add_patch(triangle)

    def add_scale_bar(self, lon0, lat0, length=2000, size=200):
        """
        lon0: longitude
        lat0: latitude
        length: length
        size: size
        """
        ax = self.__ax
        ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length, colors="black", ls="-", lw=1, label=f'{length} km')
        ax.vlines(x=lon0, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)
        ax.vlines(x=lon0 + length, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)

        scale_distance = distance([lon0, lat0], [lon0 + length, lat0])

        ax.text(lon0 + length / 2, lat0 + 2 * size, f'{int(scale_distance / 1000)} km', horizontalalignment='center')

    # TODO: add legend
    def add_legend(self):
        ax = self.__ax
        isle_of_wight = mpatches.Patch(color='white', label='isle of wight',alpha=.5)

        road, = ax.plot([], label="road", color='black')
        shortest_distance_path, = ax.plot([], label="shortest distance path", color='red', linewidth=3)
        shortest_time_path, = ax.plot([], label="shortest time path", color='blue', linewidth=3)

        ax.legend(handles=[isle_of_wight, road, shortest_distance_path, shortest_time_path], loc='lower left',
                  fontsize='small')

    def show(self):
        self.add_north()
        self.add_scale_bar(lon0=462500, lat0=77000)
        self.add_legend()

        plt.show()
          
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


def main():
    # TODO：use user input from task_1
    user_input_p, state = user_input()

    # input_p, highest_p, rc_height_hash_table = highest_point_identify(Point(450000, 85000))
    input_p, highest_p, rc_height_hash_table = highest_point_identify(user_input_p)

    input_nearest_itn = nearest_itn_node(input_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)

    short_distance_path, short_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                         rc_height_hash_table=rc_height_hash_table)

    plotter = plot_result()
    fig, ax = plotter.get_figure()

    plotter.add_vector(short_distance_path, linewidth=3, color='red', alpha=.7)
    plotter.add_vector(short_time_path, linewidth=3, color='blue', alpha=.6)

    ax.scatter([input_p.get_geometry().x, highest_p.get_geometry().x],
               [input_p.get_geometry().y, highest_p.get_geometry().y], s=15, color='black')

    ax.annotate('start', xy=(input_p.get_geometry().x, input_p.get_geometry().y), color='blue')
    ax.annotate('end', xy=(highest_p.get_geometry().x, highest_p.get_geometry().y), color='green')

    plotter.show()


```

### TODO

- [ ] Task6
- [x] Too slow
- [x] Plotter class
- [ ] Error class
- [x] Task 4 how to add weight





