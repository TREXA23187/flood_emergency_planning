## Flood Emergency Planning	

```python
# main.py
# 可能需要导入的库
import geopandas
import rasterio
import matplotlib.pyplot as plt
import shapely
import rtree
import networkx

# 各项数据读取
# vector
isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')
# raster
background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')
```

### geometry

```python
# geometry.py

class PointWithHeight:
    def __init__(self, x, y, height):
        self.__x = x
        self.__y = y
        self.__geometry = Point(x, y)
        self.__height = height

class ItnPoint(PointWithHeight):
    def __init__(self, x, y, height, fid):
        super().__init__(x, y, height)
        self.__fid = fid

    def get_fid(self):
        return self.__fid
```

### Task 1: User Input

```python
# task_1.py

from shapely.geometry import Point

def user_input():
    """
    :return: Point(x,y) 类型为shapely.geometry.Point
    :return: state 类型为String
    """
    x = float(input('please input a coordinates x: '))
    y = float(input('please input a coordinates y: '))
    
    # 判断点Point(x,y)是否在boundary内，结合Task6
    # if in boundary -> state = 1
    # elif not in boundary but in island -> state = 2
    # else not in ilse -> state = 3 同时结束程序quit()
  
  return Point(x,y), state
```

1. `geom1.contains(geom2)`: 判断geom1是否包含geom2
2. `geom1.touches`:判断geom1和geom2两者是否有边缘接触

### Task 2: Highest Point Identification

```python
# task_2.py

from shapely.geometry import Point  

def highest_point_identify(input_point, buffer = 5):
    """
    :param point: task_1中return的用户输入的值，类型为shapely.geometry.Point
    :param buffer: 缓冲区范围，根据任务书默认为5km范围，防止后续可能更变因此以参数形式输入
    :return: 类型为PointWithHeight(见geometry.py部分), 第一个PointWithHeight为参数输入的input_point携带height后返回（因为在task_1中的Point还未携带height信息），第二个返回buffer范围内最高点信息
    """
  
  # 以point为中心，建立buffer范围的缓冲区，默认范围为5km
  # 利用'Material/elevation/SZ.asc'文件获取该范围内最高点
  
  return PointWithHeight(x_input,y_input,height_input), \
								PointWithHeight(x_highest,yhighest,height_highest)
  
```

1. `numpy.max`：返回数组中最大值，用于elevation中即可获取到最大值
2. `rasterio.mask.mask`：通过以中心点建立5km缓冲区域作为mask，搜寻最大值所在位置即为最高点位置。

### Task 3: Nearest Integrated Transport Network

```python
# task_3.py

import json
from rtree import index

def nearest_itn_node(input_node):
    """
    :param input_node: Point(x,y) 类型为shapely.geometry.Point
    :return ItnPoint(继承自PointWithHeight)，即距离input_node最近的itn点，且携带高程信息与fid信息
    """
  
  # 'Materialitn/solent_itn.json'为itn点信息
    with open('Material/itn/solent_itn.json') as file:
        itn_json = json.load(file)
        
    itn_road_nodes = itn_json['roadnodes']
    itn_road_links = itn_json['roadlinks']
    
  # 利用rtree找到距离input_node距离最近的itn
    for itn_node_index, itn_node_info in enumerate(itn_road_nodes.items()):
				# ...
        itn_rtree_index.insert(id=, coordinates=)
        # ...
    
  # ...
  nearest_itn_to_input = itn_rtree_index.nearest(coordinates=,num_results=)
  
  
	return nearest_itn_to_input

```

1. `index.Index.nearest(coordinates, num_results)`：
   - coordinates：用户输入点信息，即所需要获取距离其最近的点
   - num_results：返回结果数可设置为1

### Task 4: Shortest Path

```python
# task_4.py

from shapely.geometry import LineString
import networkx as nx

class Edge:
    def __init__(self, index, start_node, end_node, length):
        self.index = index
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.height_diff = end_node.get_height() - start_node.get_height()
        
    # 利用两点间的高程差以及距离进行权重的设置
    def add_weight(self):
        base_weight = self.length / walking_speed
        slope = self.height_diff / self.length * 100

        # print(abs(self.height_diff / 10))
        ascent_weight = base_weight + abs(self.height_diff / 10)

        # there is no rule for descent scene
        descent_weight = ascent_weight

        return base_weight, ascent_weight, descent_weight

shortest_path(node_begin, node_end):
    """
    :param node_begin: 类型为PointWithHeight,查询最短路径的起点
    :param node_end: 类型为PointWithHeight,查询最短路径的终点
    :return 返回距离最短和时间最短路径，由shapely.geometry.LineString构成的GeoDataframe（方便task_5绘图）
    """
    
    # 利用两点间的高程差以及距离进行权重的设置，类似于：
    # nodes[n]和nodes[n+1]连接的路线的最终权重为weight = 						 
    #                                 weight_distance+weight_height
    
    # 计算出每两点间的权重，最后可以通过networkx计算最短路径
    graph = nx.DiGraph()
    # ...
    graph.add_edge(...)
    short_path = nx.dijkstra_path(G=graph, source=, target=, weight='length')
    #... 
    
    # result
    short_distance_path_df = gpd.GeoDataFrame({'fid':..., 'geometry': [LineString(),LineString(),...]})
    short_time_path_df =  gpd.GeoDataFrame({'fid':..., 'geometry': [LineString(),LineString(),...]})
  
	return short_distance_path_df, short_time_path_df
```

1. `Edge class`: 存放两点所组成的路径信息，来计算最终自身的weight

2. `weight`：weight_base + weight_height

   - weight_base：路程/速度
   - weight_height：爬高/爬低导致的weight的增减

3. `networkx.dijkstra_path(G, source, target, weight)`：

4. 测试时该过程较慢，因此在此加了一个进度条

   - 较慢原因：获取每一给itn点的高程需要每一次都从elevation高程数据搜索一次

   - 优化点：缩小搜索范围
   - 目前在代码中暂时先写入一个常数值作为每个点的高程（无高程差）进行测试

### Task 5: Map Plotting

```python
# task_5.py

# 定义Plotter类辅助画图
class Plotter:
        def __init__(self):
      			...
        		...
          
```

### Task 6: Extend the Region

```python
# task_6.py
# 感觉 可以结合task_1.py
```

### main

```python
# main.py
		
  # 1
  user_input_point = user_input()
  # 2
  input_p, highest_p = highest_point_identify(user_input_point)
	# 3
  input_nearest_itn = nearest_itn_node(input_p.get_geometry())
  highest_nearest_itn = nearest_itn_node(highest_p.get_geometry())
  # 4
  short_distance_path_data, short_time_path_data = shortest_path(input_nearest_itn, highest_nearest_itn)
  # 5
  Plotter.plot()


```

### TODO

- [ ] Task6
- [ ] Too slow
- [ ] Plotter class
- [ ] Error class
- [ ] Task 4 how to add weight





