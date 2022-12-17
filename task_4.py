import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
from geometry import ItnPoint
import json
from task_2 import highest_point_identify, get_height_from_xy
from task_3 import nearest_itn_node

import sys
import time

radius = 5000
walking_speed = radius / 60


class Edge:
    def __init__(self, index, start_node, end_node, length):
        self.index = index
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.height_diff = end_node.get_height() - start_node.get_height()

    # TODO: not finished
    # 利用两点间的高程差以及距离进行权重的设置
    def add_weight(self):
        base_weight = self.length / walking_speed
        slope = self.height_diff / self.length * 100

        # print(abs(self.height_diff / 10))
        ascent_weight = base_weight + abs(self.height_diff / 10)

        # there is no rule for descent scene
        descent_weight = ascent_weight

        return base_weight, ascent_weight, descent_weight


def shortest_path(point_start, point_end):
    """
    :param point_start: ItnPoint
    :param point_end: ItnPoint
    :return 返回距离最短和时间最短路径，由shapely.geometry.LineString构成的GeoDataframe
    """
    with open('Material/itn/solent_itn.json') as file:
        itn_json = json.load(file)

    itn_road_nodes = itn_json['roadnodes']
    itn_road_links = itn_json['roadlinks']

    edge_list = []

    total = len(itn_road_links.items())
    progress = 0
    start = time.perf_counter()
    for itn_link_index, itn_link_info in itn_road_links.items():
        # itn_link_index -> 'osgb4000000026240481'
        # itn_link_info  -> {'length': 330.77,
        #                  'coords': [[], [], []], 'start': 'osgb4000000026240451', 'end': 'osgb5000005189928990',
        #                  'natureOfRoad': 'Single Carriageway', 'descriptiveTerm': 'Private Road - Restricted Access'}
        # itn_link_info.keys() -> ['length', 'coords', 'start', 'end', 'natureOfRoad', 'descriptiveTerm']

        start_x = itn_road_nodes[itn_link_info['start']]['coords'][0]
        start_y = itn_road_nodes[itn_link_info['start']]['coords'][1]
        end_x = itn_road_nodes[itn_link_info['end']]['coords'][0]
        end_y = itn_road_nodes[itn_link_info['end']]['coords'][1]
        # start_h = get_height_from_xy(start_x, start_y)
        # end_h = get_height_from_xy(end_x, end_y)
        # start_node = ItnPoint(start_x, start_y, start_h, itn_link_info['start'])
        # end_node = ItnPoint(end_x, end_y, end_h, itn_link_info['end'])

        # for testing
        start_node = ItnPoint(start_x, start_y, 23, itn_link_info['start'])
        end_node = ItnPoint(end_x, end_y, 34, itn_link_info['end'])

        # progress bar
        if round(progress / total) <= 100 and round(progress / total * 100) % 2 == 0:
            spent_time = time.perf_counter() - start
            print("\r", end="")
            print(
                f'TASK 4 IN PROGRESS: {round(progress / total * 100)}% {"▓" * round(progress / total * 25)} \
                {spent_time.__format__(".2f")}s', end="")
            sys.stdout.flush()
            # time.sleep(0.05)

        progress += 1

        edge_list.append(
            Edge(index=itn_link_index, start_node=start_node, end_node=end_node, length=itn_link_info['length']))

    # 计算出每两点间的权重，最后可以通过networkx计算最短路径
    graph = nx.DiGraph()
    for edge in edge_list:
        base, ascent_weight, descent_weight = edge.add_weight()

        if edge.height_diff > 0:
            graph.add_edge(edge.start_node.get_fid(), edge.end_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=ascent_weight)
            graph.add_edge(edge.end_node.get_fid(), edge.start_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=descent_weight)
        elif edge.height_diff == 0:
            graph.add_edge(edge.start_node.get_fid(), edge.end_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=base)
            graph.add_edge(edge.end_node.get_fid(), edge.start_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=base)
        else:
            graph.add_edge(edge.start_node.get_fid(), edge.end_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=descent_weight)
            graph.add_edge(edge.end_node.get_fid(), edge.start_node.get_fid(), fid=edge.index, length=edge.length,
                           weight=ascent_weight)

    short_distance_path = nx.dijkstra_path(G=graph, source=point_start.get_fid(), target=point_end.get_fid(),
                                           weight='length')
    short_time_path = nx.dijkstra_path(G=graph, source=point_start.get_fid(), target=point_end.get_fid(), weight='time')

    short_distance_path_fids = []
    short_distance_path_geometry = []
    for i in range(len(short_distance_path) - 1):
        pre_node = short_distance_path[i]
        next_node = short_distance_path[i + 1]
        edge_fid = graph.edges[pre_node, next_node]['fid']
        short_distance_path_fids.append(edge_fid)
        road = LineString(itn_road_links[edge_fid]['coords'])
        short_distance_path_geometry.append(road)

    short_time_path_fids = []
    short_time_path_geometry = []
    for i in range(len(short_time_path) - 1):
        pre_node = short_time_path[i]
        next_node = short_time_path[i + 1]
        edge_fid = graph.edges[pre_node, next_node]['fid']
        short_time_path_fids.append(edge_fid)
        road = LineString(itn_road_links[edge_fid]['coords'])
        short_time_path_geometry.append(road)

    short_distance_path_df = gpd.GeoDataFrame(
        {'fid': short_distance_path_fids, 'geometry': short_distance_path_geometry})
    short_time_path_df = gpd.GeoDataFrame({'fid': short_time_path_fids, 'geometry': short_time_path_geometry})

    return short_distance_path_df, short_time_path_df


if __name__ == '__main__':
    input_p, highest_p = highest_point_identify(Point(450000, 85000))

    input_nearest_itn = nearest_itn_node(input_p.get_geometry())
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry())

    shortest_path_data = shortest_path(input_nearest_itn, highest_nearest_itn)

    # print(shortest_path_data)
