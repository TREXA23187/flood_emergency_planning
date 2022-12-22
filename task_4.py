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
    print('>>>>>>>>>>>>> Progressing shortest_path in TASK_4')

    with open('Material/itn/solent_itn.json') as file:
        itn_json = json.load(file)

    itn_road_nodes = itn_json['roadnodes']
    itn_road_links = itn_json['roadlinks']

    edge_list = []

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


if __name__ == '__main__':
    input_p, highest_p, row_col_height_hash_table = highest_point_identify(Point(450000, 85000))

    input_nearest_itn = nearest_itn_node(input_p.get_geometry(), row_col_height_hash_table)
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), row_col_height_hash_table)

    shortest_distance_path, shortest_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                               row_col_height_hash_table)

    print(shortest_distance_path)
