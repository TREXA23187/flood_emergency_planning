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

    # itn_node_index -> 1,2,3...10457
    # itn_node_info  -> ('osgb5000005230555073', {'coords': [449795.562, 95242.886]})
    # itn_node_info[0] -> 'osgb5000005230555073'
    # itn_node_info[1] -> {'coords': [449795.562, 95242.886]}
    for itn_node_index, itn_node_info in enumerate(itn_road_nodes.items()):
        node_coordinates = (itn_node_info[1]['coords'][0], itn_node_info[1]['coords'][1])
        itn_rtree_index.insert(id=itn_node_index, coordinates=node_coordinates, obj=itn_node_info[0])

    # find nearest itn to input_node via rtree
    nearest_itn_to_input = None

    # fid = itn_rtree_index.nearest(coordinates=(input_point.x, input_point.y), num_results=1, objects='raw').__next__()
    for fid in itn_rtree_index.nearest(coordinates=(input_point.x, input_point.y), num_results=1, objects='raw'):
        coordinates = itn_road_nodes[fid]['coords']
        x, y = coordinates

        nearest_itn_to_input = ItnPoint(x, y, get_height_from_xy(x, y, hash_table=rc_height_hash_table), fid)

    return nearest_itn_to_input


if __name__ == '__main__':
    input_p, highest_p, row_col_height_hash_table = highest_point_identify(Point(450000, 85000))
    nearest_input_itn = nearest_itn_node(input_p.get_geometry(), row_col_height_hash_table)
    nearest_highest_itn = nearest_itn_node(highest_p.get_geometry(), row_col_height_hash_table)

    print(nearest_highest_itn.get_height())
