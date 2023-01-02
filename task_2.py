from shapely.geometry import Point, Polygon
import rasterio
from rasterio import mask
import numpy as np
import time
import geopandas as gpd
from constant import CRS_BNG

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

        # Prevent buffer from exceeding the elevation range
        mask_cropped_by_isle = buffer.intersection(isle_of_wight.geometry[0])
        masked_elevation_raster, transformer = rasterio.mask.mask(dataset=elevation, shapes=[mask_cropped_by_isle],
                                                                  crop=True, nodata=0, filled=False)

        row_col_height_hash_table = define_hash()

        input_point_height = get_height_from_xy(input_point.x, input_point.y, hash_table=row_col_height_hash_table)
        point_input_with_height = PointWithHeight(input_point.x, input_point.y, input_point_height)

        highest_value = np.max(masked_elevation_raster)
        highest_point_x, highest_point_y = get_xy_from_height(height_value=highest_value,
                                                              height_data=masked_elevation_raster,
                                                              transformer=transformer)

        highest_point = PointWithHeight(highest_point_x, highest_point_y, highest_value)

        return point_input_with_height, highest_point, row_col_height_hash_table
    except Exception as error:
        raise Exception(f'{error} ==> TASK_2')


if __name__ == '__main__':
    input_p, highest_p, rc_height_hash_table = highest_point_identify(Point(450000, 85000), 5000)
    # # print(xy_hash_table[(2500, 4000)])
    print(input_p.get_height())
    print(highest_p.get_geometry())
