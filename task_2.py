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
    # TODO: if height_data is whole elevation program might have error -> To import an Error class
    highest_point_xy = np.where(height_data == height_value)
    row = highest_point_xy[1][0]
    col = highest_point_xy[2][0]

    if transformer is not None:
        transformed_x, transformed_y = rasterio.transform.xy(transformer, row, col)

        return transformed_x, transformed_y

    return height_data.xy(row, col)


# TODO: too slow since get height from whole file  -> Reduce search scope
# nearly 2s for each operation
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


if __name__ == '__main__':
    input_p, highest_p, rc_height_hash_table = highest_point_identify(Point(450000, 85000))
    # print(xy_hash_table[(2500, 4000)])
    # print(input_p.get_height())
    # print(highest_p.get_geometry())
