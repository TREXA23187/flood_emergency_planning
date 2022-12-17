from shapely.geometry import Point, Polygon
import rasterio
from rasterio import mask
import numpy as np

from geometry import PointWithHeight

elevation = rasterio.open('Material/elevation/SZ.asc')


def get_xy_from_height(height_value, height_data, transformer=None):
    # TODO: if height_data is whole elevation program might have error -> To import an Error class

    highest_point_xy = np.where(height_data == height_value)
    x = highest_point_xy[1][0]
    y = highest_point_xy[2][0]

    if transformer is not None:
        transformed_x, transformed_y = rasterio.transform.xy(transformer, x, y)

        return transformed_x, transformed_y

    return x, y


# TODO: too slow since get height from whole file  -> Reduce search scope
def get_height_from_xy(x, y, height_data=elevation):
    (row, col) = height_data.index(x, y)
    point_height = height_data.read(1)[row, col]

    return point_height


def highest_point_identify(input_point, buffer_radius=5):
    """
    :param input_point: task_1中return的用户输入的值，类型为shapely.geometry.Point
    :param buffer_radius: 缓冲区范围，根据任务书默认为5km范围，防止后续可能更变因此以参数形式输入
    :return: 类型为PointWithHeight, 第一个PointWithHeight为参数输入的input_point携带height后返回 \
            （因为在task_1中的Point还未携带height信息），第二个返回buffer范围内最高点信息
    """

    # 以point为中心，建立buffer范围的缓冲区，默认范围为5km
    buffer = input_point.buffer(buffer_radius * 1000)

    elev_boundary_right_top = (elevation.bounds[2], elevation.bounds[3])
    elev_boundary_left_top = (elevation.bounds[0], elevation.bounds[3])
    elev_boundary_left_bottom = (elevation.bounds[0], elevation.bounds[1])
    elev_boundary_right_bottom = (elevation.bounds[2], elevation.bounds[1])

    elevation_mbr = Polygon(
        [elev_boundary_right_top, elev_boundary_left_top, elev_boundary_left_bottom, elev_boundary_right_bottom])

    input_point_height = get_height_from_xy(input_point.x, input_point.y, height_data=elevation)
    point_input_with_height = PointWithHeight(input_point.x, input_point.y, input_point_height)

    # 防止buffer超出elevation范围
    mask_cropped_by_mbr = buffer.intersection(elevation_mbr)
    masked_elevation_raster, transformer = rasterio.mask.mask(dataset=elevation, shapes=[mask_cropped_by_mbr],
                                                              crop=True, nodata=0, filled=False)
    # 利用'Material/elevation/SZ.asc'文件获取该范围内最高点
    highest_value = np.max(masked_elevation_raster)
    highest_point_x, highest_point_y = get_xy_from_height(height_value=highest_value,
                                                          height_data=masked_elevation_raster,
                                                          transformer=transformer)

    highest_point = PointWithHeight(highest_point_x, highest_point_y, highest_value)

    return point_input_with_height, highest_point


if __name__ == '__main__':
    input_p, highest_p = highest_point_identify(Point(450000, 85000))
    print(highest_p.get_height())
    print(highest_p.get_geometry())
