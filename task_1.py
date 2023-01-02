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
                return input_point

            else:
                print(f'>>>>>>>>>>>>> Input Point({input_point.x},{input_point.y}) is inside given isle area in TASK_1')
                return input_point
        else:
            raise Exception(f'Input Point({input_point.x},{input_point.y}) is outside given area')
    except Exception as error:
        raise Exception(f'{error} ==> TASK_1')


if __name__ == '__main__':
    user_loc_test = user_input()
    print(user_loc_test)
