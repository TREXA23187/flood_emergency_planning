from shapely.geometry import Point, Polygon


def generate_box(x_min, y_min, x_max, y_max):
    return Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)))


# TODO: add task6 and finish like in README.md
def user_input():
    x = float(input('please input a coordinates x: '))
    y = float(input('please input a coordinates y: '))

    box = generate_box(x_min=430000, x_max=465000, y_min=80000, y_max=95000)
    input_point = Point(x, y)

    return input_point, box.contains(input_point) or box.touches(input_point)


if __name__ == '__main__':
    user_loc_test = user_input()
    print(user_loc_test)
