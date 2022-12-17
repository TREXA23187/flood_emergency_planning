from shapely.geometry import Point


# 封装可携带高程信息的Point类型，可以扩展一下
class PointWithHeight:
    def __init__(self, x, y, height):
        self.__x = x
        self.__y = y
        self.__geometry = Point(x, y)
        self.__height = height

    def get_geometry(self):
        return self.__geometry

    def get_height(self):
        return self.__height

    def set_x(self, new_x):
        self.__x = new_x

    def set_y(self, new_y):
        self.__y = new_y

    def set_height(self, new_height):
        self.__height = new_height


class ItnPoint(PointWithHeight):
    def __init__(self, x, y, height, fid):
        super().__init__(x, y, height)
        self.__fid = fid

    def get_fid(self):
        return self.__fid
