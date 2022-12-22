from shapely.geometry import Point
from constant import speed


class PlainPoint:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y
        self.__geometry = Point(x, y)

    def get_geometry(self):
        return self.__geometry


class PointWithHeight(PlainPoint):
    def __init__(self, x, y, height):
        super().__init__(x, y)
        self.__height = height

    def get_height(self):
        return self.__height

    def set_height(self, new_height):
        self.__height = new_height


class ItnPoint(PointWithHeight):
    def __init__(self, x, y, height, fid):
        super().__init__(x, y, height)
        self.__fid = fid

    def get_fid(self):
        return self.__fid


class PlainLine:
    def __init__(self, p1, p2):
        self.__p1 = p1
        self.__p2 = p2
        self.__geometry = [p1, p2]

    def get_geometry(self):
        return self.__geometry


class DirectedLine(PlainLine):
    def __init__(self, start_node, end_node):
        super().__init__(start_node, end_node)
        self.__start_node = self.get_geometry()[0]
        self.__end_node = self.get_geometry()[1]

    def get_start_node(self):
        return self.__start_node

    def get_end_node(self):
        return self.__end_node


class Edge(DirectedLine):
    def __init__(self, fid, start_node, end_node, length):
        super().__init__(start_node, end_node)
        self.__fid = fid
        self.__length = length
        self.__height_diff = end_node.get_height() - start_node.get_height()

    def get_fid(self):
        return self.__fid

    def get_length(self):
        return self.__length

    def get_height_diff(self):
        return self.__height_diff

    def add_weight(self):
        base_weight = self.__length / speed

        # if height_diff is positive, it's ascending -> base_weight+height_weight
        # if height_diff is negative, it's descending -> base_weight-abs(height_weight)
        height_weight = (self.__height_diff / 10)

        if self.__height_diff == 0:
            return base_weight
        else:
            return base_weight + height_weight
