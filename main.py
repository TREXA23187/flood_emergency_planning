import matplotlib.pyplot as plt
from pyproj import CRS
from shapely.geometry import Point
from task_2 import highest_point_identify
from task_3 import nearest_itn_node
from task_4 import shortest_path
from task_5 import plot_sample

CRS_BNG = CRS('epsg:27700')


def main():
    # TODO：use user input from task_1
    input_p, highest_p = highest_point_identify(Point(450000, 85000))

    input_nearest_itn = nearest_itn_node(input_p.get_geometry())
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry())

    short_distance_path_data, short_time_path_data = shortest_path(input_nearest_itn, highest_nearest_itn)

    ax = plot_sample()
    short_distance_path_data.plot(ax=ax, linewidth=3, color='red', alpha=.7)
    short_time_path_data.plot(ax=ax, linewidth=3, color='blue', alpha=.6)
    ax.scatter([input_p.get_geometry().x, highest_p.get_geometry().x],
               [input_p.get_geometry().y, highest_p.get_geometry().y], s=15, color='black')

    plt.annotate('start', xy=(input_p.get_geometry().x, input_p.get_geometry().y), color='blue')
    plt.annotate('end', xy=(highest_p.get_geometry().x, highest_p.get_geometry().y), color='green')


if __name__ == '__main__':
    main()

    plt.show()
