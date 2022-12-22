from task_1 import user_input
from task_2 import highest_point_identify
from task_3 import nearest_itn_node
from task_4 import shortest_path
from task_5 import plot_result


def main():
    # TODO：use user input from task_1
    user_input_p, state = user_input()

    # input_p, highest_p, rc_height_hash_table = highest_point_identify(Point(450000, 85000))
    input_p, highest_p, rc_height_hash_table = highest_point_identify(user_input_p)

    input_nearest_itn = nearest_itn_node(input_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)
    highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)

    short_distance_path, short_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                         rc_height_hash_table=rc_height_hash_table)

    plotter = plot_result()
    fig, ax = plotter.get_figure()

    plotter.add_vector(short_distance_path, linewidth=3, color='red', alpha=.7)
    plotter.add_vector(short_time_path, linewidth=3, color='blue', alpha=.6)

    ax.scatter([input_p.get_geometry().x, highest_p.get_geometry().x],
               [input_p.get_geometry().y, highest_p.get_geometry().y], s=15, color='black')

    ax.annotate('start', xy=(input_p.get_geometry().x, input_p.get_geometry().y), color='blue')
    ax.annotate('end', xy=(highest_p.get_geometry().x, highest_p.get_geometry().y), color='green')

    plotter.show()


if __name__ == '__main__':
    main()
