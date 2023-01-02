from task_1 import user_input
from task_2 import highest_point_identify
from task_3 import nearest_itn_node
from task_4 import shortest_path
from task_5 import plot_result
from constant import buffer_radius
from exception_handler import ExceptionHandler


def main():
    try:
        # task 1
        # 435000, 87000
        user_input_p = user_input()

        # task 2
        input_p, highest_p, rc_height_hash_table = highest_point_identify(user_input_p, buffer_radius=buffer_radius)

        # task 3
        input_nearest_itn = nearest_itn_node(input_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)
        highest_nearest_itn = nearest_itn_node(highest_p.get_geometry(), rc_height_hash_table=rc_height_hash_table)

        # task 4
        short_distance_path, short_time_path = shortest_path(input_nearest_itn, highest_nearest_itn,
                                                             rc_height_hash_table=rc_height_hash_table)

        # task 5
        plotter = plot_result(start_point=input_p, end_point=highest_p)

        plotter.add_vector(short_distance_path, linewidth=3, color='blue', alpha=.7)
        plotter.add_vector(short_time_path, linewidth=3, color='red', alpha=.6)

        plotter.show()
    except Exception as error:
        exc = ExceptionHandler(error)
        exc.log()


if __name__ == '__main__':
    main()
