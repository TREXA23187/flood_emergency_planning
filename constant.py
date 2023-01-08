from pyproj import CRS

map_x_min = 430000
map_x_max = 465000
map_y_min = 75000
map_y_max = 97500

CRS_BNG = CRS('epsg:27700')

buffer_radius = 5 * 1000
speed = buffer_radius / 1 * 60
map_view_range = 20 * 1000

plot_move_speed = 6
