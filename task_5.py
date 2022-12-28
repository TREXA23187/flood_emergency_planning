import geopandas as gpd
import rasterio
from plotter import Plotter
from constant import CRS_BNG, buffer_radius, map_view_range, map_x_min, map_x_max, map_y_min, map_y_max
from geometry import PlainPoint
import matplotlib.patches as mpatches

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')

background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')


def plot_result(start_point, end_point):
    try:
        plotter = Plotter(CRS_BNG)
        fig, ax = plotter.get_figure()

        # plotter.add_vector(isle_of_wight, color='white', alpha=.6)
        # plotter.add_vector(road_nodes, markersize=2, color='grey', alpha=.3)
        # plotter.add_vector(road_links, linewidth=1, color='black', alpha=.5)

        buffer_circle = mpatches.Circle((start_point.get_geometry().x, start_point.get_geometry().y), buffer_radius,
                                        alpha=.3)
        plotter.add_artist(buffer_circle)

        start_x = start_point.get_geometry().x
        start_y = start_point.get_geometry().y

        # set range 20km*20km
        x_min = max(map_x_min, start_x - map_view_range / 2)
        x_max = min(map_x_max, start_x + map_view_range / 2)
        y_min = max(map_y_min, start_y - map_view_range / 2)
        y_max = min(map_y_max, start_y + map_view_range / 2)

        # reset map view to 20*20 km2 if out of range
        if x_max - x_min != map_view_range:
            if x_min == map_x_min:
                x_max += map_x_min - (start_x - map_view_range / 2)
            if x_max == map_x_max:
                x_min -= (start_x + map_view_range / 2) - map_x_max
        if y_max - y_min != map_view_range:
            if y_min == map_y_min:
                y_max += map_y_min - (start_y - map_view_range / 2)
            if y_max == map_y_max:
                y_min -= (start_y + map_view_range / 2) - map_y_max
        plotter.set_xy_lim(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        # add background of isle of wight
        plotter.add_raster(background, cmap='terrain')

        # mark start and end points on map
        plotter.add_point(start_point.get_geometry().x, start_point.get_geometry().y, color='red', markersize=4)
        plotter.add_point(end_point.get_geometry().x, end_point.get_geometry().y, color='green', markersize=4)

        # add legend to plotter
        plotter.add_legend(legend_type='point', legend_label='start point', color='red', markersize=4)
        plotter.add_legend(legend_type='point', legend_label='end point', color='green', markersize=4)
        plotter.add_legend(legend_type='line', legend_label='road', color='brown')
        plotter.add_legend(legend_type='line', legend_label='shortest distance path', color='blue', linewidth=3)
        plotter.add_legend(legend_type='line', legend_label='shortest time path', color='red', linewidth=3)
        plotter.add_legend(legend_type='line', legend_label='if paths coincide', color='purple', linewidth=3)

        return plotter
    except Exception as error:
        raise Exception(f'{error} ==> TASK_5')


if __name__ == '__main__':
    p = plot_result(PlainPoint(440000, 92000), PlainPoint(450000, 85000))
    p.show()
