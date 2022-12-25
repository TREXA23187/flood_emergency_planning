import geopandas as gpd
import rasterio
from plotter import Plotter
from constant import CRS_BNG, buffer_radius, map_view_range, map_x_min, map_x_max, map_y_min, map_y_max
from geometry import PlainPoint

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')

background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')


# TODO: maybe add a Plotter class because OOP
# TODO: add compass and scale bar and legend
def plot_result(start_point, end_point):
    plotter = Plotter(CRS_BNG)
    fig, ax = plotter.get_figure()

    # plotter.add_vector(isle_of_wight, color='white', alpha=.6)
    # plotter.add_vector(road_nodes, markersize=2, color='grey', alpha=.3)
    # plotter.add_vector(road_links, linewidth=1, color='black', alpha=.5)
    plotter.add_buffer(start_point.get_geometry().x, start_point.get_geometry().y, radius=buffer_radius, alpha=.3)
    x_min = max(425000, start_point.get_geometry().x - map_view_range / 2)
    x_max = min(470000, start_point.get_geometry().x + map_view_range / 2)
    y_min = max(75000, start_point.get_geometry().y - map_view_range / 2)
    y_max = min(100000, start_point.get_geometry().y + map_view_range / 2)
    plotter.set_xy_lim(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    plotter.add_raster(background, cmap='terrain')

    ax.scatter([start_point.get_geometry().x, end_point.get_geometry().x],
               [start_point.get_geometry().y, end_point.get_geometry().y], s=15, color='black')

    ax.annotate('start', xy=(start_point.get_geometry().x, start_point.get_geometry().y), color='blue')
    ax.annotate('end', xy=(end_point.get_geometry().x, end_point.get_geometry().y), color='green')

    return plotter


if __name__ == '__main__':
    plter = plot_result(PlainPoint(450000, 85000), PlainPoint(450000, 85000))
    plter.show()
