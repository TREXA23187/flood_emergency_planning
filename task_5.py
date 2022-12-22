import geopandas as gpd
import rasterio
from plotter import Plotter
from constant import CRS_BNG

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')

background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')


# TODO: maybe add a Plotter class because OOP
# TODO: add compass and scale bar and legend
def plot_result():
    plotter = Plotter(CRS_BNG)

    plotter.add_vector(isle_of_wight, color='white', alpha=.6)
    # plotter.add_vector(road_nodes, markersize=2, color='grey', alpha=.3)
    plotter.add_vector(road_links, linewidth=1, color='black', alpha=.5)

    plotter.add_raster(background)

    return plotter


if __name__ == '__main__':
    plter = plot_result()
    plter.show()
