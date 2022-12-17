import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio import plot as raster_plot
from pyproj import CRS

CRS_BNG = CRS('epsg:27700')

isle_of_wight = gpd.read_file('Material/shape/isle_of_wight.shp')
road_links = gpd.read_file('Material/roads/links.shp')
road_nodes = gpd.read_file('Material/roads/nodes.shp')

background = rasterio.open('Material/background/raster-50k_2724246.tif')
elevation = rasterio.open('Material/elevation/SZ.asc')


# TODO: maybe add a Plotter class because OOP
# TODO: add compass and scale bar and legend
def plot_sample():
    fig_base = plt.figure(figsize=(9, 6), dpi=100)
    ax = fig_base.add_subplot(111)
    ax.set_axis_off()

    isle_of_wight.to_crs(CRS_BNG).plot(ax=ax, color='white')
    road_nodes.to_crs(CRS_BNG).plot(ax=ax, markersize=2, color='grey')
    road_links.to_crs(CRS_BNG).plot(ax=ax, linewidth=1, cmap='RdYlGn', column='descript_1')

    raster_plot.show(background, ax=ax)

    return ax


if __name__ == '__main__':
    plot_sample()

    plt.show()
