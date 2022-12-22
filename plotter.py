import matplotlib.pyplot as plt
from rasterio import plot as raster_plot
import matplotlib.patches as mpatches
from shapely.geometry import Point
import geopandas as gpd
from constant import CRS_BNG


def distance(p1, p2, crs=CRS_BNG):
    pnt1 = Point(p1[0], p1[1])
    pnt2 = Point(p2[0], p2[1])
    points_df = gpd.GeoDataFrame({'geometry': [pnt1, pnt2]}, crs=crs)
    points_df2 = points_df.shift()

    distance_df = points_df.distance(points_df2)
    distance_df = distance_df.reset_index()
    distance_df.columns = ['index', 'distance']
    distance_df.index = ['d1', 'd2']

    return distance_df.loc['d2', 'distance']


class Plotter:
    def __init__(self, crs):
        self.__base_figure = plt.figure(figsize=(9, 6), dpi=100)
        self.__ax = self.__base_figure.add_subplot(111)
        self.__ax.set_axis_off()

        self.crs = crs

    def get_figure(self):
        return self.__base_figure, self.__ax

    def add_vector(self, vector, **kwargs):
        vector.to_crs(self.crs).plot(ax=self.__ax, **kwargs)

    def add_raster(self, raster, **kwargs):
        raster_plot.show(raster, ax=self.__ax, **kwargs)

    def add_north(self, label_size=10, loc_x=0.95, loc_y=0.95, width=0.05, height=0.1, pad=0.1):
        """
        Add a north arrow with 'N' text note
        :param label_size: size of 'N'
        :param loc_x: Horizontal proportion of the lower part of the text to ax
        :param loc_y: Vertical proportion of the lower part of the text to ax
        :param width: Proportion width of north arrow in ax
        :param height: Proportion height of north arrow in ax
        :param pad: Proportion clearance of 'N' in ax
        :return: None
        """
        ax = self.__ax
        minx, maxx = ax.get_xlim()
        miny, maxy = ax.get_ylim()
        ylen = maxy - miny
        xlen = maxx - minx
        left = [minx + xlen * (loc_x - width * .5), miny + ylen * (loc_y - pad)]
        right = [minx + xlen * (loc_x + width * .5), miny + ylen * (loc_y - pad)]
        top = [minx + xlen * loc_x, miny + ylen * (loc_y - pad + height)]
        center = [minx + xlen * loc_x, left[1] + (top[1] - left[1]) * .4]
        triangle = mpatches.Polygon([left, top, right, center], color='k')
        ax.text(s='N',
                x=minx + xlen * loc_x,
                y=miny + ylen * (loc_y - pad + height),
                fontsize=label_size,
                horizontalalignment='center',
                verticalalignment='bottom')
        ax.add_patch(triangle)

    def add_scale_bar(self, lon0, lat0, length=2000, size=200):
        """
        lon0: longitude
        lat0: latitude
        length: length
        size: size
        """
        ax = self.__ax
        ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length, colors="black", ls="-", lw=1, label=f'{length} km')
        ax.vlines(x=lon0, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)
        ax.vlines(x=lon0 + length, ymin=lat0, ymax=lat0 + size, colors="black", ls="-", lw=1)

        scale_distance = distance([lon0, lat0], [lon0 + length, lat0])

        ax.text(lon0 + length / 2, lat0 + 2 * size, f'{int(scale_distance / 1000)} km', horizontalalignment='center')

    # TODO: add legend
    def add_legend(self):
        ax = self.__ax
        isle_of_wight = mpatches.Patch(color='white', label='isle of wight',alpha=.5)

        road, = ax.plot([], label="road", color='black')
        shortest_distance_path, = ax.plot([], label="shortest distance path", color='red', linewidth=3)
        shortest_time_path, = ax.plot([], label="shortest time path", color='blue', linewidth=3)

        ax.legend(handles=[isle_of_wight, road, shortest_distance_path, shortest_time_path], loc='lower left',
                  fontsize='small')

    def show(self):
        self.add_north()
        self.add_scale_bar(lon0=462500, lat0=77000)
        self.add_legend()

        plt.show()
