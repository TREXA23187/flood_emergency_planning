import matplotlib.pyplot as plt
from rasterio import plot as raster_plot
import matplotlib.patches as mpatches
from shapely.geometry import Point
import geopandas as gpd
from constant import CRS_BNG


# get distance between 2 given points
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
        self.__legend_ax = None
        self.__legends = []

    def get_figure(self):
        return self.__base_figure, self.__ax

    def add_point(self, x, y, **kwargs):
        self.__ax.plot(x, y, 'o', **kwargs)

    def add_vector(self, vector, **kwargs):
        vector.to_crs(self.crs).plot(ax=self.__ax, **kwargs)

    def add_raster(self, raster, **kwargs):
        raster_image = raster_plot.show(raster, ax=self.__ax, **kwargs)
        im = raster_image.get_images()[0]

        cax = self.__base_figure.add_axes(
            [self.__ax.get_position().x1 + 0.01, self.__ax.get_position().y0, 0.015,
             self.__ax.get_position().height])
        self.__legend_ax = cax
        self.__base_figure.colorbar(im, cax=cax)

    def add_artist(self, artist):
        self.__ax.add_artist(artist)

    # reset range of xy
    def set_xy_lim(self, x_min, x_max, y_min, y_max):
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    def add_north(self, label_size=10, loc_x=0.05, loc_y=1, width=0.04, height=0.06, pad=0.1):
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
        y_len = maxy - miny
        x_len = maxx - minx
        left = [minx + x_len * (loc_x - width * .5), miny + y_len * (loc_y - pad)]
        right = [minx + x_len * (loc_x + width * .5), miny + y_len * (loc_y - pad)]
        top = [minx + x_len * loc_x, miny + y_len * (loc_y - pad + height)]
        center = [minx + x_len * loc_x, left[1] + (top[1] - left[1]) * .4]
        triangle = mpatches.Polygon([left, top, right, center], color='k')
        ax.text(s='N',
                x=minx + x_len * loc_x,
                y=miny + y_len * (loc_y - pad + height),
                fontsize=label_size,
                horizontalalignment='center',
                verticalalignment='bottom')
        ax.add_patch(triangle)

    def add_scale_bar(self, label_size=10, loc_x=0.85, loc_y=0.15, length=2000, size=200, pad=0.1):
        """
        lon0: longitude
        lat0: latitude
        length: length
        size: size
        """
        ax = self.__ax
        min_x, max_x = ax.get_xlim()
        min_y, max_y = ax.get_ylim()
        x_len = max_x - min_x
        y_len = max_y - min_y

        lon = min_x + x_len * loc_x
        lat = min_y + y_len * (loc_y - pad)

        ax.hlines(y=lat, xmin=lon, xmax=lon + length, colors="black", ls="-", lw=1)
        ax.vlines(x=lon, ymin=lat, ymax=lat + size, colors="black", ls="-", lw=1)
        ax.vlines(x=lon + length, ymin=lat, ymax=lat + size, colors="black", ls="-", lw=1)

        scale_distance = distance([lon, lat], [lon + length, lat])
        ax.text(s=f'{int(scale_distance / 1000)} km',
                x=lon + length / 2,
                y=lat + 2 * size,
                fontsize=label_size,
                horizontalalignment='center',
                verticalalignment='bottom')

    def add_legend(self, legend_type, legend_label, **kwargs):
        ax = self.__ax
        if legend_type == 'point':
            point_legend, = ax.plot([], 'o', label=legend_label, **kwargs)
            self.__legends += [point_legend]
        elif legend_type == 'line':
            line_legend, = ax.plot([], label=legend_label, **kwargs)
            self.__legends += [line_legend]
        else:
            polygon_legend = mpatches.Patch(label='isle of wight', **kwargs)
            self.__legends += [polygon_legend]

    def show_legend(self):
        self.__ax.legend(handles=self.__legends, loc='center left', fontsize='x-small', bbox_to_anchor=(0, .1))

    def show(self):
        self.__ax.set_title('Feasible path planning', fontsize=17)
        self.__legend_ax.set_title('Elevation/m', fontsize=8)
        self.add_north()
        self.add_scale_bar()
        self.show_legend()

        plt.show()
