#!/usr/bin/python
# GoogleMapDownloader.py
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import csv
from math import log
import urllib.request
from PIL import Image
import os
import math
import sys

NUM_IMAGES = 10
TILE_SIZE = 1280

SATIMG_INTERVALS = [(101, 199), (5011, 5199)]
# SATIMG_INTERVALS = [(101, 102), (5011, 5012)]


class GoogleMapsLayers:
    ROADMAP = "roadmap"
    TERRAIN = "terrain"
    SATELLITE = "satellite"
    HYBRID = "hybrid"


class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12, layer=GoogleMapsLayers.ROADMAP):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def project(self):

        siny = math.sin(self._lat * math.pi / 180)
        siny = math.min(math.max(-0.9999), 0.9999)
        return TILE_SIZE * (0.5 + self._lng / 360), TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = TILE_SIZE

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size /
                   360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
            tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 1)
        tile_height = kwargs.get('tile_height', 1)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = TILE_SIZE * tile_width, TILE_SIZE * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                print("lat = ", self._lat, ", lng = ", self._lng)
                # url = f'https://mt0.google.com/vt?lyrs={self._layer}&x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                # self._zoom)

                url = f'https://maps.googleapis.com/maps/api/staticmap?center={self._lat},{self._lng}&maptype={self._layer}&zoom={self._zoom}&scale=2&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'
                # url = f'https://maps.googleapis.com/maps/api/staticmap?v=beta&center={self._lat:.6f},{self._lng:.6f}&maptype={self._layer}&zoom={self._zoom}&scale=1&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'
                # url = f'https://maps.googleapis.com/maps/api/vt?v=beta&mapId=1&x=1&y=0&maptype={self._layer}&zoom={self._zoom}&scale=1&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'
                # print("url = ", url)
                # url = f'https://maps.googleapis.com/maps/api/staticmap?center=51.477222,0&maptype=satellite&zoom={self._zoom}&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'

                # print("start_x = ", start_x, ", start_y = ", start_y)
                # url = f'https://maps.googleapis.com/maps/api/staticmap?x=100&y=100&maptype=satellite&zoom={self._zoom}&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'

                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * TILE_SIZE, y * TILE_SIZE))

                os.remove(current_tile)

        return map_img


# import required module


def GetLatLong(filepath):
    f = open(filepath, "r")
    line = f.readline().split(" ")
    lat, lng = float(line[0]), float(line[1])
    return lat, lng


def main():

    if len(sys.argv) < 2:
        print("Error format.\npython3 getSatImg.py <zoom level>")
        return

    print("zoom level: " + str(sys.argv[1]))

    TRY_CATCH = False
    INPUT_DIR = '/mnt/workspace/users/leekt/kitti-360/KITTI-360/data_poses/'
    drive_name = '2013_05_28_drive_0000_sync/'
    # INPUT_DIR = '/mnt/workspace/users/leekt/kitti-raw/raw_data/2011_09_26/'
    # drive_name = '2011_09_26_drive_0001_sync/'
    OUTPUT_DIR = '/mnt/workspace/datasets/kitti-360-SLAM/satmap/'
    # OUTPUT_DIR = '/mnt/workspace/users/leekt/satimgs_test/'
    INPUT_DIR_END = 'oxts/data'

    output_image_dir = OUTPUT_DIR + drive_name
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir, exist_ok=True)

    directory = INPUT_DIR + drive_name + INPUT_DIR_END

    files = sorted(list(os.listdir(directory)))
    selected_files = []
    for start, end in SATIMG_INTERVALS:
        selected_files += files[start: end + 1]

    # # Get trajectory bound
    # lat_min, lat_max, lng_min, lng_max = 0, 0, 0, 0
    # for index, filename in enumerate(selected_files):
    #     filepath = os.path.join(directory, filename)
    #     lat, lng = GetLatLong(filepath)
    #     if index == 0:
    #         lat_min = lat_max = lat
    #         lng_min = lng_max = lng
    #     lat_min = min(lat_min, lat)
    #     lng_min = min(lng_min, lng)
    #     lat_max = max(lat_max, lat)
    #     lng_max = max(lng_max, lng)

    for filename in selected_files:
        filepath = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(filepath):
            # Extract filename
            fn = os.path.splitext(filename)[0]
            print("Input filename: " + fn)

            # If satimg already exists, continue
            output_filename = output_image_dir + fn + '.png'
            if os.path.isfile(output_filename):
                print(f"Filepath: {output_filename} already exist, skip for now")
                continue

            # Get (lat,long)
            lat, long = GetLatLong(filepath)
            print("(lat, long): " + str(lat) + ", " + str(long))

            # Generate satellite image
            # Create a new instance of GoogleMap Downloader
            gmd = GoogleMapDownloader(lat, long, int(
                sys.argv[1]), GoogleMapsLayers.SATELLITE)

            print("The tile coorindates are {}".format(gmd.getXY()))

            img = gmd.generateImage()
            # save images to disk
            img.save(output_filename)

        else:
            raise Exception(f"filepath: {filepath} doesn't exist")


if __name__ == '__main__':
    main()


# TODOs:
# 1. Use a for-loop to generate a list of (lat,lon)
# 2. Find out what resolution we want
# 3. Design the output file format (Name)=> Same as iput?
# e.g.


# Read input images from

"""
Kitti-360-SLAM/
    KITTI-360/
        data_2d_raw/
            2013_05_28_drive_0000_sync/
                image_00/
                    data_rect/
                        0000000000.png


Read: 
    KITTI-360/
        data_poses/
            2013_05_28_drive_0000_sync/
                oxts/
                    data/
                        0000000000.txt
                        0000000001.txt
                        0000000002.txt
                        0000000003.txt
First two numbers in xxxx.txt are the (lat, long) pairs!


=> Outpt:
KITTI-360/
    satmap/
        2013_05_28_drive_0000_sync/
            0000000000.png
"""
