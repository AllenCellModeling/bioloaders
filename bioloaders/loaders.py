"""
A utility class to load various types of images and convert them to numpy.arrays

Author: Sarah Kefayati
Email: sara61@gmail.com
"""

__author__ = "Sarah Kefayati"
__email__ = "sara61@gmail.com"

import os
import shutil

import bioformats as bf
import javabridge as jv
import numpy as np
import pandas as pd


class Loaders:

    @staticmethod
    def lif_reader(lif_file_path, pickled_data_path):
        """
        This function reads lif file, separates channels, series, and time points and
        converts the images into 3D numpy.arrays. It then stores the images into pickle
        files and saves their path in a pandas.dataframe.

        :param lif_file_path: Input lif file.
        :param pickled_data_path: Path of the folder to store pickled images.
        :return: A dataframe containing the metadata for pickled images.
        """
        jv.start_vm(class_path=bf.JARS, max_heap_size='8G')
        md = bf.get_omexml_metadata(lif_file_path)
        o = bf.OMEXML(md)
        n_channels = o.image().Pixels.channel_count
        n_series = o.get_image_count()
        time_points = o.image().Pixels.SizeT
        size_x = o.image().Pixels.SizeX
        size_y = o.image().Pixels.SizeY
        size_z = o.image().Pixels.SizeZ
        rdr = bf.ImageReader(lif_file_path, perform_init=True)

        if os.path.exists(pickled_data_path):
            shutil.rmtree(pickled_data_path)
        os.makedirs(pickled_data_path)

        records = []

        for c in range(0, n_channels):
            for n in range(n_series):
                for t in range(time_points):
                    img_out = np.empty(shape=(size_x, size_y, size_z))

                    full_path_out = os.path.join(pickled_data_path, 'image_c{}_n{}_t{}'.format(c, n, t))

                    if not os.path.exists(full_path_out):
                        for z in range(size_z):
                            img_out[:, :, z] = rdr.read(c=c, z=z, t=t, series=n, rescale=False)
                        np.save(full_path_out, img_out, allow_pickle=True, fix_imports=True)

                    records.append((full_path_out, c, n, t))

        df = pd.DataFrame.from_records(records, columns=['image_path', 'channel', 'series', 'time'])

        jv.kill_vm()
        return df
