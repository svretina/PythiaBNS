#!/usr/bin/env python3

# Copyright (C) 2022 Stamatis Vretinaris, Christos Mermigkas
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

import os
from gw_pipe import config as cfg
from gw_pipe import global_vars

config_path = global_vars.config_path


def make_structure():
    if not os.path.exists(global_vars.strain_path):
        os.mkdir(global_vars.strain_path)
    if not os.path.exists(global_vars.results_path):
        os.mkdir(global_vars.results_path)


def list_to_download():
    config = cfg.Config()
    config = config.config_dict
    # read if you want actually to download the waveforms
    if config["Download"]['download']:
        names2download = config["Download"]["names"]
        # check if already there and delete them from list
        already_there = os.listdir(global_vars.strain_path)
        for i in already_there:
            if i in names2download:
                names2download.remove(i)
        return names2download
    return None


def download_data():
    names = list_to_download()
    for name in names:
        # create directory for downloading data
        nr_data_path = f"{global_vars.project_path}/NR_strains/{name}"
        os.mkdir(nr_data_path)

        # split name in three parts
        first = name.split(":")[0]
        second = name.split(":")[1]
        third = name.split(":")[2]

        # download hdf5 data file
        os.system(
            f"wget https://core-gitlfs.tpi.uni-jena.de/core_database/{first}_{second}/-/raw/master/{third}/data.h5 -P {nr_data_path}"
        )

        # download metadata
        os.system(
            f"wget https://core-gitlfs.tpi.uni-jena.de/core_database/{first}_{second}/-/raw/master/{third}/metadata.txt -P {nr_data_path}"
        )


if __name__ == "__main__":
    make_structure()
    download_data()
