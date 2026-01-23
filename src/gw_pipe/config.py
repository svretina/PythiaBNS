#!/usr/bin/env python3

# Copyright (C) 2022 Stamatis Vretinaris
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

import configparser
import numpy as np
from itertools import product
from gw_pipe import global_vars as glb


class Config:
    def __init__(self,config_path=None):
        if config_path is None:
            self.config_path = glb.config_path
        else:
            self.config_path = f"{glb.project_path}/configs/{config_path}"
        self.config = self.read_config()
        self.config_dict = self.get_config_dict()

    def read_config(self):
        """Reads a config file and returns a dictionary with its
        contents.

        :returns: Dictionary with the values of the config file
              variable.
        :rtype: configparser.ConfigParser
        """
        config = configparser.ConfigParser()
        config.read(self.config_path)
        return config

    def get_config_dict(self):
        """Creates numpy arrays spaced according to the steps
        provided in the config file.

        :param cfg_parser: Dictionary containing the config information
        :type cfg_parser: configparser.ConfigParser
        :returns: Dictionary with the numpy arrays
        :rtype: dict
        """
        cfg_parser = self.config
        config_dict = {}
        for section in cfg_parser.keys():
            try:
                content = dict()
                for subsection in cfg_parser[section].keys():
                    temp_array = cfg_parser[section][subsection].split(",")
                    tmp = [i.strip("\n") for i in temp_array]
                    subcontent = list()
                    for i in tmp:
                        if ":" in i:
                            try:
                                start, step, end = [float(j) for j in i.split(":")]
                                array = np.linspace(
                                    start, end, int((end - start) / step + 1)
                                )
                                subcontent = list(array)
                            except ValueError:
                                subcontent.append(i)
                        elif i.lower() == 'true':
                            subcontent.append(True)
                        elif i.lower() == 'false':
                            subcontent.append(False)
                        else:
                            if "." in i or "e" in i or "E" in i:

                                try:
                                    subcontent.append(float(i))
                                except ValueError:
                                    subcontent.append(i)
                            else:
                                try:
                                    subcontent.append(int(i))
                                except ValueError:
                                    subcontent.append(i)

                    if len(subcontent) > 1:
                        try:
                            content[subsection] = np.concatenate(subcontent)
                        except ValueError:
                            content[subsection] = subcontent
                    else:
                        content[subsection] = subcontent[0]

                    config_dict[section] = content
            except (ValueError, KeyError):
                config_dict[section] = np.nan
        return config_dict

    def create_iterator_dict(self):
        """Creates a list of simulation dictionaries by taking the Cartesian product
        of all parameters in the configuration file.
        """
        # Ensure all values are lists for product
        processed_cfg = {}
        for section in ["Sampler", "Model", "Run", "Priors"]:
            if section in self.config_dict:
                for key, val in self.config_dict[section].items():
                    if not isinstance(val, (list, np.ndarray)):
                        processed_cfg[key] = [val]
                    else:
                        processed_cfg[key] = list(val)

        # Handle 'times' separately if needed, but it's usually in Run
        if 'times' in processed_cfg:
            n_runs = processed_cfg.pop('times')
            processed_cfg['run_id'] = list(range(max(n_runs) if isinstance(n_runs, list) else n_runs))
        else:
            processed_cfg['run_id'] = [0]

        # Rename keys to match expected names in spine.py if they differ
        # (e.g., name -> model if it's in the Model section)
        # But wait, the current spine.py uses 'model' for the model name.
        # In config.cfg, it's [Model] name = last_hope. 
        # So we need to map 'name' from [Model] to 'model'.
        
        # Let's do a more explicit mapping to avoid confusion
        final_params = {}
        if 'Sampler' in self.config_dict:
            final_params['sampler'] = self.config_dict['Sampler']['name']
            if not isinstance(final_params['sampler'], list): final_params['sampler'] = [final_params['sampler']]
            for k, v in self.config_dict['Sampler'].items():
                if k != 'name':
                    final_params[k] = v if isinstance(v, list) else [v]
                    
        if 'Model' in self.config_dict:
            final_params['model'] = self.config_dict['Model']['name']
            if not isinstance(final_params['model'], list): final_params['model'] = [final_params['model']]
            for k, v in self.config_dict['Model'].items():
                if k != 'name':
                    final_params[k] = v if isinstance(v, list) else [v]

        if 'Run' in self.config_dict:
            # 'waveforms' maps to 'waveform' in spine.py
            final_params['waveform'] = self.config_dict['Run']['waveforms']
            if not isinstance(final_params['waveform'], list): final_params['waveform'] = [final_params['waveform']]
            final_params['times'] = list(range(self.config_dict['Run']['times'])) if not isinstance(self.config_dict['Run']['times'], list) else list(range(max(self.config_dict['Run']['times'])))
            for k, v in self.config_dict['Run'].items():
                if k not in ['waveforms', 'times', 'n_cpus']:
                    final_params[k] = v if isinstance(v, list) else [v]

        if 'Priors' in self.config_dict:
            for k, v in self.config_dict['Priors'].items():
                final_params[k] = v if isinstance(v, list) else [v]

        # Create product
        keys = list(final_params.keys())
        values = list(final_params.values())
        
        dict_list = []
        for combination in product(*values):
            sim_dict = dict(zip(keys, combination))
            dict_list.append(sim_dict)
            
        return dict_list
