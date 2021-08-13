#!/usr/bin/env python
# coding: utf-8

from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import re
import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./cif',
                        help='path to the input cif directory')
    parser.add_argument('--output', default='./lithium_datasets.pkl',
                        help='name of the output pickle file')
    args = parser.parse_args()
    return args

class read_CIF():
    def __init__(self,
                 cif_dir_path):
        self.cif_dir_path = cif_dir_path
        self.cif_list = os.listdir(cif_dir_path)
    
    def cif2structure(self, cif_path):
        # convert CIF into pymatgen structure data
        parser = CifParser(cif_path)
        structure = parser.get_structures()[0]
        
        # get useful labels
        atoms = structure.formula.replace(",","").split()
        atoms = [re.sub(r'[0-9]',"",atom) for atom in atoms]
        finder = SpacegroupAnalyzer(structure)
        space_group = finder.get_space_group_symbol()
        material_formula = structure.formula
        
        return structure, atoms, space_group, material_formula
    
    def stochastic_dwf(self, atoms):
        # randomly select the Debye-Waller factors for each atom
        dwfs = []
        for atom in atoms:
            dwfs.append((atom, random.uniform(0.5,3)))
        dwfs = dict(dwfs)
        return dwfs

    def structure2XRDpeaks(self, structure, dwfs, verbose=False):
        xrd = XRDCalculator(
            debye_waller_factors=dwfs,
        ).get_pattern(
            structure,
            scaled=True,
            two_theta_range=(0,120),
        )
        if verbose:
            plt.vlines(xrd.x, 0, xrd.y, lw=0.5)
        return xrd
    
    def cif_availability(self, cif_path):
        try:
            converted_data = self.cif2structure(cif_path)
            dwfs = self.stochastic_dwf(converted_data[1])
            xrd = self.structure2XRDpeaks(converted_data[0], dwfs)
            cif_avail = True
        except:
            cif_avail = False
            converted_data = 0
        return cif_avail, converted_data

    def structure_loader(self):
        structures = []
        atoms = []
        attributes = []
        for count, cif_name in enumerate(self.cif_list):
            cif_path = self.cif_dir_path + '/' + cif_name
            cif_avail, converted_data = self.cif_availability(cif_path)
            if cif_avail:
                structures.append(converted_data[0])
                atoms.append(converted_data[1])
                attributes.append(converted_data[2:])
                print('Loading OK: '+ cif_name)
            else:
                print('Loading NG: '+ cif_name)
            print(str(count+1)+' / '+str(len(self.cif_list))+" completed.")
        return structures, atoms, attributes

if __name__ == '__main__':
    args = parse_args()
    cif = read_CIF(args.input)
    cod_datasets = cif.structure_loader()
    with open(args.output, mode='wb') as f:
        pickle.dump(cod_datasets, f)
    print('Successfully loaded.')

