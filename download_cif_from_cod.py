#!/usr/bin/env python
# coding: utf-8

import urllib
import urllib.request
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='./COD-selection.txt',
                        help='path to the cod search result text')
    parser.add_argument('--output', default='./cif',
                        help='path to the output directory')
    args = parser.parse_args()
    return args

def download_cif(txt_path, path_output_dir):
    # reading text file
    with open(txt_path) as f:
        cif_urls = f.read().splitlines()

    # make the output directory
    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    
    # download CIFs
    for idx, cif_url in enumerate(cif_urls):
        print(str(idx)+'/'+str(len(cif_urls)-1))
        urllib.request.urlretrieve(
            cif_url,
            path_output_dir+'/'+cif_url.split('/')[-1],
        )

if __name__ == '__main__':
    args = parse_args()
    download_cif(args.input, args.output)

