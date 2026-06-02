#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

#!/usr/bin/env python3

import argparse
from PIL import Image

def make_concat(im1, im2, color=(255, 255, 255)):

    # Make a new canvas
    dst = Image.new('RGBA', (im1.width + im2.width, max(im1.height, im2.height)), color)

    # Paste image 1, vertical centering
    offset = (dst.height - im1.height) / 2
    dst.paste(im1, (0, int(offset)))

    # Paste image 2, vertical centering
    offset = (dst.height - im2.height) / 2
    dst.paste(im2, (im1.width, int(offset)))

    return dst

parser = argparse.ArgumentParser(description='Concats images side by side, left to right')
parser.add_argument('im1', help='First image')
parser.add_argument('im2', help='Second image')
parser.add_argument('dst_file', help='Where to save the output')
args = parser.parse_args()

im1 = Image.open(args.im1)
im2 = Image.open(args.im2)
make_concat(im1, im2).save(args.dst_file)
