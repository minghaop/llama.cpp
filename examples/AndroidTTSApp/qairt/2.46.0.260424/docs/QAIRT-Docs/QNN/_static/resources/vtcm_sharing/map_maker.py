#=============================================================================
#
#  Copyright (c) 2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================

#!/usr/bin/env python3

import argparse
import codecs
import drawSvg as draw
import numpy as np

colors = [
        '#a9a9a9', # grey
        '#4363d8', # blue
        '#3cb44b', # green
        '#e6194B', # red
        '#f032e6', # pink
        '#f58231', # orange
        ]


def generate_map(src_file, dst_file, square_size, padding, font_size):

    # Padding as a fraction of the input

    with codecs.open(src_file, encoding='utf-8-sig') as f:
        data = np.loadtxt(f, dtype=int)
        rows,cols = data.shape

        width = (padding[0] + square_size) * cols  - padding[0]
        height = (padding[1] + square_size) * (rows + 1) - padding[1]

        font_delta = int((square_size - font_size) / 2)

        drawing = draw.Drawing(width, height, displayInline=False)
    
        # Draw Infographic
        drawing.append(draw.Text('VTCM', font_size-4, 0, height-square_size +  font_delta))
        for i in range(1,cols):
            y = height - square_size
            x = i * (square_size + padding[0])
            color = colors[0]

            drawing.append(draw.Rectangle(x, y, square_size, square_size, fill=color))

            # Draw the number
            drawing.append(draw.Text(f'{i-1}', font_size, x+font_delta, y+font_delta))


        # Draw from grid
        for j in range(0,rows):

            y = height - ((j + 1) * (square_size + padding[1])) - square_size

            # Draw the number
            dy = int((square_size - font_size) / 2)
            drawing.append(draw.Text(f'[{data[j][0]:>2}]', font_size, 0, y+font_delta))

            for i in range(1,cols):
                color_index = data[j][i]

                if color_index > len(colors):
                    raise Exception('Using more colors than supported. Add a new color please')

                color = colors[color_index]
                x = i * (square_size + padding[0])

                # Draw a square
                drawing.append(draw.Rectangle(x, y, square_size, square_size, fill=color))

        drawing.savePng(dst_file)

parser = argparse.ArgumentParser(description='Generates VTCM usage images from a map file')
parser.add_argument('map_file', help='Numpy array file of thread IDs (0,1,2...)')
parser.add_argument('dst_file', help='Where to save the output (png file)')
parser.add_argument('--square', type=int, default=32, help='Default size for the squares')
parser.add_argument('--padding_top', type=int, default=8, help='Vertical Padding between cells')
parser.add_argument('--padding_ltor', type=int, default=3, help='Horizontal padding between cells')
parser.add_argument('--font_size', type=int, default=14)
args = parser.parse_args()

generate_map(args.map_file, args.dst_file, args.square, (args.padding_ltor, args.padding_top), args.font_size)

