# Maker options

import argparse

def maker(s):
    try:
        x, y = s.split(':')
        if type(x) is str: # This could be better
            if x == 'blobs': x = 1
            if x == 'moons': x = 2
            if x == 'circles': x = 3
            if x == 'classf': x = 4
            if x == 'multi_classf': x = 5
        y = float(y)
        return x, y
    except:
        raise argparse.ArgumentTypeError("Maker must be formatted as "
        "<Maker Option>:<Value of Option>")
