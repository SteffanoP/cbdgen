# Complexity Measures

import argparse

def cm(s) -> tuple[str, float]:
    try:
        x, y = s.split(':')
        y = float(y)
        return x, y
    except:
        raise argparse.ArgumentTypeError("Complexity Measures must be "
        "Formatted as <Complexity Acronym>:<Value>")