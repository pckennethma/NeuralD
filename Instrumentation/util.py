from copy import copy
import numpy as np

histogram = None

def instrumentation_visited(bb_no:int):
    histogram[bb_no] += 1

def flush(max_bb_no):
    global histogram
    histogram = np.zeros(max_bb_no)

def dump_hist():
    return copy(histogram)
