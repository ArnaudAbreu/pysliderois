# coding: utf8
import numpy
from openslide import OpenSlide
from skimage.segmentation import mark_boundaries
from skimage.draw import rectangle
from matplotlib import pyplot as plt
from pysliderois import tma

slidepath = '/Users/administrateur/Pictures/FondClair/sarcome/SLG_LMSICGC_TMA-2B-ICGC.mrxs'
slide = OpenSlide(slidepath)

rois = tma.get_tma_rois(slide)

print(rois)
