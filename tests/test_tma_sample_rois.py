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

sample_rois = tma.get_sample_rois(slide, rois[0])

print(sample_rois.shape)
print(sample_rois)
