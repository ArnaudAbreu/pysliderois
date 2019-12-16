# coding: utf8
from skimage.exposure import is_low_contrast
import numpy
from openslide import OpenSlide
from pysliderois.tissue import *
from pysliderois.util import *
import argparse
import os
from skimage.io import imsave

parser = argparse.ArgumentParser()

parser.add_argument("--level", type=int, default=5,
                    help="int, pyramid level, sample resolution.")

parser.add_argument("--interval", type=int, default=125,
                    help="int, interval between two samples, at level=--level.")

parser.add_argument("--size", type=int, default=125,
                    help="int, size of sample on x axis, at level=--level.")

parser.add_argument("--infolder", type=str, help="path to slide folder.")

parser.add_argument("--outfolder", type=str, help="path to outfolder.")

parser.add_argument("--ratiotissue", type=float, default=0.5, help="tissue ration per patch.")

args = parser.parse_args()


slidepaths = slides_in_folder(args.outfolder)


def m_regular_seed(shape, width):
    maxi = width * int(shape[0] / width)
    maxj = width * int(shape[1] / width)
    col = numpy.arange(start=0, stop=maxj, step=width, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=width, dtype=int)
    for p in itertools.product(line, col):
        yield p


def slide_rois(slide, level, interval):

    dim = slide.level_dimensions[level]

    for i, j in m_regular_seed((dim[1], dim[0]), interval):

        yield i * (2 ** level), j * (2 ** level)


for slidepath in slidepaths:

    slide = OpenSlide(slidepath)

    rois = slide_rois(slide, args.level, args.size*int(1.5**max(0, 5-args.level)))

    base_name = slide_basename(slidepath)

    base_dir = os.path.join(args.outfolder, base_name)

    for i, j in rois:

        outname = base_name + '_' + str(j) + '_' + str(i) + '.png'

        image = slide.read_region((j, i), args.level, (args.size, args.size))

        image = numpy.array(image)[..., :3]

        mask = get_tissue(image)

        if not is_low_contrast(image) and mask.sum() > args.ratiotissue * args.size * args.size:

            imsave(os.path.join(base_dir, outname), image)
