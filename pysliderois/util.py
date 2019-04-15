# coding: utf8
import numpy


def iterpatches(slide, rois):

    for patch in rois:
        istart = patch[0]
        jstart = patch[2]
        di = patch[1] - patch[0]
        dj = patch[3] - patch[2]

        image = slide.read_region((jstart, istart), 0, (dj, di))
        image = numpy.array(image)[:, :, 0:3]

        yield patch, image


def listpatches(slide, rois):
    lp = []
    li = []

    for patch in rois:
        istart = patch[0]
        jstart = patch[2]
        di = patch[1] - patch[0]
        dj = patch[3] - patch[2]

        image = slide.read_region((jstart, istart), 0, (dj, di))
        image = numpy.array(image)[:, :, 0:3]
        li.append(image)
        lp.append(patch)

    return lp, li
