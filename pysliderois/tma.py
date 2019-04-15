# coding: utf8
import numpy
from matplotlib import pyplot as plt
from skimage.morphology import dilation, erosion, closing
from skimage.morphology import disk, square
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.segmentation import mark_boundaries, relabel_sequential
from skimage.draw import rectangle
from joblib import Parallel, delayed
import itertools

# Usually, comfortable choice of slide level for the RAM is 5
RAM_LEVEL = 5
IMSPLIT_LEVEL = 2


# We need a function to crop images
def crop_image(image, tol=0):
    """
    Given an image and a tolerance value for black pixels,
    returns the corresponding cropped image, i.e. without black pixels.

    Arguments:
        - image: numpy ndarray, rgb image.
        - tol: float or int, tolerance value for black pixels.

    Returns:
        - x: xmin, xmax of the computed roi.
        - y: ymin, ymax of the computed roi.
        - cropped: numpy ndarray, cropped rgb image.
    """

    # Mask of non-black pixels (assuming image has a single channel).
    mask = image[:, :, 0] > tol

    # Coordinates of non-black pixels.
    coords = numpy.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]

    x = x0, x1
    y = y0, y1

    return x, y, cropped


def get_tissue(image, blacktol=0, whitetol=230):
    """
    Given an image and a tolerance on black and white pixels,
    returns the corresponding tissue mask segmentation, i.e. true pixels
    for the tissue, false pixels for the background.

    Arguments:
        - image: numpy ndarray, rgb image.
        - blacktol: float or int, tolerance value for black pixels.
        - whitetol: float or int, tolerance value for white pixels.

    Returns:
        - binarymask: true pixels are tissue, false are background.
    """

    binarymask = numpy.ones_like(image[:, :, 0], bool)

    for color in range(3):
        # for all color channel, find extreme values corresponding to black or white pixels
        binarymask = numpy.logical_and(binarymask, image[:, :, color] < whitetol)
        binarymask = numpy.logical_and(binarymask, image[:, :, color] > blacktol)

    return binarymask


def clean_tissue(tissuemask, holesize=3000):
    """
    Given a tissue mask and a tolerated hole size,
    returns a repaired mask, without holes.

    Arguments:
        - tissuemask: numpy ndarray of booleans, true values for tissue.
        - holesize: size of the holes to fill (inpixels).

    Returns:
        - cleanmask: numpy ndarray of booleans, true values for tissue.
    """

    # first repair fractures
    repaired = dilation(tissuemask)

    # then remove holes
    cleanmask = remove_small_objects(repaired, min_size=holesize)

    return cleanmask


def get_tma_rois(slide, whitetol=230, blacktol=0, holesize=3000):
    """
    Given a slide, returns a list of absolute rois for tissue circles.

    Arguments:
        - slide: openslide OpenSlide object.
        - whitetol: int or float, tolerance value for white pixels.
        - blacktol: int or float, tolerance value for black pixels.
        - holesize: int, max size of holes to fill when capturing tissue areas.

    Returns:
        - rois: numpy ndarray, shape=(n_rois, 4), rois[0]=[xmin, xmax, ymin, ymax].
    """

    image = numpy.asarray(slide.read_region((0, 0), RAM_LEVEL, slide.level_dimensions[RAM_LEVEL]))[:, :, 0:3]
    bi, bj, croppedimage = crop_image(image, tol=blacktol)

    binarytissue = get_tissue(croppedimage)

    tissuecleanmask = clean_tissue(binarytissue)

    labelimage = label(tissuecleanmask)

    # get rois
    labels = [l for l in numpy.unique(labelimage) if l > 0]
    rois = []
    for lab in labels:
        i, j = numpy.where(labelimage == lab)
        rois.append((i.min(), i.max(), j.min(), j.max()))

    # get absolute rois
    abs_rois = numpy.asarray(rois)
    abs_rois[:, 0:2] += bi[0]
    abs_rois[:, 2::] += bj[0]
    abs_rois *= (2**RAM_LEVEL)

    return abs_rois


def m_regular_seed(shape, width):
    maxi = width * int(shape[0] / width)
    maxj = width * int(shape[1] / width)
    col = numpy.arange(start=0, stop=maxj, step=width, dtype=int)
    line = numpy.arange(start=0, stop=maxi, step=width, dtype=int)
    for p in itertools.product(line, col):
        yield p


def get_sample_rois(slide, bbx, patchsize=250, whitetol=230, tissueratio=0.75):
    """
    Given a slide, the bounding box of a tma circle, a white value tolerance and a patchsize,
    returns a list of patch rois usable for processing (with tissue on it...).

    Arguments:
        - slide: openslide OpenSlide object.
        - bbx: numpy ndarray, (xmin, xmax, ymin, ymax), bounding box of the tma circle of interest.
        - whitetol: int or float value considered as white.
        - patchsize: int, pixel size of patches at full resolution.

    Returns:
        - rois: numpy ndarray, rois for each processable patch.
    """
    istart = bbx[0]
    jstart = bbx[2]
    di = bbx[1] - bbx[0]
    dj = bbx[3] - bbx[2]
    image = numpy.asarray(slide.read_region((jstart, istart), IMSPLIT_LEVEL, (int(dj / 2**IMSPLIT_LEVEL), int(di / 2**IMSPLIT_LEVEL))))[:, :, 0:3]

    # get tissue
    tissuemask = get_tissue(image, whitetol=whitetol)
    # fill holes
    tissuemask = closing(tissuemask, selem=disk(20))
    # get structuring element size of the patch
    width = int(patchsize / 2**IMSPLIT_LEVEL)
    channels = image.shape[2]
    elem = square(int(0.5 * width))
    # dilation with structuring element
    patchsamplingmask = erosion(tissuemask, selem=elem)
    rois = []
    # loop over patch locations
    for position in m_regular_seed(tissuemask.shape, width):

        i, j = position
        i -= int(0.5 * width)
        j -= int(0.5 * width)
        if patchsamplingmask[i:i + width, j:j + width].sum() > tissueratio * width * width:
            rois.append((i, i + width, j, j + width))

    rois = numpy.asarray(rois)
    rois *= 2**IMSPLIT_LEVEL
    rois[:, 0:2] += istart
    rois[:, 2::] += jstart

    return rois
