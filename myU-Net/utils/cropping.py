import numpy as np


def create_nonzero_mask(data, min):
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data > min
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    if len(mask.shape)==3:
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    else:
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1]))
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        bbox = [[minzidx, maxzidx], [minxidx, maxxidx]]
    return bbox


def crop_to_bbox(image, bbox):
    if len(image.shape)==3:
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    else:
        resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]))
    return image[resizer]

def adjust_to_bbox(image, bbox, dsize, hsize, wsize):
    original = np.zeros([dsize, hsize, wsize])
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    original[resizer] = image
    #box = original[resizer]
    #D = np.minimum(box.shape[0], image.shape[0])
    #H = np.minimum(box.shape[1], image.shape[1])
    #W = np.minimum(box.shape[2], image.shape[2])
    #original[resizer] = image[:D,:H,:W]
    return original

def crop_to_nonzero(data, min):

    nonzero_mask = create_nonzero_mask(data, min)
    bbox = get_bbox_from_mask(nonzero_mask, 0)
    data_crop = crop_to_bbox(data, bbox)

    return data_crop, bbox

