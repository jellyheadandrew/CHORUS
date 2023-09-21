""" Modified 'def get_iou()' from https://github.com/charlesq34/votenet-1/blob/master/utils/box_util.py """
def bbox_iou(bb1, bb2, return_areas_too=False):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : ndarray
        x1, y1, x2, y2
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : ndarray
        x1, y1, x2, y2
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])



    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])


    if x_right < x_left or y_bottom < y_top:
        if return_areas_too:
            return 0.0, bb1_area, bb2_area, 0.0
        else:
            return 0.0
        
    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)


    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    if return_areas_too:
        return iou, bb1_area, bb2_area, intersection_area
    else:
        return iou



def intersection_over_smaller_bbox(bb1, bb2):
    """
    Calculate the 'Intersection over Smaller Boudning-Box' of two bounding boxes.

    Parameters
    ----------
    bb1 : ndarray
        x1, y1, x2, y2
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : ndarray
        x1, y1, x2, y2
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # compute areas of 'bb1', 'bb2', 'intersection'
    _, bb1_area, bb2_area, intersection_area = bbox_iou(bb1, bb2, return_areas_too=True)
    
    # compute 'intersection-over-small-bounding-box'
    iosbb = intersection_area / min(bb1_area, bb2_area)

    return iosbb