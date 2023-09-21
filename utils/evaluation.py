import numpy as np

## Computes mean PAP for all images (within category)
def compute_mean_pap(evaluate_results):
    # average values
    mPAP = 0.
    mPAP_strict = 0.
    mPAP_hoa = 0.
    mPAP_strict_hoa = 0.
    PAP_count = 0
    PAP_count_hoa = 0
    for img_name in evaluate_results['results'].keys():
        image_results = evaluate_results['results'][img_name]
        if image_results['target_area'] == 0:
            # since the target does not exist, we do not account as evaluation
            continue
        elif image_results['target_area_hoa'] == 0:
            # just skip the hoa accumulation
            mPAP += image_results['PAP']
            mPAP_strict += image_results['PAP_strict']
            PAP_count += 1
        else:
            # accumulate for both                    
            mPAP += image_results['PAP']
            mPAP_strict += image_results['PAP_strict']
            PAP_count += 1
            # human occlusion aware
            mPAP_hoa += image_results['PAP_hoa']
            mPAP_strict_hoa += image_results['PAP_strict_hoa']
            PAP_count_hoa += 1
    
    ## add average to dict
    # 'mean-projective-average-precision'
    mPAP /= max(1,PAP_count)
    mPAP_strict /= max(1,PAP_count)
    # 'mean-projective-average-precision' (human occlusion aware)
    mPAP_hoa /= max(1,PAP_count_hoa)
    mPAP_strict_hoa /= max(1,PAP_count_hoa)
    # add to dict
    evaluate_results['results']['mPAP'] = mPAP
    evaluate_results['results']['mPAP_strict'] = mPAP_strict
    # hoa
    evaluate_results['results']['mPAP_hoa'] = mPAP_hoa
    evaluate_results['results']['mPAP_strict_hoa'] = mPAP_strict_hoa
    
    return evaluate_results


def compute_pap(pred, target_mask, verbose):
    """
        pred: A dictionary where
            -key: threshold
            -value: dict of
                - 'human_mask': Human mask rendered
                - 'object_mask': Object mask rendered at 'threshold'
                        
        target: A target object mask    
    """
    # precision-recall list
    precision_recall_list, precision_recall_hoa_list = return_precision_recall_list(
        pred=pred, 
        target_mask=target_mask,
        verbose=verbose
    )
    
    PAP, PAP_strict, target_area = projective_average_precision(precision_recall_list)
    PAP_hoa, PAP_strict_hoa, target_area_hoa = projective_average_precision(precision_recall_hoa_list)
    assert target_area >= target_area_hoa

    return PAP, PAP_strict, PAP_hoa, PAP_strict_hoa, target_area, target_area_hoa


def return_precision_recall_list(pred, target_mask, verbose):
    ## placeholder to save pap values
    precision_recall_list = []
    precision_recall_hoa_list = []

    # iterate for thresholds
    for _threshold in pred.keys():
        # threshold
        threshold = float(_threshold)

        # object mask and human mask
        pred_mask = pred[_threshold]['object_mask']
        human_mask = pred[_threshold]['human_mask']

        # human-occlusion aware masks
        pred_mask_hoa = np.logical_and(pred_mask, 1-human_mask)
        target_mask_hoa = np.logical_and(target_mask, 1-human_mask)

        # pixel-wise precision & recall
        target_area = float(target_mask.sum())
        pred_area = float(pred_mask.sum())
        intersection = np.logical_and(target_mask, pred_mask)
        precision = float(intersection.sum() / max(1, pred_mask.sum()))
        recall = float(intersection.sum() / max(1,target_mask.sum()))
        if pred_area > 0 and verbose: print(f"======\nThreshold:{threshold}\nPrecision:{precision}\nRecall:{recall}\nPrediction Area:{pred_area}")

        # human-occlusion aware precision & recall
        target_area_hoa = float(target_mask_hoa.sum())
        pred_area_hoa = float(pred_mask_hoa.sum())
        intersection_hoa = np.logical_and(target_mask_hoa, pred_mask_hoa)
        precision_hoa = float(intersection_hoa.sum() / max(1,pred_mask_hoa.sum()))
        recall_hoa = float(intersection_hoa.sum() / max(1,target_mask_hoa.sum()))
        if pred_area_hoa > 0 and verbose: print(f"======\nThreshold:{threshold}\nPrecision (human-occlusion-aware):{precision_hoa}\nRecall (human-occlusion-aware):{recall_hoa}\nPrediction Area (human-occlusion-aware):{pred_area_hoa}")


        # save-to-list
        precision_recall_list.append(
            (threshold, precision, recall, pred_area, target_area)
        )
        precision_recall_hoa_list.append(
            (threshold, precision_hoa, recall_hoa, pred_area_hoa, target_area_hoa)
        )

    return precision_recall_list, precision_recall_hoa_list
    

def projective_average_precision(precision_recall_list: list):
    PAP = 0. # original version -> do not accumulate when precision starts to become 0.
    PAP_strict = 0. # strict -> set precision as 0 if nothing is predicted
    # what we have is a list of 3-tuple (theshold, precision, recall)
    precision_to_accumulate = precision_recall_list[0][1] # precision
    recall_before = precision_recall_list[0][2]
    target_area = precision_recall_list[0][-1]
    for threshold, precision, recall, prediction_area, target_area_ in precision_recall_list[1:]:                        
        assert target_area == target_area_
        
        # get the width of the bin in precision-recall graph
        assert recall <= recall_before
        width = recall_before - recall
        
        # accumulate PAP
        if prediction_area == 0:
            PAP += width * precision_to_accumulate
            PAP_strict += 0.
        else:
            PAP += width * precision_to_accumulate
            PAP_strict += width * precision_to_accumulate

        # update "precision_to_accumulate"
        if precision > precision_to_accumulate:
            precision_to_accumulate = precision
        else:
            # keep
            pass

        # update "recall before"
        recall_before = recall

    return PAP, PAP_strict, target_area