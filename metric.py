import numpy as np


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou


def AP(annot, rois, class_pred, iou_threshold = 0.4):    
    classes = [0,1]
    ret = []

    bboxs = np.array(annot[:,:4], np.int)
    labels = np.array(annot[:,4], np.int)

    for c in classes:

        n_pos = 0
        bboxs_ = bboxs[labels == c]    # gt
        rois_ = rois[class_pred == c]  # detect


        flag = np.zeros(len(bboxs_))

        n_pos += len(bboxs_)

        TP = np.zeros(len(rois_))
        FP = np.zeros(len(rois_))

        for r in range(len(rois_)):
            iou_max = 0

            for b in range(len(bboxs_)):
                iou = IoU(rois_[r],bboxs_[b])
                if iou > iou_max:
                    iou_max = iou
                    b_max = b

            if iou_max >= iou_threshold:
                if flag[b_max] == 0:
                    TP[r] = 1       
                    flag[b_max] = 1
                else:
                    FP[r] = 1       
            else:
                FP[r] = 1


        FP = np.sum(FP)
        TP = np.sum(TP)
        rec = TP / n_pos
        prec = np.divide(TP, (FP + TP))


        r = {
        'class': c,
        'precision': prec,
        'recall': rec,
        'total positives': n_pos,
        'total TP': np.sum(TP),
        'total FP': np.sum(FP) }

        ret.append(r)
    return ret