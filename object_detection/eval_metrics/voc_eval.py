import numpy as np
import os
from xml.etree.ElementTree import parse as ET_parse
from object_detection.data.data_loader import VOCDataset
import json


class VOCMetric:
    def __init__(self, root_path="C:/Users/98311/Downloads/VOCtest_06-Nov-2007/VOCdevkit", json_file='./data/pascal_classes_2007.json',
                 detect_path='./results', year=2007, use_2007=True):
        self.use_2007 = use_2007
        self.annotation_path = f"{root_path}/VOC{year}/Annotations"
        self.imageset_path = f"{root_path}/VOC{year}/ImageSets/Main"
        self.detect_path = detect_path
        f = open(json_file)
        self.map_dict = json.load(f)
        f.close()

    def evaluate(self):
        map = 0.
        for label in self.map_dict:
            _, _, ap = self.eval_per_label(label)
            map += ap
        return map / len(self.map_dict)

    def voc_ap(self, rec, prec):
        if self.use_2007:
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11
        # TODO: Complete metrics after 2012
        else:
            ap = 0
        return ap

    def parse_rec(self, filename):
        label = VOCDataset.get_xml_dict(ET_parse(f"{self.annotation_path}/{filename}").getroot())
        objects = []
        for obj in label['annotation']['object']:
            target = {}
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            if "difficult" in obj:
                target['difficult'] = int(obj["difficult"])
            else:
                target['difficult'] = 0
            target['box'] = [xmin, ymin, xmax, ymax]
            target['name'] = obj['name']
            target['truncated'] = int(obj['truncated'])
            target['pose'] = obj['pose']

            objects.append(target)

        return objects

    def eval_per_label(self, label, threshold=0.5):
        f = open(f"{self.imageset_path}/{label}_test.txt")
        lines = f.readlines()
        f.close()
        imagenames = [name.strip().split(' ')[0] for name in lines]

        # get gt objects
        class_recs = {}
        npos = 0
        for i, imagename in enumerate(imagenames):
            recs = self.parse_rec(f"{imagename}.xml")
            R = [obj for obj in recs if obj['name'] == label]
            box = np.array([x['box'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'box': box,
                                     'difficult': difficult,
                                     'det': det}

        # read detections
        f = open(f'{self.detect_path}/{label}.txt')
        lines = f.readlines()
        f.close()
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        box_pred = np.array([[float(z) for z in x[2:]] for x in splitlines])

        sorted_idx = np.argsort(-confidence)
        box_pred = box_pred[sorted_idx, :]
        image_ids = [image_ids[x] for x in sorted_idx]

        n_detects = len(image_ids)
        tp = np.zeros(n_detects)
        fp = np.zeros(n_detects)
        for i in range(n_detects):
            R = class_recs[image_ids[i]]
            box = box_pred[i, :]
            real_box = R['box'].astype(float)
            ovmax = -np.inf
            jmax = -1
            if real_box.size > 0:
                ixmin = np.maximum(real_box[:, 0], box[0])
                iymin = np.maximum(real_box[:, 1], box[1])
                ixmax = np.minimum(real_box[:, 2], box[2])
                iymax = np.minimum(real_box[:, 3], box[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                       (real_box[:, 2] - real_box[:, 0] + 1.) *
                       (real_box[:, 3] - real_box[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > threshold:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[i] = 1
                        R['det'][jmax] = 1.
                    else:
                        fp[i] = 1.
            else:
                fp[i] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / npos
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec)
        return rec, prec, ap


if __name__ == '__main__':
    eval = VOCMetric()
    map = eval.evaluate()