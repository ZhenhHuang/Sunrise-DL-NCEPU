from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms as transforms
from xml.etree.ElementTree import parse as ET_parse
import numpy as np
from PIL import Image
import os
from xml.etree.ElementTree import Element
import matplotlib.pyplot as plt
import json


class VOCDataset(Dataset):
    def __init__(self, root_path=r"C:\Users\98311\Downloads\VOCtrainval_06-Nov-2007\VOCdevkit",
                 year=2007, flag="train", transform=None, json_file='../datasets/PascalVOC2007/pascal_classes_2007.json'):
        super(VOCDataset, self).__init__()
        assert flag in ['train', 'val']
        self.root_path = f"{root_path}\VOC{year}"
        self.annotation_path = f"{self.root_path}/Annotations"
        self.image_path = f"{self.root_path}/JPEGImages"
        self.flag = flag
        self.split_path = f"{self.root_path}/ImageSets/Main/{flag}.txt"
        self.xml_list, self.image_list = self._get_xml_list()
        f = open(json_file)
        self.map_dict = json.load(f)
        f.close()
        self.transform = transform

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert("RGB")
        label = self.get_xml_dict(ET_parse(self.xml_list[index]).getroot())
        boxes = []
        labels = []
        iscrowd = []
        for obj in label['annotation']['object']:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(self.map_dict[obj['name']]))

        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def _get_xml_list(self):
        xml_list = []
        img_list = []
        with open(self.split_path) as f:
            for line in f.readlines():
                xml_path = f"{self.annotation_path}/{line.strip()}.xml"
                img_path = f"{self.image_path}/{line.strip()}.jpg"
                if os.path.exists(xml_path):
                    xml_list.append(xml_path)
                if os.path.exists(img_path):
                    img_list.append(img_path)
        f.close()
        assert len(xml_list) == len(img_list), "The length between targets and images are not same."
        return xml_list, img_list

    @staticmethod
    def get_xml_dict(xml: Element):
        if len(xml) == 0:
            return {xml.tag: xml.text}
        xml_dict = {}
        for node in xml:
            node_dict = VOCDataset.get_xml_dict(node)
            if node.tag != 'object':
                xml_dict[node.tag] = node_dict[node.tag]
            else:
                if node.tag not in xml_dict.keys():
                    xml_dict[node.tag] = []
                xml_dict[node.tag].append(node_dict[node.tag])
        return {xml.tag: xml_dict}


if __name__ == '__main__':
    import transforms as t
    transform = t.Compose([t.ToTensor(), t.RandomHorizontalFlip(prob=0.)])
    dataset = VOCDataset(transform=transform)
    index = 9
    img, target = dataset[index]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for box in target['boxes']:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='r')
        ax.add_patch(rect)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()










