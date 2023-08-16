from PIL import Image
import os
from os.path import abspath, expanduser
import torch
import numpy as np
import pathlib
import cv2
import pandas as pd
import copy


from typing import Any, Callable, List, Dict, Optional, Tuple, Union


class WIDERFace(torch.utils.data.Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, dataset_type: str = "train", balance_data: bool = False) -> None:
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index: int) -> Tuple[str, Any, Any, Any]:
        # print("loading data")
        image_info = self.data[index]
        # print("image_info: ", image_info)
        image = self._read_image(image_info['image_id'])
        # print("image: ", image.shape)
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        # given is xc, yc, w, h convert it into xmin, ymin, xmax, ymax
        boxes[:, 0] = boxes[:, 0]
        boxes[:, 1] = boxes[:, 1]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # print("boxes: ", boxes.shape)
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        # print(labels)
        # print("Im sdfgahjksbgasjkdgbfaksjb")
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        # print("===================fskgldkfgnsflxnvbld+++++++++++++++++")
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
            # print("boxes_after_processing: ", boxes.shape)
        # print(type(image_info['image_id']), type(
            # image), type(boxes), type(labels))
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image_path, image, boxes, labels = self._getitem(index)
        # print(image.min(), image.max(), image.shape)
        return image_path, image, boxes, labels

    def get_annotation(self, index: int) -> Tuple[str, Tuple[Any, Any, Any]]:
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index: int) -> Any:
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, int]]:
        data = []
        class_names = ['BACKGROUND', "Face"]
        class_dict = {class_name: i for i,
                      class_name in enumerate(class_names)}
        widerface_path = os.path.join(
            self.root)
        with open(os.path.join(widerface_path, 'wider_face_split', 'wider_face_{}_bbx_gt.txt'.format(self.dataset_type)), 'r') as f:
            lines = f.readlines()
            file_name_line, num_boxes_line, box_annotation_line = True, False, False
            num_boxes, box_counter = 0, 0
            labels = []
            type_ = []
            for line in lines:
                line = line.rstrip()
                # print("=============== line ===============")
                # print(line)
                if file_name_line:
                    # print("file_name_line")
                    img_path = os.path.join(
                        self.root, "WIDER_" + self.dataset_type, "images", line)
                    img_path = abspath(expanduser(img_path))
                    file_name_line = False
                    num_boxes_line = True
                elif num_boxes_line:
                    # print('num_boxes_line')
                    num_boxes = int(line)
                    num_boxes_line = False
                    box_annotation_line = True
                elif box_annotation_line:
                    # print('box_annotation_line')
                    box_counter += 1
                    line_split = line.split(" ")
                    line_values = [int(x) for x in line_split]
                    labels.append(line_values)
                    type_.append(class_dict[class_names[-1]])
                    # print(labels)
                    if box_counter >= num_boxes:
                        box_annotation_line = False
                        file_name_line = True
                        labels_tensor = torch.tensor(labels)
                        type_tensor = torch.tensor(type_)
                        data.append({
                            "image_id": img_path,
                            # x, y, width, height
                            "boxes": labels_tensor[:, 0:4].numpy(),
                            # 0 represents background
                            "labels": type_tensor.numpy(),
                        })
                        # print('data[-1] ', data[-1])
                        # print(box_counter)
                        box_counter = 0
                        labels.clear()
                        type_.clear()
        print("============= reading data =============")
        print("data length: ", len(data))
        print("class_names: ", class_names)
        print("class_dict: ", class_dict)
        return data, class_names, class_dict

    def _balance_data(self) -> List[Dict[str, Any]]:
        """Balance the data by duplicating images with fewer number of faces."""
        # get the number of faces in each image
        num_faces = [len(info['boxes']) for info in self.data]
        # get the minimum number of faces among all images
        self.min_image_num = min(num_faces)
        # get the index of images with the minimum number of faces
        min_image_idx = [i for i, num_face in enumerate(
            num_faces) if num_face == self.min_image_num]
        # duplicate images with the minimum number of faces
        data = []
        for i in range(len(self.data)):
            if i in min_image_idx:
                data.extend([self.data[i]] * (self.min_image_num - 1))
            else:
                data.append(self.data[i])
        return data

    def _read_image(self, image_id: str) -> Any:
        image_path = os.path.join(self.root, 'WIDER_{}'.format(
            self.dataset_type), 'images', image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self) -> int:
        return len(self.data)
