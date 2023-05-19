import pickle
from pathlib import Path
import os 
import numpy as np

import cv2

        # {'height': [array(24.), array(24.), array(24.)],
            # 'label': [array(1.), array(5.), array(1.)],
            # 'left': [array(143.), array(156.), array(167.)],
            # 'top': [array(17.), array(17.), array(20.)],
            # 'width': [array(14.), array(14.), array(13.)]}

class DataConverter:

    def __init__(self, digit_struct_path):
        # 'data/train/digit_struct.pickle'
        with  open(digit_struct_path, 'rb') as handle:
            digit_struct = pickle.load(handle)

        
        self.read_path_prefix = os.path.dirname(digit_struct_path)
        self.name = digit_struct['digitStruct']['name']
        self.bbox_ = digit_struct['digitStruct']['bbox']
        self.bbox = []
        self.label = []
        self.length = []

        self.name_resized = []
        self.bbox_resized = []

        self.collect_data()

        # resize image, resize bbox
        # padding y_labels (10 for none) and bbox_labels (0, 64, 0, 64 for none)

        # length : (batch_size, 1)
        # img : (batch_size, 3, 64, 64)
        # y_labels: (batch_size, 5)
        # bbox_labels: (batch_size, 5, 4)

    def collect_data(self):
        
        N = len(self.name)
        for i in range(N):
            height, label, left, top, width = self.bbox_[i]['height'], self.bbox_[i]['label'], self.bbox_[i]['left'], self.bbox_[i]['top'], self.bbox_[i]['width']
            bbox_list = []
            label_list = []
            if not isinstance(label, list):
                height, label, left, top, width  = [height], [label], [left], [top], [width] 
            for j in range(len(label)):
                xmin = left[j]
                xmax = left[j] + width[j]
                ymin = top[j]
                ymax = top[j] + height[j]
                bbox_list.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                label_list.append(label[j] if label[j] != 10 else 0)
            self.bbox.append(bbox_list)
            self.label.append(label_list)
            self.length.append(len(label_list))

    def read_image(self, path):
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    def create_bbox_mask(self, bbox, img):

        rows, cols, _ = img.shape
        mask = np.zeros((rows, cols))
        mask[bbox[0]:bbox[2], bbox[1]: bbox[3]] = 1.
        return mask
    
    def mask_to_bbox(self, mask):
        cols, rows = np.nonzero(mask)
        if len(cols) == 0:
            return np.zeros(4, dtype=np.float32)
        
        xmin = np.min(cols)
        ymin = np.min(rows)
        xmax = np.max(cols)
        ymax = np.max(rows)

        return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    
    def resize_image_and_bboxes(self, write_path, size):

        read_path_prefix = Path(self.read_path_prefix)

        N = len(self.name)
        for i in range(N):
            read_path = read_path_prefix.joinpath(self.name[i])
            img = self.read_image(read_path) 
            img_resized = cv2.resize(img, (int(1.5*size), size))
            resized_bboxes = []
            for bbox in self.bbox[i]:
                mask_resized = cv2.resize(self.create_bbox_mask(bbox, img), (int(1.5*size), size))
                bbox_resized = self.mask_to_bbox(mask_resized)
                resized_bboxes.append(bbox_resized)
            self.bbox_resized.append(resized_bboxes)

            new_path = write_path + '{}'.format(self.name[i])
            cv2.imwrite(new_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            self.name_resized.append(new_path)

            if i % 1000 == 0:
                print('processed: {}'.format(i))
        


if __name__ == '__main__':
    converter = DataConverter('data/test/digit_struct.pickle')
    converter.resize_image_and_bboxes(write_path='data/test_resized/', size=64)
    with open('data/test_dataconverter.pickle', 'wb') as handle:
        pickle.dump(converter, handle, protocol=pickle.HIGHEST_PROTOCOL)


                





    
