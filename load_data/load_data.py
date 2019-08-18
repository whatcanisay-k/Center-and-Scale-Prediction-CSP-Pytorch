import os
import numpy as np


def get_citypersons(root_dir='data/cityperson', type='train'):

    image_data = []
    image_set_path = os.path.join(root_dir, 'ImageSets', type +'.txt')
    fid_set = open(image_set_path)
    while True:
        image_name = fid_set.readline().replace('\n', '')
        if len(image_name) == 0:
            break
        fid_label = open(os.path.join(root_dir, 'Annotations', image_name[:-3]+'txt'))
        boxes = []
        while True:
            box_info = fid_label.readline().replace('\n', '')
            if len(box_info) == 0:
                break
            box_info = box_info.split(' ')
            box = [int(box_info[x]) for x in range(len(box_info))]
            boxes.append(box)

        annotation = {}
        annotation['filepath'] = os.path.join(root_dir, 'Images', image_name)
        annotation['bboxes'] = np.array(boxes)
        image_data.append(annotation)

    return image_data

