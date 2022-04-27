import cv2
import numpy as np
from torchvision.datasets import CocoDetection

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def test():
    mode = 'val'
    if mode == 'train':
        root = '../data/coco/train2017'
        annFile = '../data/coco/annotations/instances_train2017.json'
    else:
        root = '../data/coco/val2017'
        annFile = '../data/coco/annotations/instances_val2017.json'

    coco = CocoDetection(root, annFile)

    for i in range(len(coco)):
        image, data = coco[i]
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for d in data:
            cat_id = d['category_id']
            bbox = list(map(int, d['bbox']))
            p = (bbox[0], bbox[1])
            label = '%d %s' % (cat_id, COCO_INSTANCE_CATEGORY_NAMES[cat_id])
            cv2.putText(image, label,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, bbox, (0, 255, 0), 1)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    test()
