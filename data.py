import cv2
import numpy as np
from torchvision.datasets import CocoDetection

# TODO: evaluate on COCO val2017


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
            cv2.putText(image, '%d' % cat_id,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, bbox, (0, 255, 0), 1)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            break


if __name__ == '__main__':
    test()
