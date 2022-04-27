import cv2
import numpy as np
import torch
from torchvision.datasets import CocoDetection
from torchvision.models import detection

# TODO: evaluate on COCO val2017

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = model.to(device).eval()

    for i in range(len(coco)):
        image, data = coco[i]
        image = np.asarray(image)
        x = image.astype(np.float32).transpose(2, 0, 1) / 255
        x = torch.from_numpy(x).to(device).unsqueeze(0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        view = image.copy()

        with torch.no_grad():
            out = model(x)[0]
            labels = out['labels']
            scores = out['scores']
            boxes = out['boxes']
            for k in range(len(scores)):
                score = scores[k]
                if score < 0.5:
                    break
                label = labels[k].cpu().detach().numpy()
                bbox = boxes[k].cpu().detach().numpy().astype(int)
                p0 = (bbox[0], bbox[1])
                p1 = (bbox[2], bbox[3])
                tag = '%d %s' % (label, COCO_INSTANCE_CATEGORY_NAMES[label])
                cv2.putText(view, tag,  p0, 0, 0.5, (0, 255, 255), 1)
                cv2.rectangle(view, p0, p1, (0, 255, 255), 1)

        for d in data:
            label = d['category_id']
            bbox = list(map(int, d['bbox']))
            p = (bbox[0], bbox[1])
            tag = '%d %s' % (label, COCO_INSTANCE_CATEGORY_NAMES[label])
            cv2.putText(image, tag,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, bbox, (0, 255, 0), 1)

        cv2.imshow('image', image)
        cv2.imshow('view', view)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
