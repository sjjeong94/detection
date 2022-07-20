import cv2
import torch
import numpy as np

import datasets
import transforms


def test_dataset():
    mode = 'val'
    if mode == 'train':
        root = '../data/coco/train2017'
        annFile = '../data/coco/annotations/instances_train2017.json'
    else:
        root = '../data/coco/val2017'
        annFile = '../data/coco/annotations/instances_val2017.json'

    dataset = datasets.CocoDetection(root, annFile)

    idx = 0
    while True:
        image, target = dataset[idx]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for t in target:
            cat_id = t['category_id']
            bbox = list(map(int, t['bbox']))
            p = (bbox[0], bbox[1])
            label = '%d %s' % (cat_id, dataset.category_names[cat_id])
            cv2.putText(image, label,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, bbox, (0, 255, 0), 1)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0

    cv2.destroyAllWindows()


def test_dataset2():
    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0, 0.5), (0.5, 0.5, 1))
    ])

    root = '../data/coco/val2017'
    annFile = '../data/coco/annotations/instances_val2017.json'
    dataset = datasets.CocoDetection(root, annFile, transform=T_compose)

    image, label = dataset[5]
    print(image.shape, image.dtype, image.max(), image.min())
    print(image[0].min(), image[0].max())
    print(image[1].min(), image[1].max())
    print(image[2].min(), image[2].max())
    print()


def visualize_gt(gt, scale=255):
    views = []
    for view in gt:
        view = np.clip(view * scale, 0, 255)
        view = view.astype(np.uint8)
        view[:, 0] = 64
        views.append(view)
    views = np.concatenate(views, 1)
    return views


def visualize_gt2(gt, scale=255):
    views = []
    for view in gt:
        view = np.clip(view * scale, 0, 255)
        view = view.astype(np.uint8)
        view[:, 0] = 64
        view[0, :] = 64
        views.append(view)
    views = np.reshape(views, (10, 9 * view.shape[0], view.shape[1]))
    views = np.concatenate(views, 1)
    return views


def test_encode_decode():
    mode = 'val'
    if mode == 'train':
        root = '../data/coco/train2017'
        annFile = '../data/coco/annotations/instances_train2017.json'
    else:
        root = '../data/coco/val2017'
        annFile = '../data/coco/annotations/instances_val2017.json'

    T_compose = transforms.Compose([
        transforms.RandomResize(320, 480),
        transforms.RandomCrop(320),
        transforms.RandomHorizontalFlip(),
    ])

    dataset = datasets.CocoDetection(root, annFile, T_compose)

    print(len(dataset.category_names))

    idx = 0
    while True:
        image, target = dataset[idx]
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = image1.copy()
        for t in target:
            cat_id = t['category_id']
            bbox = list(map(int, t['bbox']))
            p = (bbox[0], bbox[1])
            label = '%d %s' % (cat_id, dataset.category_names[cat_id])
            cv2.putText(image1, label,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image1, bbox, (0, 255, 0), 1)
            cv2.rectangle(image2, bbox, (0, 255, 0), 1)

        encoded = transforms.label_encode(image, target)
        decoded = transforms.label_decode(encoded)

        reg = visualize_gt(encoded[:4], scale=1)
        cv2.imshow('gt_reg', reg)
        cen = visualize_gt(encoded[4:5])
        cv2.imshow('gt_centerness', cen)
        cls = visualize_gt2(encoded[5:])
        cv2.imshow('gt_cls', cls)

        for t in decoded:
            cat_id = t['category_id']
            bbox = list(map(int, t['bbox']))
            p = (bbox[0], bbox[1])
            label = '%d %s' % (cat_id, dataset.category_names[cat_id])
            cv2.putText(image2, label,  p, 0, 0.5, (0, 0, 255), 1)
            cv2.rectangle(image2, bbox, (0, 0, 255), 1)

        cv2.imshow('image1', image1)
        cv2.imshow('image2', image2)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0

    cv2.destroyAllWindows()


def test_transform():
    mode = 'val'
    if mode == 'train':
        root = '../data/coco/train2017'
        annFile = '../data/coco/annotations/instances_train2017.json'
    else:
        root = '../data/coco/val2017'
        annFile = '../data/coco/annotations/instances_val2017.json'

    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CocoDetection(root, annFile, T_compose)

    idx = 0
    while True:
        image, target = dataset[idx]

        image = (image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        decoded = transforms.label_decode(target.numpy())

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for t in decoded:
            cat_id = t['category_id']
            bbox = list(map(int, t['bbox']))
            p = (bbox[0], bbox[1])
            label = '%d %s' % (cat_id, dataset.category_names[cat_id])
            cv2.putText(image, label,  p, 0, 0.5, (0, 255, 0), 1)
            cv2.rectangle(image, bbox, (0, 255, 0), 1)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0

    cv2.destroyAllWindows()


def test_dataloader():
    mode = 'val'
    if mode == 'train':
        root = '../data/coco/train2017'
        annFile = '../data/coco/annotations/instances_train2017.json'
    else:
        root = '../data/coco/val2017'
        annFile = '../data/coco/annotations/instances_val2017.json'

    T_compose = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CocoDetection(root, annFile, T_compose)

    batch_size = 8

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    for x, y in loader:
        break

    print('DataLoader Test')
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)


if __name__ == '__main__':
    # test_dataset()
    # test_dataset2()
    test_encode_decode()
    # test_transform()
    # test_dataloader()
