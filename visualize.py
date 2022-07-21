import cv2
import torch
import torchvision

import tests
import models
import datasets
import transforms


class Module:
    def __init__(self, model_path):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = models.CenterNet()

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        net = net.to(device)
        net = net.eval()

        T_compose = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        self.device = device
        self.net = net
        self.transform = T_compose

    @torch.inference_mode()
    def __call__(self, image):
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)
        out = self.net(x)
        print(out.shape)
        out[:, 4:] = torch.sigmoid(out[:, 4:])
        print(out[:, 4:].max(), out[:, 4:].min())
        out = out.cpu().numpy().squeeze()
        return out


def visualze_eval(
    model_path='./logs/coco/fcos2/models/model_010.pt',
    size=(320, 320),
):

    module = Module(model_path)

    root = '../data/coco/val2017'
    annFile = '../data/coco/annotations/instances_val2017.json'
    dataset = datasets.CocoDetection(
        root,
        annFile,
        transforms.Resize(size)
    )

    idx = 0
    while True:
        image, label = dataset[idx]
        print(len(label))

        out = module(image)

        reg = tests.visualize_gt(out[:4], scale=1)
        cv2.imshow('out_reg', reg)
        cen = tests.visualize_gt(out[4:5])
        cv2.imshow('out_cen', cen)
        cls = tests.visualize_gt2(out[5:])
        cv2.imshow('out_cls', cls)

        decoded = transforms.label_decode(out)

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


if __name__ == '__main__':
    visualze_eval()
