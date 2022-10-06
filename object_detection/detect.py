import torch
from utils import target_decode
from object_detection.data.data_factory import data_factory
from object_detection.data.data_loader import VOCDataset
from tools import visualize
from torchvision.transforms import Normalize


def detect(args, model, device):
    print("----------loading test set--------")
    _, test_loader = data_factory(args, flag='test')
    test_set = VOCDataset(flag='test')
    model.load_state_dict(torch.load(f"./checkpoints/{args.model_path}"))
    model.eval()
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image = image.float().to(device)
            target = target.float().to(device)
            output = model(image).cpu()
            box_corner, class_label, confidence, class_score = \
                target_decode(target.cpu().squeeze(0), args.threshold, args.S, args.B)
            box_corner = box_corner * args.size[0]
            image = image.cpu().squeeze(0)
            class_label = class_label.cpu().numpy()
            map_dict = {v:k for k, v in test_set.map_dict.items()}
            class_label = [map_dict[i] for i in class_label]
            # if i % 20 == 0:
            visualize(image.cpu().squeeze(0), box_corner.numpy(), class_label, confidence.cpu().numpy(), i)


