import object_detection.data.transforms as t
from torch.utils.data import DataLoader
from object_detection.data.data_loader import VOCDataset


def data_factory(args, flag):
    data_dict = {
        'VOC': VOCDataset
    }
    Data = data_dict[args.data]

    if flag == 'train':
        transform = t.Compose([t.ToTensor(),
                               t.Resize((args.size[0], args.size[1])),
                               t.RandomHorizontalFlip(prob=0.),
                               t.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif flag == 'val' or flag == 'test':
        transform = t.Compose([t.ToTensor(),
                               t.Resize((args.size[0], args.size[1])),
                               t.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        transform = t.Compose([t.ToTensor(), t.Resize((args.size[0], args.size[1]))])   # predict

    return_size = False
    year = args.trainval_year
    if flag == 'train':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True

    elif flag == 'val':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
        
    elif flag == 'test':
        batch_size = args.batch_size
        shuffle = True
        drop_last = True
        year = args.test_year

    else:
        # predict
        batch_size = 1
        shuffle = False
        drop_last = False
        flag = 'test'
        return_size = True
        year = args.test_year

    data_set = Data(**vars(args), transform=transform, flag=flag, return_size=return_size, year=year)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=args.num_workers)
    return data_set, data_loader
