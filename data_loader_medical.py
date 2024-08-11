from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
from PIL import Image
from skimage.transform import resize


class DatasetCOVIDFL(Dataset):

    def __init__(self, file_path, phase, transform):
        super(DatasetCOVIDFL, self).__init__()

        self.img_paths = list({line.strip().split(',')[0] for line in open(file_path)})
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open('/home/Data1/Medical-Public/COVID-FL/labels.csv')}

        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join('/home/Data1/Medical-Public/COVID-FL', self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        img = np.array(Image.open(path).convert("RGB"))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetRetina(Dataset):

    def __init__(self, file_path, phase, transform):
        super(DatasetRetina, self).__init__()

        self.img_paths = list({line.strip().split(',')[0] for line in open(file_path)})
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open('/home/Data1/Medical-Public/Retina/labels.csv')}

        self.transform = transform
        self.phase = phase

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join('/home/Data1/Medical-Public/Retina', self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        img = np.load(path)
        img = resize(img, (256, 256))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, index=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in index] if index is not None else [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_dataset_medical(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    train_dataset = []
    test_dataset = []
    train_loader = []
    test_loader = []
    global_train_loader = []
    global_test_loader = []

    ## --------------------------------------------------------------------------------------------------------
    ## data allocation
    if args.dataset == 'covid_fl':
        split_folders = os.listdir('/home/Data1/Medical-Public/COVID-FL/12_clients/split_real/')
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.8, 1.2)),#224-384
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])
        for split_id in split_folders:
            cur_client_path = '/home/Data1/Medical-Public/COVID-FL/12_clients/split_real/' + split_id
            cur_train_dataset = DatasetCOVIDFL(file_path=cur_client_path, phase='train', transform=transform_train)
            cur_train_loader = DataLoader(cur_train_dataset, batch_size=args.local_bs, shuffle=True)
            train_loader.append(cur_train_loader)
        
        # for COVID-FL, test_loader = global_test_loader
        test_file_path = '/home/Data1/Medical-Public/COVID-FL/test.csv'
        transform_test = transforms.Compose([
            transforms.Resize(size=384),#224-384
            transforms.CenterCrop(size=(384, 384)),#224-384
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])
        test_dataset = DatasetCOVIDFL(file_path=test_file_path, phase='test', transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        
        # add global train loader
        train_file_path = '/home/Data1/Medical-Public/COVID-FL/train.csv'
        train_dataset = DatasetCOVIDFL(file_path=train_file_path, phase='train', transform=transform_train)
        global_train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)

    elif args.dataset == 'retina':
        split_folders = os.listdir('/home/Data1/Medical-Public/Retina/5_clients/split_'+args.retina_split+'/')
        mean, std = (0.5007, 0.5010, 0.5019), (0.0342, 0.0535, 0.0484)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.6, 1.0)),#224-384
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])
        for split_id in split_folders:
            cur_client_path = '/home/Data1/Medical-Public/Retina/5_clients/split_'+args.retina_split+'/' + split_id
            cur_train_dataset = DatasetRetina(file_path=cur_client_path, phase='train', transform=train_transform)
            cur_train_loader = DataLoader(cur_train_dataset, batch_size=args.local_bs, shuffle=True)
            train_loader.append(cur_train_loader)

        # for Retina, test_loader = global_test_loader
        test_file_path = '/home/Data1/Medical-Public/Retina/test.csv'
        test_transform = transforms.Compose([
            transforms.Resize(size=384),#224-384
            transforms.CenterCrop(size=(384, 384)),#224-384
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        ])
        test_dataset = DatasetRetina(file_path=test_file_path, phase='test', transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False)
        
        # add global train loader
        train_file_path = '/home/Data1/Medical-Public/Retina/train.csv'
        train_dataset = DatasetRetina(file_path=train_file_path, phase='train', transform=train_transform)
        global_train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)

    else:
        raise NotImplementedError()

    return train_loader, test_loader, global_train_loader, global_test_loader
