import json
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def batch_data_multiple_iters(data, batch_size, num_iters):
    data_x = data['x']
    data_y = data['y']

    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    idx = 0

    for i in range(num_iters):
        if idx+batch_size >= len(data_x):
            idx = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
        batched_x = data_x[idx: idx+batch_size]
        batched_y = data_y[idx: idx+batch_size]
        idx += batch_size
        yield (batched_x, batched_y)

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


class Syn_Dataset(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset):
        self.dataset = {k: np.array(v) for k, v in dataset.items()}

    def __len__(self):
        return len(self.dataset['y'])

    def __getitem__(self, item):
        image, label = self.dataset['x'][item], self.dataset['y'][item]
        image = image.astype(np.float32)
        label = label.astype(np.int64)
        return image, label


def get_dataset_synthetic(args):
    clients, _, train_data, test_data = read_data(f'./data/{args.dataset}/data/train/', f'./data/{args.dataset}/data/test/')
    train_loader = []
    test_loader = []
    global_train_loader = []
    global_test_loader = []
    for client in clients:
        train_data_client = train_data[client]
        test_data_client = test_data[client]
        loader1 = DataLoader(Syn_Dataset(train_data_client), batch_size=args.local_bs, shuffle=True)
        loader2 = DataLoader(Syn_Dataset(test_data_client), batch_size=args.local_bs, shuffle=False)
        train_loader.append(loader1)
        test_loader.append(loader2)
        
    all_train_data = ConcatDataset([Syn_Dataset(train_data[client]) for client in clients])
    all_test_data = ConcatDataset([Syn_Dataset(test_data[client]) for client in clients])
    global_train_loader = DataLoader(all_train_data, batch_size=args.local_bs, shuffle=False)
    global_test_loader = DataLoader(all_test_data, batch_size=args.local_bs, shuffle=False)
    return train_loader, test_loader, global_train_loader, global_test_loader
