import numpy as np
import torch
import torch.nn as nn
import math
import json
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST, FC_SYN, ResNet50
from options import args_parser
import copy
from data_loader_synthetic import get_dataset_synthetic
from data_loader_medical import get_dataset_medical
import wandb
from svd_tools import initialize_grad_len
import torchvision.models as torch_models
torch.set_num_threads(4)

Local_loss_all = []
Distribution_shift_loss_all = []
Aggregation_loss_all = []
Mean_loss_all = []
Acc_all = []

if __name__ == '__main__':
    args = args_parser()
    
    if 'synthetic' in args.dataset:
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset_synthetic(args)
    elif args.dataset == 'retina' or args.dataset == 'covid_fl':
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset_medical(args)
    else:
        train_loader, test_loader, global_train_loader, global_test_loader = get_dataset(args)
    
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # construct model
    if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(args.device)
        # args.lr = 0.02
    elif args.dataset == 'fmnist':
        global_model = CNN_FMNIST().to(args.device)
    elif args.dataset == 'emnist':
        args.num_classes = 62
        global_model = CNN_FMNIST(num_classes=args.num_classes).to(args.device)
    elif args.dataset in ['synthetic_iid', 'synthetic_0_0', 'synthetic_0.5_0.5', 'synthetic_1_1']:
        args.num_classes = 10
        args.num_users = 30
        global_model = FC_SYN().to(args.device)
    elif args.dataset == 'retina':
        args.num_classes = 2
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(args.device)
    elif args.dataset == 'covid_fl':
        args.num_classes = 3
        args.num_users = 12
        global_model = ResNet50()
        pretrained_resnet50 = torch_models.resnet50(pretrained=True)
        global_model.load_state_dict(pretrained_resnet50.state_dict())
        global_model.fc = nn.Linear(2048, args.num_classes)
        global_model.to(args.device)
    else:
        raise NotImplementedError()
    
    print(args)
    wandb.init(project="Cifar", name='FedGH_05', config=args)
       
    # Set Training Rule
    LocalUpdate = local_update(args.train_rule)
    # Choose One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)
    
    #====================================== Set the local clients =======================================#
    local_clients = []
    for idx in range(args.num_users):
        train_loader_idx = torch.randint(1, 5, (1,)).item()
        if args.dataset == 'covid_fl' or args.dataset == 'retina':
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader,
                                             model=copy.deepcopy(global_model)))
        else:
            local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader, 
                                             model=copy.deepcopy(global_model)))
            
    #==================================== Set global test dataset =======================================#
    global_test_dataset = []
    if args.dataset == 'covid_fl' or args.dataset == 'retina':
        for idx in range(args.num_users):
            global_test_dataset.append(test_loader)
    else:
        for idx in range(args.num_users):
            global_test_dataset.append(test_loader[idx])
            
    #======================================= Local Training =========================================#

    if args.train_rule == 'FedLD' or args.train_rule == 'FedGH':
        # initialize grad_history and grad_len
        grad_history = {}
        for k in global_model.state_dict().keys():
            grad_history[k] = None
        grad_len = initialize_grad_len(global_model, grad_history)

        grad_history['grad_len'] = grad_len

        for round in range(args.epochs):
            loss4, global_acc = train_round_parallel(args, global_model, local_clients, round, global_train_loader, global_test_dataset, grad_history)
            Mean_loss_all.append(loss4)
            Acc_all.append(global_acc)
            wandb.log({'Mean Loss': loss4, 'Global Accuracy': global_acc}, step=round)
            
    else:
        for round in range(args.epochs):
            loss4, global_acc = train_round_parallel(args, global_model, local_clients, round, global_train_loader, global_test_dataset)

            Mean_loss_all.append(loss4)
            Acc_all.append(global_acc)
            wandb.log({'Mean Loss': loss4, 'Global Accuracy': global_acc}, step=round)
            
    #======================================= Record the Result ======================================#
