import torch
import tools
import numpy as np
import copy
from tools import average_weights_weighted, get_head_agg_weight, agg_classifier_weighted_p
import wandb
from svd_tools import pcgrad_svd, get_grads_, set_grads_, pcgrad_hierarchy
from options import args_parser
from tqdm import tqdm


args = args_parser()

def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg':train_round_fedavg,
                   'LG_FedAvg':train_round_lgfedavg,
                   'FedPer':train_round_fedper,
                   'Local':train_round_standalone,
                   'FedPAC':train_round_fedpac,
                   'FedProx':train_round_fedavg,
                   'FedLD': train_round_fedld,
                   'FedGH': train_round_fedgh,
    }
    return Train_Round[rule]

## training methods -------------------------------------------------------------------
# local training only
def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


# vanila FedAvg
def train_round_fedavg(args, global_model, local_clients, rnd, global_train_loader, global_test_dataset, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    global_acc = [] 
    agg_weight = []
    local_losses3 = []
    global_weight = global_model.state_dict()
    
    
    #============================ Begin local training ==============================#
    print('\n------ Local training begins ------')
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, loss3 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_losses3.append(loss3)
    print('\n------ Local training ends ------')
        

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    loss_mean = sum(local_losses3) / len(local_losses3)
    
        
    print('\n------ Calculation of the global accuracy begins ------')
    #============= Calculate Global Accuracy ==================#
    for data_loader in global_test_dataset:
        global_model.eval()
        device = args.device
        correct = 0
        lenth = len(data_loader.dataset)
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            acc = 100.0 * correct / lenth
            global_acc.append(acc)
            
    return loss_mean.item(), sum(global_acc)/len(global_acc)


# parameter decoupling
def train_round_lgfedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # global_weight = average_weights(local_weights)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_fedper(args, global_model, local_clients, rnd, global_train_loader, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []
    local_losses3 = []

    global_weight = global_model.state_dict()
    
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        agg_weight.append(local_client.agg_weight)
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2, loss3 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)
        local_losses3.append(copy.deepcopy(loss3))

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)
    loss_mean = sum(local_losses3) / len(local_losses3)
    
    # calculate Local loss, Distribution shift loss, and Aggregation loss
    local_losses, distribution_shift_losses, aggregation_losses = [], [], []
    loss_weight = [agg_weight[i] / sum(agg_weight) for i in range(len(agg_weight))]
    # print(loss_weight, sum(loss_weight))
    temp_server = copy.deepcopy(local_clients[0])
    temp_server.update_local_model(global_weight=global_weight)
    all_data_loss = temp_server.get_training_loss(global_train_loader)
    for idx in idx_users:
        local_client = local_clients[idx]
        local_loss = local_client.get_training_loss(local_client.train_data)
        local_losses.append(local_loss)
        for idx_2 in idx_users:
            local_loss_2 = local_client.get_training_loss(local_clients[idx_2].train_data)
            dist_shift_loss = (local_loss_2 - local_loss) * loss_weight[idx_2]
            distribution_shift_losses.append(dist_shift_loss)
             
        all_data_loss_idx = local_client.get_training_loss(global_train_loader)
        agg_loss = all_data_loss - all_data_loss_idx
        # agg_loss = all_data_loss - all_data_loss_idx / 2
        aggregation_losses.append(agg_loss)
         
    Local_loss = sum([local_losses[i] * loss_weight[i] for i in range(len(loss_weight))])
    # Local_loss = sum([local_losses[i] * loss_weight[i] for i in range(len(loss_weight))]) / 2
    Distribution_shift_loss = sum([distribution_shift_losses[i] * loss_weight[i] for i in range(len(loss_weight))])
    # Distribution_shift_loss = sum([distribution_shift_losses[i] * loss_weight[i] for i in range(len(loss_weight))]) / 2
    Aggretation_loss = sum([aggregation_losses[i] * loss_weight[i] for i in range(len(loss_weight))])
    print(f'Local Loss: {Local_loss}, Distribution Shift Loss: {Distribution_shift_loss}, Aggregation Loss: {Aggretation_loss}')
    wandb.log({"Local Loss": Local_loss, "Distribution Shift Loss": Distribution_shift_loss, "Aggregation Loss": Aggretation_loss}, step=rnd)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, loss_mean


def train_round_fedpac(args, global_model, local_clients, rnd, **kwargs):
        print(f'\n---- Global Communication Round : {rnd+1} ----')
        num_users = args.num_users
        m = max(int(args.frac * num_users), 1)
        if (rnd >= args.epochs):
            m = num_users
        idx_users = np.random.choice(range(num_users), m, replace=False)
        idx_users = sorted(idx_users)

        local_weights, local_losses1, local_losses2 = [], [], []
        local_acc1 = []
        local_acc2 = []
        agg_weight = []  # aggregation weights for f
        avg_weight = []  # aggregation weights for g
        sizes_label = []
        local_protos = []

        Vars = []
        Hs = []

        agg_g = args.agg_g # conduct classifier aggregation or not

        if rnd <= args.epochs:
            for idx in idx_users:
                local_client = local_clients[idx]
                ## statistics collection
                v, h = local_client.statistics_extraction()
                Vars.append(copy.deepcopy(v))
                Hs.append(copy.deepcopy(h))
                ## local training
                local_epoch = args.local_epoch
                sizes_label.append(local_client.sizes_label)
                w, loss1, loss2, acc1, acc2, protos = local_client.local_training(local_epoch=local_epoch, round=rnd)
                local_weights.append(copy.deepcopy(w))
                local_losses1.append(copy.deepcopy(loss1))
                local_losses2.append(copy.deepcopy(loss2))
                local_acc1.append(acc1)
                local_acc2.append(acc2)
                agg_weight.append(local_client.agg_weight)
                local_protos.append(copy.deepcopy(protos))

            # get weight for feature extractor aggregation
            agg_weight = torch.stack(agg_weight).to(args.device)

            # update global feature extractor
            global_weight_new = average_weights_weighted(local_weights, agg_weight)

            # update global prototype
            global_protos = tools.protos_aggregation(local_protos, sizes_label)

            for idx in range(num_users):
                local_client = local_clients[idx]
                local_client.update_base_model(global_weight=global_weight_new)
                local_client.update_global_protos(global_protos=global_protos)

            # get weight for local classifier aggregation
            if agg_g and rnd < args.epochs:
                avg_weights = get_head_agg_weight(m, Vars, Hs)
                idxx = 0
                for idx in idx_users:
                    local_client = local_clients[idx]
                    if avg_weights[idxx] is not None:
                        new_cls = agg_classifier_weighted_p(local_weights, avg_weights[idxx], local_client.w_local_keys, idxx)
                    else:
                        new_cls = local_weights[idxx]
                    local_client.update_local_classifier(new_weight=new_cls)
                    idxx += 1

        loss_avg1 = sum(local_losses1) / len(local_losses1)
        loss_avg2 = sum(local_losses2) / len(local_losses2)
        acc_avg1 = sum(local_acc1) / len(local_acc1)
        acc_avg2 = sum(local_acc2) / len(local_acc2)

        return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_fedld(args, global_model, local_clients, rnd, global_train_loader, grad_history, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []
    local_losses3 = []

    global_weight = global_model.state_dict()
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2, loss3 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        local_acc1.append(acc1)
        local_acc2.append(acc2)
        local_losses3.append(loss3)
    
    # without SVD
    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)
    
    # below for SVD
    # # get global weights
    # local_clients_grads = []
    # local_weights_new = []
    # for idx in idx_users:
    #     local_clients_grads.append(get_grads_(local_clients[idx].local_model, global_model))
    # grad_new, grad_history = pcgrad_svd(num_users, local_clients_grads, grad_history)
    # for idx in idx_users:
    #     local_clients[idx].local_model = set_grads_(local_clients[idx].local_model, global_model, grad_new)
    # for idx in idx_users:
    #     local_weights_new.append(copy.deepcopy(local_clients[idx].local_model.state_dict()))
    # global_weight = average_weights_weighted(local_weights_new, agg_weight)
    # global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)
    loss_mean = sum(local_losses3) / len(local_losses3)

    # calculate Local loss, Distribution shift loss, and Aggregation loss
    local_losses, distribution_shift_losses, aggregation_losses = [], [], []
    loss_weight = [agg_weight[i] / sum(agg_weight) for i in range(len(agg_weight))]
    # print(loss_weight, sum(loss_weight))
    temp_server = copy.deepcopy(local_clients[0])
    temp_server.update_local_model(global_weight=global_weight)
    all_data_loss = temp_server.get_training_loss(global_train_loader)
    for idx in idx_users:
        local_client = local_clients[idx]
        local_loss = local_client.get_training_loss(local_client.train_data)
        local_losses.append(local_loss)
        for idx_2 in idx_users:
            local_loss_2 = local_client.get_training_loss(local_clients[idx_2].train_data)
            dist_shift_loss = (local_loss_2 - local_loss) * loss_weight[idx_2]
            distribution_shift_losses.append(dist_shift_loss)
             
        all_data_loss_idx = local_client.get_training_loss(global_train_loader)
        agg_loss = all_data_loss - all_data_loss_idx
        aggregation_losses.append(agg_loss)
         
    Local_loss = sum([local_losses[i] * loss_weight[i] for i in range(len(loss_weight))])
    Distribution_shift_loss = torch.abs(sum([distribution_shift_losses[i] * loss_weight[i] for i in range(len(loss_weight))]))
    Aggretation_loss = torch.abs(sum([aggregation_losses[i] * loss_weight[i] for i in range(len(loss_weight))]))
    print(f'Local Loss: {Local_loss}, Distribution Shift Loss: {Distribution_shift_loss}, Aggregation Loss: {Aggretation_loss}')
    wandb.log({"Local Loss": Local_loss, "Distribution Shift Loss": Distribution_shift_loss, "Aggregation Loss": Aggretation_loss}, step=rnd)
    
    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, loss_mean

def train_round_fedgh(args, global_model, local_clients, rnd, global_train_loader, global_test_dataset, grad_history, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2, local_losses3 = [], [], [], []
    agg_weight = []
    global_acc = []

    global_weight = global_model.state_dict()

    #============================= Local Training Begins ===========================#
    for idx in tqdm(idx_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, loss3 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_losses3.append(copy.deepcopy(loss3))
    #=============================== Local Training Ends ============================#
        
    # get global weights
    local_clients_grads = []
    local_weights_new = []
    for idx in idx_users:
        local_clients_grads.append(get_grads_(local_clients[idx].local_model, global_model))
    grad_new, grad_history = pcgrad_hierarchy(num_users, local_clients_grads, grad_history)
    for idx in idx_users:
        local_clients[idx].local_model = set_grads_(local_clients[idx].local_model, global_model, grad_new)
    for idx in idx_users:
        local_weights_new.append(copy.deepcopy(local_clients[idx].local_model.state_dict()))
    global_weight = average_weights_weighted(local_weights_new, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)
        
    print('\n------ Calculation of the global accuracy begins -------')
    #========================== Calculate Global Accuracy =======================#
    for data_loader in global_test_dataset:
        global_model.eval()
        device = args.device
        correct = 0
        lenth = len(data_loader.dataset)
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = global_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            acc = 100.0 * correct / lenth
            global_acc.append(acc)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    loss_mean = sum(local_losses3) / len(local_losses3)

    # return loss_avg1, loss_avg2, acc_avg1, acc_avg2, loss_mean
    return loss_mean.item(), sum(global_acc)/len(global_acc)
