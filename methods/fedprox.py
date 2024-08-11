import numpy as np
import tools
import math
import copy
import torch
from torch import nn
import time
# ---------------------------------------------------------------------------- #

class LocalUpdate_FedProx(object):
    def __init__(self, idx, args, train_set, test_set, model):
        self.idx = idx
        self.args = args
        self.train_data = train_set
        self.test_data = test_set
        self.device = args.device
        self.criterion = nn.CrossEntropyLoss()
        self.local_model = model
        self.local_model_finetune = copy.deepcopy(model)
        self.w_local_keys = self.local_model.classifier_weight_keys
        self.agg_weight = self.aggregate_weight()
        self.mu = 0.02

    def aggregate_weight(self):
        data_size = len(self.train_data.dataset)
        w = torch.tensor(data_size).to(self.device)
        return w
    
    def local_test(self, test_loader, test_model=None):
        model = self.local_model if test_model is None else test_model
        model.eval()
        device = self.device
        correct = 0
        total = len(test_loader.dataset)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = 100.0*correct/total
        return acc
    
    def update_local_model(self, global_weight):
        self.local_model.load_state_dict(global_weight)
    
    def local_training(self, local_epoch, round=0):
        model = self.local_model
        model.train()
        round_loss = 0
        iter_loss = []
        model.zero_grad()
        grad_accum = []
        global_model = copy.deepcopy(model)  # global model
        global_model.eval()

        w0 = tools.get_parameter_values(model)

        acc1 = self.local_test(self.test_data)

        # Set optimizer for the local updates, default sgd
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5, weight_decay=0.0005)
        for ep in range(local_epoch):
            data_loader = iter(self.train_data)
            iter_num = len(data_loader)
            for it in range(iter_num):
                images, labels = next(data_loader)
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                _, output = model(images)
                ce_loss = self.criterion(output, labels)
                proximal_term = 0.0
                # for param_local, param_global in zip(model.parameters(), global_model.parameters()):
                #     proximal_term += ((param_local - param_global) ** 2).sum()
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                # print(proximal_term)
                loss = ce_loss + self.mu / 2 * proximal_term
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        
        # loss value
        round_loss1 = iter_loss[0]
        round_loss2 = iter_loss[-1]
        acc2 = self.local_test(self.test_data)
        
        return model.state_dict(), round_loss1, round_loss2, acc1, acc2, np.mean(iter_loss)

    def get_training_loss(self, train_loader, train_model=None):
        model = self.local_model if train_model is None else train_model
        model.eval()
        device = self.device
        losses = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
        return np.mean(losses)