
from .fedavg import *
from .fedper import *
from .lg_fedavg import *
from .local import *
from .fedpac import *
from .fedprox import *
from .fedld import *
from .fedgh import *


def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedPer':LocalUpdate_FedPer,
                   'LG_FedAvg':LocalUpdate_LG_FedAvg,
                   'Local':LocalUpdate_StandAlone,
                   'FedPAC':LocalUpdate_FedPAC,
                   'FedProx':LocalUpdate_FedProx,
                   'FedLD':LocalUpdate_FedLD,
                   'FedGH':LocalUpdate_FedGH,
    }

    return LocalUpdate[rule]