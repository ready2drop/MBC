import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from .lr_scheduler import LinearWarmupCosineAnnealingLR



def get_optimizer_loss_scheduler(PARAMS, model):
    # Update the loss function based on the PARAMS
    if PARAMS['loss_function'] == 'CE':
        loss = nn.CrossEntropyLoss()
    elif PARAMS['loss_function'] == 'BCE':
        loss = nn.BCEWithLogitsLoss()     
    else:
        raise ValueError(f"Unsupported loss function: {PARAMS['loss_function']}")

    # Update the optimizer based on the PARAMS
    if PARAMS['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['reg_weight'])
    elif PARAMS['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['reg_weight'])
    elif PARAMS['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=PARAMS['learning_rate'])
    elif PARAMS['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=PARAMS['learning_rate'], momentum=PARAMS['momentum'])
    else:
        raise ValueError(f"Unsupported optimizer: {PARAMS['optimizer']}")

    # Update the scheduler based on the PARAMS
    if PARAMS['scheduler'] == 'StepLR':
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    elif PARAMS['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer)
    elif PARAMS['scheduler'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.1)
    elif PARAMS['scheduler'] == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=50, max_epochs=PARAMS['epochs'])
        
    else:
        raise ValueError(f"Unsupported scheduler: {PARAMS['scheduler']}")


    return loss, optimizer, scheduler



