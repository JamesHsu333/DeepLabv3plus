import torch.nn as nn

def CrossEntropy(params):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    if params.cuda:
           criterion = criterion.cuda()
    return criterion

loss_fns = {
    'CrossEntropy': CrossEntropy,
}