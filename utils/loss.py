import torch.nn as nn

def CrossEntropy(outputs, labels):
    return nn.CrossEntropyLoss(ignore_index=255)(outputs, labels)

loss_fns = {
    'CrossEntropy': CrossEntropy,
}