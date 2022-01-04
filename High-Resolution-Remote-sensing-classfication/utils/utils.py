import shutil
import torch
import sys
import os
from config import config


def save_checkpoint(state, is_best, fold):
    filename = config.weights + config.model_name + os.sep + str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = config.best_models + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar'
        print("Get Better top1 : %s saving weights to %s" % (state["best_precision1"], message))
        with open("./logs/%s.txt" % config.model_name, "a") as f:
            print("Get Better top1 : %s saving weights to %s" % (state["best_precision1"], message), file=f)
        shutil.copyfile(filename, message)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = config.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule(current_epoch, current_lrs, **logs):
    lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
    epochs = [0, 1, 6, 8, 12]
    for lr, epoch in zip(lrs, epochs):
        if current_epoch >= epoch:
            current_lrs[5] = lr
            if current_epoch >= 2:
                current_lrs[4] = lr * 1
                current_lrs[3] = lr * 1
                current_lrs[2] = lr * 1
                current_lrs[1] = lr * 1
                current_lrs[0] = lr * 0.1
    return current_lrs


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, savename,num_classes, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(num_classes)
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(num_classes))
    plt.xticks(xlocations, range(num_classes), rotation=90)
    plt.yticks(xlocations, range(num_classes))
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(num_classes)) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)


    else:
        raise NotImplementedError
