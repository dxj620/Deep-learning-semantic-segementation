import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from IPython import embed
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# 2. evaluate func
def evaluate(val_loader, model, criterion, epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    # progress bar
    val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=config.epochs, model_name=config.model_name,
                                 total=len(val_loader))
    # 2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        target_all = []
        output_all = []
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = input.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()
            # target = target.to(device)
            # 2.2.1 compute output
            output = model(input)
            loss = criterion(output, target)

            # 2.2.2 measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            for j in target.cpu().detach().numpy():
                target_all.append(j)
            for j in torch.argmax(output.cpu(), 1).numpy():
                output_all.append(j)
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
        p_class, r_class, f_class, support_micro = precision_recall_fscore_support(target_all, output_all)
        num_classes = 6
        # mask = (target_all >= 0) & (target_all < num_classes)
        conf_mat = confusion_matrix(target_all, output_all)
        acc_per_class = np.diag(conf_mat) / conf_mat.sum(axis=1)
        print(acc_per_class)
        print(r_class)
        print(f_class)

        # plot_confusion_matrix(cm, 'C:\\code\\dxj\\pytorch-image-classification\\sy_result\\confusion_matrix_' + str(
        #     epoch) + '.png', config.num_classes,
        #                       title='confusion matrix_'+str(epoch))
    return top1.avg.item()


# 3. test model on public dataset and save the probability matrix
# def test(test_loader, model, folds):
#     # print("jjjj")
#     # 3.1 confirm the model converted to cuda
#     csv_map = OrderedDict({"filename": [], "probability": []})
#     model.to(device)
#     model.eval()
#     for i, (input, filepath) in enumerate(tqdm(test_loader)):
#         # 3.2 change everything to cuda and get only basename
#         filepath = [os.path.basename(x) for x in filepath]
#         with torch.no_grad():
#             image_var = input.to(device)
#             # 3.3.output
#             # print(filepath)
#             # print(input,input.shape)
#             y_pred = model(image_var)
#             print(y_pred.shape)
#             smax = nn.Softmax(1)
#             smax_out = smax(y_pred)
#         # 3.4 save probability to csv files
#         csv_map["filename"].extend(filepath)
#         for output in smax_out:
#             prob = ";".join([str(i) for i in output.data.tolist()])
#             csv_map["probability"].append(prob)
#     result = pd.DataFrame(csv_map)
#     result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
#     result.to_csv("./submit/{}_submission.csv".format(config.model_name + "_" + str(folds)), index=False, header=None)


# 4. more details to build main function
def main():
    fold = 0
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep + str(fold) + os.sep)
        # 4.2 get model and optimizer
    # model = generate_model()
    model = get_net()
    model = nn.DataParallel(model, device_ids=[0, 1])
    # model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()
    # 4.3 some parameters for  K-fold and restart model
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False
    # 4.4 restart the training process
    if resume:
        checkpoint = torch.load(config.best_models + str(fold) + "/model_best.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 4.5 get files and split for K-fold dataset
    # 4.5.1 read files
    train_data_list = get_files(config.train_data, "train")
    val_data_list = get_files(config.val_data, "val")
    # test_files = get_files(config.test_data,"test")

    """ 
    #4.5.2 split
    split_fold = StratifiedKFold(n_splits=3)
    folds_indexes = split_fold.split(X=origin_files["filename"],y=origin_files["label"])
    folds_indexes = np.array(list(folds_indexes))
    fold_index = folds_indexes[fold]

    #4.5.3 using fold index to split for train data and val data
    train_data_list = pd.concat([origin_files["filename"][fold_index[0]],origin_files["label"][fold_index[0]]],axis=1)
    val_data_list = pd.concat([origin_files["filename"][fold_index[1]],origin_files["label"][fold_index[1]]],axis=1)
    """
    # train_data_list,val_data_list = train_test_split(origin_files,test_size = 0.1,stratify=origin_files["label"])
    # 4.5.4 load dataset
    train_dataloader = DataLoader(ChaojieDataset(train_data_list), batch_size=config.batch_size, shuffle=True,
                                  collate_fn=collate_fn, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(ChaojieDataset(val_data_list, train=False), batch_size=config.batch_size * 2,
                                shuffle=True, collate_fn=collate_fn, pin_memory=False, num_workers=4)
    # test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=1,shuffle=False,pin_memory=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",verbose=1,patience=3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf, 0, 0]
    model.train()
    # 4.5.5 train
    start = timer()
    for epoch in range(start_epoch, config.epochs):
        scheduler.step(epoch)
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=config.epochs,
                                       model_name=config.model_name, total=len(train_dataloader))
        # train
        # global iter
        for iter, (input, target) in enumerate(train_dataloader):
            # 4.5.5 switch to continue train process
            train_progressor.current = iter
            model.train()
            input = input.cuda()
            target = torch.from_numpy(np.array(target)).long().cuda()
            # target = target.to(device)
            output = model(input)
            loss = criterion(output, target)

            precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_progressor()
        train_progressor.done()
        # evaluate
        lr = get_learning_rate(optimizer)
        # evaluate every half epoch
        valid_acc = evaluate(val_dataloader, model, criterion, epoch)
        is_best = valid_loss[1] > best_precision1
        best_precision1 = max(valid_loss[1], best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass

        # save_checkpoint({
        #     "epoch": epoch + 1,
        #     "model_name": config.model_name,
        #     "state_dict": model.state_dict(),
        #     "best_precision1": best_precision1,
        #     "optimizer": optimizer.state_dict(),
        #     "fold": fold,
        #     "valid_loss": valid_loss,
        # }, is_best, fold)
        # torch.save(model.module.state_dict(),
        #            "D:\\dataset\\Graduation_project\\model\\resnet50_model_512_sy\\" + str(round(valid_acc, 3)) + '_' + str(
        #                epoch) + '.pkl')


if __name__ == "__main__":
    main()
