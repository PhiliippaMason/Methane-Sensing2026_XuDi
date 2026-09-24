#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F

# from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np

# from torch import optim
import utils.visualization as visual
from utils import data_loader
from utils import lovasz_loss as L
from torch.optim import lr_scheduler
from tqdm import tqdm
import random

# from utils.utils import clip_gradient
from utils.metrics import Evaluator

from network.N_BPMSNet import (
    N_BPMSNet,
    N_BPMSNet_Lite_v1,
    N_BPMSNet_Lite_v2,
    N_BPMSNet_Lite_v3,
    N_BPMSNet_ST,
    N_BPMSNet_VSS,
)

import time

start = time.time()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(
    train_loader,
    val_loader,
    Eva_train,
    Eva_val,
    model_name,
    save_path,
    net,
    criterion,
    optimizer,
    num_epoches,
):
    vis = visual.Visualization()
    vis.create_summary(model_name)
    global best_iou
    epoch_loss = 0
    net.train(True)

    length = 0
    # st = time.time()
    for i, (A, B, C, mask, _) in enumerate(tqdm(train_loader)):
        A, B, C, Y = A, B, C, mask.unsqueeze(1)
        optimizer.zero_grad()
        preds = net(A, B, C)
        if model_name == "N_BPMSNet_Lite_v1" or model_name == "N_BPMSNet_Lite_v2":
            loss = criterion(preds[3], Y) + 0.75 * L.lovasz_hinge(
                preds[-1], Y, ignore=255
            )
        else:
            loss_1 = criterion(preds[0], Y)
            loss_2 = criterion(preds[1], Y)
            loss_3 = criterion(preds[2], Y)  # ADD
            loss_4 = criterion(preds[3], Y)
            lovasz_loss = L.lovasz_hinge(preds[-1], Y, ignore=255)

            loss = loss_1 + loss_2 + loss_3 + loss_4 + 0.75 * lovasz_loss

        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[-1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)

        # print('loss:', loss_1, loss_2, criterion(preds[0], Y), criterion(preds[1], Y), criterion(preds[2], Y))
        Eva_train.add_batch(target, pred)

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, "mIoU")
    vis.add_scalar(epoch, Pre, "Precision")
    vis.add_scalar(epoch, Recall, "Recall")
    vis.add_scalar(epoch, F1, "F1")
    vis.add_scalar(epoch, train_loss, "train_loss")

    print(
        "Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f"
        % (epoch, num_epoches, train_loss, IoU, Pre, Recall, F1)
    )
    print("Strat validing!")

    net.train(False)
    net.eval()
    for i, (A, B, C, mask, _) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A, B, C, Y = A, B, C, mask.unsqueeze(1)
            preds = net(A, B, C)[-1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)

            length += 1
    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print(
        "[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f"
        % (IoU[1], Pre[1], Recall[1], F1[1])
    )
    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        best_net = net.state_dict()
        print(
            "Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d"
            % (IoU[1], F1[1], best_epoch)
        )
        print(save_path)
        torch.save(best_net, save_path + "_best_iou_single.pth")
    print("Best Model Iou :%.4f; F1 :%.4f" % (best_iou, F1[1]))

    # if epoch==999:
    #     best_net = net.state_dict()
    #     torch.save(best_net, save_path + '_best_iou_final_all.pth')
    vis.close_summary()


if __name__ == "__main__":
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=1000, help="epoch number")  # 1000
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=80, help="training dataset size"
    )
    parser.add_argument(
        "--clip", type=float, default=0.5, help="gradient clipping margin"
    )
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )  # 50
    parser.add_argument("--gpu_id", type=str, default="2", help="train use gpu")
    parser.add_argument("--root", type=str, default="./dataset/")
    parser.add_argument(
        "--model_name", type=str, default="N_BPMSNet", help="the test rgb images root"
    )
    parser.add_argument("--save_path", type=str, default="./output")
    opt = parser.parse_args()

    opt.save_path = opt.save_path + "/" + opt.model_name

    train_loader = data_loader.get_loader_single(
        opt.root,
        opt.batchsize,
        opt.trainsize,
        "train",
        channels=13,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = data_loader.get_loader_single(
        opt.root,
        opt.batchsize,
        opt.trainsize,
        "val",
        channels=13,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    Eva_train = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)

    device = torch.device("cuda")
    if opt.model_name == "N_BPMSNet":
        model = N_BPMSNet().cuda()
    elif opt.model_name == "N_BPMSNet_Lite_v1":
        model = N_BPMSNet_Lite_v1().to(device)
    elif opt.model_name == "N_BPMSNet_Lite_v2":
        model = N_BPMSNet_Lite_v2().to(device)
    elif opt.model_name == "N_BPMSNet_Lite_v3":
        model = N_BPMSNet_Lite_v3().to(device)
    elif opt.model_name == "N_BPMSNet_ST":
        model = N_BPMSNet_ST().to(device)
    elif opt.model_name == "N_BPMSNet_VSS":
        model = N_BPMSNet_VSS().to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)

    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2
    )

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # checkpoint_path = save_path + '_best_iou_w4.pth'
    # state_dict = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(state_dict)

    model_name = opt.model_name
    best_iou = 0.0

    print("Start train...")
    # args = parser.parse_args()
    # print('mode is：',args.model_name)

    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group["lr"])
        # cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        Eva_train.reset()
        Eva_val.reset()
        train(
            train_loader,
            val_loader,
            Eva_train,
            Eva_val,
            model_name,
            save_path,
            model,
            criterion,
            optimizer,
            opt.epoch,
        )
        lr_scheduler.step()

end = time.time()
print("Training time:", end - start)
