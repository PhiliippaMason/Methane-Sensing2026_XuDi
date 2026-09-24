#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,

import time

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F

# import torch.nn as nn
import numpy as np
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator

# from network.Net import HANet_v2
# from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.metrics import jaccard_score, average_precision_score  # for IoU

from network.N_BPMSNet import (
    N_BPMSNet,
    N_BPMSNet_Lite_v1,
    N_BPMSNet_Lite_v2,
    N_BPMSNet_Lite_v3,
    N_BPMSNet_ST,
    N_BPMSNet_VSS,
)

# import time

start = time.time()


def binary_iou(y_true, y_pred, class_label):
    intersection = np.logical_and(y_true == class_label, y_pred == class_label).sum()
    union = np.logical_or(y_true == class_label, y_pred == class_label).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def M_iou(y_true, y_pred):
    iou_background = binary_iou(y_true, y_pred, class_label=0)
    iou_foreground = binary_iou(y_true, y_pred, class_label=1)
    mean_iou = (iou_background + iou_foreground) / 2
    return mean_iou, iou_background, iou_foreground


def dice_coefficient(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    dice_score = (
        (2 * intersection) / (y_true.sum() + y_pred.sum())
        if (y_true.sum() + y_pred.sum()) != 0
        else 0
    )
    return dice_score


def sensitivity(y_true, y_pred):
    # Sensitivity (Recall for foreground)
    true_positive = np.logical_and(y_pred == 1, y_true == 1).sum()
    false_negative = np.logical_and(y_pred == 0, y_true == 1).sum()
    return (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) != 0
        else 0
    )


def specificity(y_true, y_pred):
    # Specificity (Recall for background)
    true_negative = np.logical_and(y_pred == 0, y_true == 0).sum()
    false_positive = np.logical_and(y_pred == 1, y_true == 0).sum()
    return (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive) != 0
        else 0
    )


# Scene-level Metrics
def scene_level_metrics(predictions, labels, probs):
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probs)
    conf_matrix = confusion_matrix(labels, predictions).ravel()

    tn, fp, fn, tp = conf_matrix if len(conf_matrix) == 4 else [0, 0, 0, 0]

    return {
        "Scene Accuracy": acc,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "True Negatives": tn,
    }


#  Pixel-level
def pixel_level_metrics(predictions, labels, probs):
    predictions = predictions.flatten()
    labels = labels.flatten()
    probs = probs.flatten()

    accuracy = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    ap_score = average_precision_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    # f0_5 = f1_score(labels, predictions, beta=0.5)
    iou = jaccard_score(labels, predictions)
    auc = roc_auc_score(labels, probs)
    mean_iou, iou_background, iou_foreground = M_iou(labels, predictions)
    dice_coeffi = dice_coefficient(labels, predictions)
    sens = sensitivity(labels, predictions)
    spec = specificity(labels, predictions)

    return {
        "Pixel Accuracy": accuracy,
        "Balanced Accuracy": balanced_acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        # 'F0.5': f0_5,
        "IoU": iou,
        "AUC": auc,
        "AP Score": ap_score,
        "Mean IOU": mean_iou,
        "Background IOU": iou_background,
        "Foreground IOU": iou_foreground,
        "Dice Coefficient": dice_coeffi,
        "Sensitivity": sens,
        "Specificity": spec,
    }


def test(test_loader, Eva_test, save_path, net):
    print("Strat validing!")

    net.train(False)
    net.eval()

    for i, (A, B, C, mask, data_name) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            C = C.cuda()
            # Y = mask.cuda()
            preds = net(A, B, C)
            probs = F.sigmoid(preds[-1])
            predictions = F.sigmoid(preds[-1])
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            predictions = predictions.squeeze(1).data.cpu().numpy().astype(int)
            # labels = Y.cpu().numpy()
            probs = probs.cpu().numpy()

            for i in range(predictions.shape[0]):
                pred_image = predictions[i]
                # label_image = labels[i]
                np.save(
                    f"../../dataset/mask_gray_p/{data_name.cpu().numpy()[i]}_p.npy",
                    pred_image,
                )

    print("done")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=80, help="training dataset size"
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="train use gpu"
    )  # Change!!!
    parser.add_argument("--root", type=str, default="./dataset/")  # Change!!!
    parser.add_argument(
        "--model_name",
        type=str,
        default="N_BPMSNet",  # Change!!!
        help="the test rgb images root",
    )
    parser.add_argument("--save_path", type=str, default="./output/")
    opt = parser.parse_args()

    test_loader = data_loader.get_loader(
        opt.root,
        opt.batchsize,
        opt.trainsize,
        "train",
        channels=13,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )
    # test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_test = Evaluator(num_class=2)

    if opt.model_name == "N_BPMSNet":
        model = N_BPMSNet().cuda()
    elif opt.model_name == "N_BPMSNet_Lite_v1":
        model = N_BPMSNet_Lite_v1().cuda()
    elif opt.model_name == "N_BPMSNet_Lite_v2":
        model = N_BPMSNet_Lite_v2().cuda()
    elif opt.model_name == "N_BPMSNet_Lite_v3":
        model = N_BPMSNet_Lite_v3().cuda()
    elif opt.model_name == "N_BPMSNet_ST":
        model = N_BPMSNet_ST().cuda()
    elif opt.model_name == "N_BPMSNet_VSS":
        model = N_BPMSNet_VSS().cuda()

    opt.load = "./output/" + opt.model_name + "_best_iou.pth"

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print("load model from ", opt.load)

    test(test_loader, Eva_test, opt.save_path, model)

end = time.time()
print("程序测试test的时间为:", end - start)
