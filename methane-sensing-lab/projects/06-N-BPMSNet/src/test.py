#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,

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
import time
from thop import profile

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


# 计算 Scene-level 指标 (包括 AUC)
def scene_level_metrics(predictions, labels, probs, data_names):
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    auc = roc_auc_score(labels, probs)  # 计算 AUC
    conf_matrix = confusion_matrix(labels, predictions).ravel()

    tn, fp, fn, tp = conf_matrix if len(conf_matrix) == 4 else [0, 0, 0, 0]

    fp_indices = [
        data_names[i] for i in np.where((labels == 0) & (predictions == 1))[0]
    ]
    fn_indices = [
        data_names[i] for i in np.where((labels == 1) & (predictions == 0))[0]
    ]

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
        "FP IDs": fp_indices,
        "FN IDs": fn_indices,
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

    all_scene_predictions = []
    all_scene_labels = []
    all_scene_probs = []
    all_scene_names = []
    all_pixel_predictions = []
    all_pixel_labels = []
    all_pixel_probs = []

    for i, (A, B, C, mask, data_name) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            C = C.cuda()
            Y = mask.cuda()
            preds = net(A, B, C)
            probs = F.sigmoid(preds[-1])
            predictions = F.sigmoid(preds[-1])
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            predictions = predictions.squeeze(1).data.cpu().numpy().astype(int)
            labels = Y.cpu().numpy()
            probs = probs.cpu().numpy()
            names = data_name.cuda().cpu().numpy()

            for i in range(predictions.shape[0]):
                pred_image = predictions[i]
                label_image = labels[i]
                name = names[i]

                scene_label = label_image.sum().item() > 0
                if int(scene_label) > 0:
                    pred_intersection = (pred_image * label_image).sum().item()
                else:
                    pred_intersection = (pred_image).sum().item()
                scene_pred = pred_intersection > 0

                all_scene_predictions.append(int(scene_pred))
                all_scene_labels.append(int(scene_label))
                all_scene_probs.append(probs[i].mean().item())
                all_scene_names.append(name)

                all_pixel_predictions.append(pred_image)
                all_pixel_labels.append(label_image)
                all_pixel_probs.append(probs[i])

                # label_image_uint8 = (label_image * 255).astype(np.uint8)
                # label_img = Image.fromarray(label_image_uint8)
                # label_img.save(f'./res_image/{data_name.cpu().numpy()[i]}.png')

                # pred_image_uint8 = (pred_image * 255).astype(np.uint8)
                # img = Image.fromarray(pred_image_uint8)
                # img.save(f'./res_image/{data_name.cpu().numpy()[i]}_p.png')
            # print(labels.shape, predictions.shape)
            Eva_test.add_batch(labels, predictions)

    scene_metrics = scene_level_metrics(
        np.array(all_scene_predictions),
        np.array(all_scene_labels),
        np.array(all_scene_probs),
        np.array(all_scene_names),
    )
    pixel_metrics = pixel_level_metrics(
        np.concatenate(all_pixel_predictions),
        np.concatenate(all_pixel_labels),
        np.concatenate(all_pixel_probs),
    )

    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA = Eva_test.OA()
    Kappa = Eva_test.Kappa()

    # print('[Test] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print(
        "[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f,IoU: %.4f"
        % (F1[1], Pre[1], Recall[1], OA[1], Kappa[1], IoU[1])
    )
    # print('F1-Score: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}\nOA: {:.2f}\nKappa: {:.2f}\nIoU: {:.2f}\n}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100, IoU[1] * 100))
    print("F1-Score: Precision: Recall: OA: Kappa: IoU: ")
    print(
        "{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}".format(
            F1[1] * 100,
            Pre[1] * 100,
            Recall[1] * 100,
            OA[1] * 100,
            Kappa[1] * 100,
            IoU[1] * 100,
        )
    )
    print(
        "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(
            F1[1] * 100,
            Pre[1] * 100,
            Recall[1] * 100,
            OA[1] * 100,
            Kappa[1] * 100,
            IoU[1] * 100,
        )
    )
    return scene_metrics, pixel_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16, help="training batch size")
    parser.add_argument(
        "--trainsize", type=int, default=80, help="training dataset size"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="train use gpu")
    parser.add_argument("--root", type=str, default="./dataset/")
    parser.add_argument(
        "--model_name", type=str, default="N_BPMSNet", help="the test rgb images root"
    )
    parser.add_argument("--save_path", type=str, default="./output/")
    opt = parser.parse_args()

    test_loader = data_loader.get_loader(
        opt.root,
        opt.batchsize,
        opt.trainsize,
        "test",
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
    # opt.load = './output/' + opt.model_name + '_best_iou_site.pth'
    # opt.load = './output/' + opt.model_name + '_local.pth'

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load, weights_only=True))
        print("load model from ", opt.load)

    print("Total params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    # -------------------------
    #  FLOPs
    # -------------------------
    dummy_input1 = torch.randn(1, 13, 160, 160).cuda()
    dummy_input2 = torch.randn(1, 13, 160, 160).cuda()
    dummy_input3 = torch.randn(1, 3, 160, 160).cuda()
    flops, params = profile(
        model,
        inputs=(
            dummy_input1,
            dummy_input2,
            dummy_input3,
        ),
        verbose=False,
    )
    print(f"Params: {params/1e6:.2f} M")
    print(f"FLOPs: {flops/1e9:.2f} G")

    # -------------------------
    # Infer Time
    # -------------------------
    n_runs = 50
    model.cuda()
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_input1, dummy_input2, dummy_input3)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / n_runs
    print(f"Inference Time: {avg_inference_time:.4f} s per image")

    scene_metrics, pixel_metrics = test(test_loader, Eva_test, opt.save_path, model)
    # Output
    print("Scene-level metrics:")
    for key, value in scene_metrics.items():
        if key in ["FP IDs", "FN IDs"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")

    print("\nPixel-level metrics:")
    for key, value in pixel_metrics.items():
        print(f"{key}: {value:.4f}")

end = time.time()
print("test time:", end - start)
