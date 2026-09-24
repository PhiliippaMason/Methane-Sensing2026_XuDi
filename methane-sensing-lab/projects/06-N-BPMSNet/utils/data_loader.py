# import os
from PIL import Image
import torch.utils.data as data

# import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
from PIL import ImageEnhance

# from PIL import Image
from numpy import random as npr
import torch


def cv_random_flip(img_A, img_B, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label


def randomCrop_Mosaic(image_A, image_B, label, crop_win_width, crop_win_height):
    image_width = image_A.size[0]
    image_height = image_A.size[1]
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return (
        image_A.crop(random_region),
        image_B.crop(random_region),
        label.crop(random_region),
    )


def randomCrop(image_A, image_B, label):
    border = 30
    image_width = image_A.size[0]
    image_height = image_B.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return (
        image_A.crop(random_region),
        image_B.crop(random_region),
        label.crop(random_region),
    )


def randomRotation(image_A, image_B, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image_A = image_A.rotate(random_angle, mode)
        image_B = image_B.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    bright_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Brightness(image_A).enhance(bright_intensity)
    image_B = ImageEnhance.Brightness(image_B).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Contrast(image_A).enhance(contrast_intensity)
    image_B = ImageEnhance.Contrast(image_B).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image_A = ImageEnhance.Color(image_A).enhance(color_intensity)
    image_B = ImageEnhance.Color(image_B).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image_A = ImageEnhance.Sharpness(image_A).enhance(sharp_intensity)
    image_B = ImageEnhance.Sharpness(image_B).enhance(sharp_intensity)
    return image_A, image_B


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class ChangeDataset(data.DataLoader):

    def __init__(
        self,
        device,
        mode,
        dataset_path,
        crop_size=80,
        plume_id=None,
        red=False,
        alli=False,
        channels=13,
    ):
        self.device = device
        self.mode = mode
        self.reduce = red
        self.channels = channels
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        if mode == "train":
            # train_info = pd.read_csv("./dataset/train.csv")
            # val_info = pd.read_csv('./dataset/val.csv')
            train_info = pd.read_csv("./dataset/train_site.csv")  # site-independent
            val_info = pd.read_csv("./dataset/val_site.csv")
            self.data_info = pd.concat(
                [train_info, val_info], axis=0, ignore_index=True
            )
        else:
            # self.data_info = pd.read_csv("./dataset/{}.csv".format(mode))
            self.data_info = pd.read_csv(
                "./dataset/{}_site_new.csv".format(mode)
            )  # site-independent
            # if mode == "test":
            #     self.data_info = self.data_info[self.data_info['site'].str.startswith('A')] #A T U
            #     self.data_info = self.data_info.reset_index(drop=True)

        self.pos_labels = list(self.data_info[self.data_info["plume"] == 2]["index"])
        self.pos_labels_next = list(
            self.data_info[self.data_info["plume"] == 2]["next_index"]
        )
        self.neg_labels = list(self.data_info[self.data_info["plume"] == 0]["index"])
        self.neg_labels_next = list(
            self.data_info[self.data_info["plume"] == 0]["next_index"]
        )
        if mode == "train":
            persist = False
        else:
            persist = True

        self.labels = self.pos_labels + self.neg_labels  # None
        self.labels_next = self.pos_labels_next + self.neg_labels_next
        if not alli:
            self.sample_labels_and_combine(persist=persist)

    def sample_labels_and_combine(self, persist=False):
        """
        Sample a subset of negative labels for each epoch
        """
        # if self.mode == "test":
        if self.mode in ["test", "val"]:  # CHANGE to ALL !!!!!!!!!!!!!!!!
            self.labels = self.pos_labels + self.neg_labels
            self.labels_next = self.pos_labels_next + self.neg_labels_next
        else:
            if persist:
                random.seed(555)

            # shuffle(self.neg_labels)
            # self.labels = self.pos_labels+self.neg_labels[:len(self.pos_labels)]

            random_indices = random.sample(
                range(len(self.neg_labels)), len(self.pos_labels) * 3
            )
            self.labels = self.pos_labels + [self.neg_labels[i] for i in random_indices]
            self.labels_next = self.pos_labels_next + [
                self.neg_labels_next[i] for i in random_indices
            ]

    def random_fliplr(self, pre_img, post_img, diff_img, label):
        if random.random() > 0.5:
            label = np.fliplr(label)
            pre_img = np.fliplr(pre_img)
            post_img = np.fliplr(post_img)
            diff_img = np.fliplr(diff_img)
        return pre_img, post_img, diff_img, label

    def random_flipud(self, pre_img, post_img, diff_img, label):
        if random.random() > 0.5:
            label = np.flipud(label)
            pre_img = np.flipud(pre_img)
            post_img = np.flipud(post_img)
            diff_img = np.flipud(diff_img)
        return pre_img, post_img, diff_img, label

    def random_rot(self, pre_img, post_img, diff_img, label):
        k = random.randrange(3) + 1
        pre_img = np.rot90(pre_img, k, axes=(1, 2)).copy()
        post_img = np.rot90(post_img, k, axes=(1, 2)).copy()
        diff_img = np.rot90(diff_img, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()

        return pre_img, post_img, diff_img, label

    def transforms(self, aug, pre_img, post_img, diff_img, label):
        if aug:
            # pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, diff_img, label = self.random_fliplr(
                pre_img, post_img, diff_img, label
            )
            pre_img, post_img, diff_img, label = self.random_flipud(
                pre_img, post_img, diff_img, label
            )
            pre_img, post_img, diff_img, label = self.random_rot(
                pre_img, post_img, diff_img, label
            )

        # pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        # pre_img = np.transpose(pre_img, (2, 0, 1))

        # post_img = imutils.normalize_img(post_img)  # imagenet normalization
        # post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, diff_img, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index_new = int(self.labels[index])
        index_new_next = int(self.labels_next[index])
        # target = np.load("../../../dataset/mask/{}.jpg".format(index_new))
        target = (
            np.array(
                Image.open(
                    self.dataset_path + "mask_gray/{}.jpg".format(index_new)
                ).convert("L")
            )
            / 255
        )
        context_1 = np.load(self.dataset_path + "s2/{}.npy".format(index_new))
        context_2 = np.load(self.dataset_path + "s2/{}.npy".format(index_new_next))
        # context = np.concatenate([context_1, context_2], axis=0)

        ndmi_C1 = (context_1[11, ...] - context_1[12, ...]) / (
            context_1[11, ...] + context_1[12, ...]
        )
        ndmi_C2 = (context_2[11, ...] - context_2[12, ...]) / (
            context_2[11, ...] + context_2[12, ...]
        )
        ndmi_C3 = ndmi_C1 - ndmi_C2
        ndmi_C1 = np.expand_dims(ndmi_C1, axis=0)
        ndmi_C2 = np.expand_dims(ndmi_C2, axis=0)
        ndmi_C3 = np.expand_dims(ndmi_C3, axis=0)
        context_3 = np.concatenate([ndmi_C1, ndmi_C2, ndmi_C3], axis=0)

        if self.channels == 2:
            context_1 = context_1[11:, ...]
            context_2 = context_2[11:, ...]
        if self.channels == 5:
            context_1 = np.concatenate(
                [context_1[1:4, ...], context_1[11:, ...]], axis=0
            )
            context_2 = np.concatenate(
                [context_2[1:4, ...], context_2[11:, ...]], axis=0
            )

        # Crop to centre
        # x_c = target.shape[0]//2
        # y_c = target.shape[1]//2
        s = self.crop_size

        if self.mode == "train":
            rng = npr.RandomState()
            mid_loc_x = rng.randint(s, target.shape[0] - s)
            mid_loc_y = rng.randint(s, target.shape[1] - s)

        else:
            mid_loc_x = target.shape[0] // 2
            mid_loc_y = target.shape[1] // 2

        target = target[mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s]

        context_1 = context_1[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]
        context_2 = context_2[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]
        context_3 = context_3[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]

        if self.reduce:
            target = np.array([np.int(target.any())])

        if self.mode == "train":
            pre_img, post_img, diff_img, label = self.transforms(
                True, context_1, context_2, context_3, target
            )
        elif self.mode == "val":
            pre_img, post_img, diff_img, label = self.transforms(
                False, context_1, context_2, context_3, target
            )
            # label = np.asarray(label)
        else:
            pre_img, post_img, diff_img, label = context_1, context_2, context_3, target

        pre_img = torch.from_numpy(context_1).float().to(self.device)  # /255
        post_img = torch.from_numpy(context_2).float().to(self.device)  # /255
        diff_img = torch.from_numpy(context_3).float().to(self.device)
        label = torch.from_numpy(target).float().to(self.device)
        data_idx = index_new

        return pre_img, post_img, diff_img, label, data_idx


def get_loader(
    root,
    batchsize,
    trainsize,
    atype,
    channels=13,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
):
    dataset = ChangeDataset(
        device="cuda",
        mode=atype,
        dataset_path=root,
        crop_size=trainsize,
        plume_id=None,
        channels=channels,
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle
    )
    return data_loader


class ChangeDataset_single(data.DataLoader):

    def __init__(
        self,
        device,
        mode,
        dataset_path,
        crop_size=80,
        plume_id=None,
        red=False,
        alli=False,
        channels=13,
    ):
        self.device = device
        self.mode = mode
        self.reduce = red
        self.channels = channels
        self.crop_size = crop_size
        self.dataset_path = dataset_path
        if mode == "train":
            train_info = pd.read_csv("./dataset/train.csv")
            val_info = pd.read_csv("./dataset/val.csv")
            self.data_info = pd.concat(
                [train_info, val_info], axis=0, ignore_index=True
            )
        else:
            self.data_info = pd.read_csv("./dataset/{}.csv".format(mode))
            # if mode == "test":
            #     self.data_info = self.data_info[self.data_info['site'].str.startswith('A')] #A T
            #     self.data_info = self.data_info.reset_index(drop=True)

        self.pos_labels = list(self.data_info[self.data_info["plume"] == 2]["index"])
        # self.pos_labels_next = list(self.data_info[self.data_info['plume']==2]['next_index'])
        self.neg_labels = list(self.data_info[self.data_info["plume"] == 0]["index"])
        # self.neg_labels_next = list(self.data_info[self.data_info['plume']==0]['next_index'])
        if mode == "train":
            persist = False
        else:
            persist = True

        self.labels = self.pos_labels + self.neg_labels  # None
        # self.labels_next = self.pos_labels_next+self.neg_labels_next
        if not alli:
            self.sample_labels_and_combine(persist=persist)

    def sample_labels_and_combine(self, persist=False):
        """
        Sample a subset of negative labels for each epoch
        """
        # if self.mode == "test":
        if self.mode in ["test", "val"]:  # CHANGE to ALL !!!!!!!!!!!!!!!!
            self.labels = self.pos_labels + self.neg_labels
            # self.labels_next = self.pos_labels_next+self.neg_labels_next
        else:
            if persist:
                random.seed(555)

            # shuffle(self.neg_labels)
            # self.labels = self.pos_labels+self.neg_labels[:len(self.pos_labels)]

            random_indices = random.sample(
                range(len(self.neg_labels)), len(self.pos_labels) * 3
            )
            self.labels = self.pos_labels + [self.neg_labels[i] for i in random_indices]
            # self.labels_next = self.pos_labels_next+[self.neg_labels_next[i] for i in random_indices]

    def random_fliplr(self, pre_img, post_img, diff_img, label):
        if random.random() > 0.5:
            label = np.fliplr(label)
            pre_img = np.fliplr(pre_img)
            post_img = np.fliplr(post_img)
            diff_img = np.fliplr(diff_img)
        return pre_img, post_img, diff_img, label

    def random_flipud(self, pre_img, post_img, diff_img, label):
        if random.random() > 0.5:
            label = np.flipud(label)
            pre_img = np.flipud(pre_img)
            post_img = np.flipud(post_img)
            diff_img = np.flipud(diff_img)
        return pre_img, post_img, diff_img, label

    def random_rot(self, pre_img, post_img, diff_img, label):
        k = random.randrange(3) + 1
        pre_img = np.rot90(pre_img, k, axes=(1, 2)).copy()
        post_img = np.rot90(post_img, k, axes=(1, 2)).copy()
        diff_img = np.rot90(diff_img, k, axes=(1, 2)).copy()
        label = np.rot90(label, k, axes=(0, 1)).copy()

        return pre_img, post_img, diff_img, label

    def transforms(self, aug, pre_img, post_img, diff_img, label):
        if aug:
            # pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, diff_img, label = self.random_fliplr(
                pre_img, post_img, diff_img, label
            )
            pre_img, post_img, diff_img, label = self.random_flipud(
                pre_img, post_img, diff_img, label
            )
            pre_img, post_img, diff_img, label = self.random_rot(
                pre_img, post_img, diff_img, label
            )

        # pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        # pre_img = np.transpose(pre_img, (2, 0, 1))

        # post_img = imutils.normalize_img(post_img)  # imagenet normalization
        # post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, diff_img, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index_new = int(self.labels[index])
        # index_new_next = int(self.labels_next[index])
        # target = np.load("../../../dataset/mask/{}.jpg".format(index_new))
        target = (
            np.array(
                Image.open(
                    self.dataset_path + "mask_gray/{}.jpg".format(index_new)
                ).convert("L")
            )
            / 255
        )
        context_1 = np.load(self.dataset_path + "s2/{}.npy".format(index_new))
        # context_2 = np.load(self.dataset_path+"s2/{}.npy".format(index_new_next))
        context_2 = context_1
        # context = np.concatenate([context_1, context_2], axis=0)

        ndmi_C1 = (context_1[11, ...] - context_1[12, ...]) / (
            context_1[11, ...] + context_1[12, ...]
        )
        ndmi_C2 = (context_2[11, ...] - context_2[12, ...]) / (
            context_2[11, ...] + context_2[12, ...]
        )
        ndmi_C3 = ndmi_C1 - ndmi_C2
        ndmi_C1 = np.expand_dims(ndmi_C1, axis=0)
        ndmi_C2 = np.expand_dims(ndmi_C2, axis=0)
        ndmi_C3 = np.expand_dims(ndmi_C3, axis=0)
        context_3 = np.concatenate([ndmi_C1, ndmi_C2, ndmi_C3], axis=0)

        if self.channels == 2:
            context_1 = context_1[11:, ...]
            context_2 = context_2[11:, ...]
        if self.channels == 5:
            context_1 = np.concatenate(
                [context_1[1:4, ...], context_1[11:, ...]], axis=0
            )
            context_2 = np.concatenate(
                [context_2[1:4, ...], context_2[11:, ...]], axis=0
            )

        # Crop to centre
        # x_c = target.shape[0]//2
        # y_c = target.shape[1]//2
        s = self.crop_size

        if self.mode == "train":
            rng = npr.RandomState()
            mid_loc_x = rng.randint(s, target.shape[0] - s)
            mid_loc_y = rng.randint(s, target.shape[1] - s)

        else:
            mid_loc_x = target.shape[0] // 2
            mid_loc_y = target.shape[1] // 2

        target = target[mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s]

        context_1 = context_1[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]
        context_2 = context_2[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]
        context_3 = context_3[
            :, mid_loc_x - s : mid_loc_x + s, mid_loc_y - s : mid_loc_y + s
        ]

        if self.reduce:
            target = np.array([np.int(target.any())])

        if self.mode == "train":
            pre_img, post_img, diff_img, label = self.transforms(
                True, context_1, context_2, context_3, target
            )
        elif self.mode == "val":
            pre_img, post_img, diff_img, label = self.transforms(
                False, context_1, context_2, context_3, target
            )
            # label = np.asarray(label)
        else:
            pre_img, post_img, diff_img, label = context_1, context_2, context_3, target

        pre_img = torch.from_numpy(context_1).float().to(self.device)  # /255
        post_img = torch.from_numpy(context_2).float().to(self.device)  # /255
        diff_img = torch.from_numpy(context_3).float().to(self.device)
        label = torch.from_numpy(target).float().to(self.device)
        data_idx = index_new

        return pre_img, post_img, diff_img, label, data_idx


def get_loader_single(
    root,
    batchsize,
    trainsize,
    atype,
    channels=13,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
):
    dataset = ChangeDataset_single(
        device="cuda",
        mode=atype,
        dataset_path=root,
        crop_size=trainsize,
        plume_id=None,
        channels=channels,
    )
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle
    )
    return data_loader
