import os
import gc
import json
import shutil
import tempfile

from .EarlyStopping import EarlyStopping

from tqdm import tqdm

from monai.data import (
    DataLoader,
    ThreadDataLoader,
    SmartCacheDataset,
    PersistentDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

import numpy as np
import nibabel as nib
import concurrent.futures

from monai.data.utils import pad_list_data_collate

import matplotlib.pyplot as plt
import monai.losses as losses

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import Transform
from monai.transforms import AsDiscrete

from monai.config import print_config
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric

from monai.data import decollate_batch

from .surface_distance import metrics as sd

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, PolynomialLR

print_config()


class Trainer:

    def __init__(self, model, loss_function, optimizer, number_of_classes, num_samples, model_root_path, max_epoch):
        self.model = model
        # self.global_step = None
        # self.dice_val_best = None
        # self.metric_values = None
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        # self.val_loss_values = None
        # self.global_step_best = None
        # self.epoch_loss_values = None
        self.num_samples = num_samples
        self.loss_function = loss_function
        self.model_root_path = model_root_path
        self.scaler = torch.cuda.amp.GradScaler()
        self.number_of_classes = number_of_classes
        self.train_info_path = f'{model_root_path}/train_info.json'
        self.model_path = f'{model_root_path}/best_metric_model.pth'
        self.test_info_path = f'{model_root_path}/test_metrics.json'
        self.post_label = AsDiscrete(to_onehot=self.number_of_classes)
        self.best_epoch_path = f'{model_root_path}/best_epochs_info.json'
        self.early_stopping = EarlyStopping(tolerance=10, min_delta=0.001)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.number_of_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.scheduler = ReduceLROnPlateau(self.optimizer, patience=3, verbose=True, mode='min',
        #                                    min_lr=1e-8, factor=0.25)
        self.scheduler = PolynomialLR(self.optimizer, total_iters=self.max_epoch, power=0.9, verbose=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, get_not_nans=False, reduction="none",
                                                        distance_metric="euclidean")
        # if self.model_name == 'Swin':
        #     self.model = SwinUNETR(
        #         img_size=(96, 96, 96),
        #         in_channels=1,
        #         out_channels=number_of_classes + 1,
        #         feature_size=48,
        #         #     drop_rate=0.25,
        #         use_checkpoint=True,
        #     ).to(self.device)

        # self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1, lambda_ce=1)
        # self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

    def load_weights(self, weights_path='./model_swinvit.pt'):
        weight = torch.load(weights_path, map_location=torch.device(self.device))
        self.model.load_from(weights=weight)

    def validation(self, epoch_iterator_val):
        model = self.model
        model.eval()
        validation_loss = []
        with torch.no_grad():
            for batch in epoch_iterator_val:
                # Free GPU memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if batch['label'].max() > 0:
                        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), self.num_samples, model)
                            loss = self.loss_function(val_outputs, val_labels).item()
                        # val_labels = [self.post_label(val_label_tensor) for val_label_tensor in decollate_batch(val_labels)]
                        # val_outputs = [self.post_pred(val_pred_tensor) for val_pred_tensor in decollate_batch(val_outputs)]
                        validation_loss.append(loss)
                        self.dice_metric(
                            y_pred=[self.post_pred(val_pred_tensor) for val_pred_tensor in
                                    decollate_batch(val_outputs)],
                            y=[self.post_label(val_label_tensor) for val_label_tensor in decollate_batch(val_labels)])
                        # print(torch.cuda.memory_reserved() / (1024**3), torch.cuda.memory_allocated() / (1024**3))

                        batch = batch["image"].detach(), batch["label"].detach()
                        # Free GPU memory
                        torch.cuda.empty_cache()
                    epoch_iterator_val.set_description(f"Validate (loss={loss:2.5f})")
            mean_dice_val = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

            mean_hausdorff = 0
            validation_loss_mean = np.nanmean(np.nan_to_num(np.array(validation_loss),
                                                            nan=np.nan, posinf=np.nan, neginf=np.nan))

            # self.val_loss_values.append(validation_loss_mean)
        return mean_dice_val, mean_hausdorff, validation_loss_mean

    def train_one_epoch(self, train_loader, global_step):
        model = self.model
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            if batch['label'].max() > 0:
                # print(batch['label'].max())
                step += 1
                x, y = (batch["image"].cuda(), batch["label"].cuda())
                with torch.cuda.amp.autocast():
                    logit_map = model(x)
                    loss = self.loss_function(logit_map, y)
                self.scaler.scale(loss).backward()
                epoch_loss += loss.item()
                if torch.isnan(loss):
                    print(torch.max(x), torch.min(x))

                self.scaler.unscale_(self.optimizer)

                self.scaler.step(self.optimizer)

                self.scaler.update()
                self.optimizer.zero_grad()
                x, y = None, None

                epoch_iterator.set_description(
                    f"Training ({global_step} / {self.max_epoch} Steps) (loss={loss.item():2.5f})")
            # break
        epoch_loss /= step
        epoch_iterator = None
        return epoch_loss

    def train(self, train_loader, val_loader):
        train_info = {}
        global_step = 0
        dice_val_best = -1.0
        best_epochs_info = {}
        self.model = self.model.to(self.device)
        torch.backends.cudnn.benchmark = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # dice_val_best = 0.0
        # global_step_best = 0
        # epoch_loss_values = []
        # metric_values = []
        # val_loss_values = []

        while global_step < self.max_epoch:
            epoch_loss = self.train_one_epoch(train_loader, global_step)
            # Free GPU memory
            torch.cuda.empty_cache()
            epoch_iterator_val = tqdm(val_loader, desc="Validate", dynamic_ncols=True)
            dice_val, hausdorff_val, loss_val = self.validation(epoch_iterator_val)
            torch.cuda.empty_cache()
            epoch_iterator_val = None
            self.scheduler.step(loss_val)
            # self.epoch_loss_values.append(epoch_loss)
            train_info[global_step] = {
                'loss_val': loss_val,
                'metric_val': dice_val,
                'loss_train': epoch_loss,
            }
            with open(self.train_info_path, 'w') as file:
                json.dump(train_info, file, indent=4)
            global_step += 1
            # self.metric_values.append(dice_val)
            self.early_stopping(loss_val)
            print(f'Mean Hausdorff disatnce: {hausdorff_val}')
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                # self.global_step_best = global_step
                best_epochs_info[global_step] = dice_val_best
                with open(self.best_epoch_path, 'w') as file:
                    json.dump(best_epochs_info, file, indent=4)
                torch.save(self.model.state_dict(), self.model_path)
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            # early stopping
            if self.early_stopping.early_stop:
                break

    def plot_train_info(self):
        with open(self.train_info_path, "r") as file:
            train_info = json.load(file)
        plt.figure("train", (12, 6))

        plt.subplot(1, 2, 1)
        plt.title("Iteration Average Loss")
        x = list(train_info.keys())
        y = [v['loss_train'] for k, v in train_info.items()]
        plt.xticks(np.arange(0, len(x), 5))
        plt.xlabel("Epochs")
        plt.plot(x, y, label='Train')
        y = [v['loss_val'] for k, v in train_info.items()]
        plt.plot(x, y, label='Validation')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        y = [v['metric_val'] for k, v in train_info.items()]
        plt.xticks(np.arange(0, len(x), 5))
        plt.xlabel("Epochs")
        plt.plot(x, y)
        plt.show()

    def continue_train(self, train_loader, val_loader, max_epoch):
        self.load_best_model()
        train_info = {}
        global_step = 0
        dice_val_best = -1.0
        best_epochs_info = {}
        with open(self.train_info_path, "r") as file:
            train_info = json.load(file)
        with open(self.best_epoch_path, "r") as file:
            best_epochs_info = json.load(file)
        global_step = list(best_epochs_info.keys())[-1]
        dice_val_best = list(best_epochs_info.values())[-1]
        for k in train_info.keys():
            if k > global_step:
                del train_info[k]

        print(dice_val_best, global_step, train_info)

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.model_path))
        return self.model

    def evaluation(self, label, pred):
        all_surf = {}
        all_dice = {}
        for i in range(1, len(label)):
            surface_distances = sd.compute_surface_distances(mask_gt=label[i],
                                                             mask_pred=pred[i],
                                                             spacing_mm=[1.5, 1.5, 1.5])
            surf_dist = sd.compute_surface_dice_at_tolerance(surface_distances=surface_distances,
                                                             tolerance_mm=3)
            dice = sd.compute_dice_coefficient(mask_gt=label[i],
                                               mask_pred=pred[i])
            if not np.isnan(surf_dist) and surf_dist != 0:
                all_surf[i] = surf_dist
            if not np.isnan(dice) and dice != 0:
                all_dice[i] = dice
            # if dice == 0.0:
            #     print(i, np.any(label[i]), np.count_nonzero(label[i]), np.any(pred[i]),
            #           np.count_nonzero(pred[i]), label[i].shape)

        return all_surf, all_dice

    def test(self, test_loader):
        self.load_best_model()
        self.model.to(self.device)
        epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
        nsd_metric = {i: [] for i in range(1, self.number_of_classes)}
        dice_metric = {i: [] for i in range(1, self.number_of_classes)}
        dice_metric1 = DiceMetric(include_background=False, reduction='none', get_not_nans=False)
        model = self.model.eval()
        test_loss = []
        test_metrics = {}
        with torch.no_grad():
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if batch['label'].max() > 0:
                        test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                        with torch.cuda.amp.autocast():
                            #                 test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), num_samples, model, overlap=0.8)
                            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples,
                                                                    model, overlap=0.5, mode='gaussian'
                                                                    )
                            loss = self.loss_function(test_outputs.detach(), test_labels.detach()).item()
                        # print('label', torch.unique(test_labels, return_counts=True))
                        # print('pred', torch.unique(test_outputs, return_counts=True))
                        test_loss.append(loss)
                        # mask_gt = np.array([np.array(f.detach(), dtype=bool) for f in
                        #                     [self.post_label(test_label_tensor).detach().cpu() for test_label_tensor in
                        #                      decollate_batch(test_labels)]])
                        # mask_prep = np.array([np.array(f.detach(), dtype=bool) for f in
                        #                       [self.post_pred(test_pred_tensor).detach().cpu() for test_pred_tensor in
                        #                        decollate_batch(test_outputs)]])
                        mask_gt = \
                        [self.post_label(test_label_tensor).detach().cpu().astype(bool) for test_label_tensor in
                         decollate_batch(test_labels)][0]
                        mask_prep = [self.post_pred(test_pred_tensor).detach().cpu().astype(bool) for test_pred_tensor in
                                     decollate_batch(test_outputs)][0]
                        # dice_metric1(y_pred=[self.post_pred(test_pred_tensor).detach().cpu() for test_pred_tensor in
                        #                      decollate_batch(test_outputs)],
                        #              y=[self.post_label(test_label_tensor).detach().cpu() for test_label_tensor in
                        #                 decollate_batch(test_labels)])
                        # print(dice_metric1.aggregate(),
                        #       type([self.post_pred(test_pred_tensor).detach().cpu() for test_pred_tensor in
                        #             decollate_batch(test_outputs)]),
                        #       np.where([self.post_label(test_label_tensor).detach().cpu() for test_label_tensor in
                        #        decollate_batch(test_labels)][0][2].numpy().astype(bool) == True))
                        test_inputs, test_outputs = test_inputs.detach().cpu(), test_outputs.detach().cpu()
                        del test_inputs, test_outputs
                        test_labels = None
                        surf, dice = self.evaluation(label=mask_gt, pred=mask_prep)
                        for k, v in surf.items():
                            nsd_metric[k].append(v)
                        for k, v in dice.items():
                            dice_metric[k].append(v)
                        # print(dice)
                        # print(surf,  dice)
                        batch = batch["image"].detach().cpu(), batch["label"].detach().cpu()
                        del batch, mask_gt, mask_prep
                        # test_labels_convert, test_output_convert = None, None
                        # Free GPU memory
                        # print(torch.cuda.list_gpu_processes())
                        # gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()
                    else:
                        # print(batch['label'].max(), batch['label'].device)
                        batch['image'], batch['label'] = batch['image'].detach().cpu(), batch['label'].detach().cpu()
                        torch.cuda.empty_cache()
                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            class_nsd, class_nsd_std, class_dice, class_std = [], [], [], []
            # print(11)
            # n = len(dice_metric.keys())
            # print(nsd_metric, dice_metric)
            # for i in range(1, n + 1):
            for i in dice_metric.keys():
                # print(i)
                nsd_i = np.array(nsd_metric[i])
                dice_i = np.array(dice_metric[i])
                class_nsd.append(np.mean(nsd_i))
                class_nsd_std.append(np.nanstd(nsd_i, axis=0) * 100)
                class_dice.append(np.mean(dice_i))
                class_std.append(np.nanstd(dice_i, axis=0) * 100)
            # class_nsd = [np.mean(i) for i in nsd_metric.values()]
            # class_dice = [np.mean(i) for i in dice_metric.values()]
            # class_std = [np.nanstd(i, axis=0) * 100 for i in dice_metric.values()]
            # class_nsd_std = [np.nanstd(i, axis=0) * 100 for i in nsd_metric.values()]

            test_loss_mean = np.nanmean(np.nan_to_num(np.array(test_loss),
                                                      nan=np.nan, posinf=np.nan, neginf=np.nan))
            test_metrics = {
                'dice': {i + 1: f'{class_dice[i]} +- {class_std[i]}' for i in range((len(class_dice)))},
                'NSD': {i + 1: f'{class_nsd[i]} +- {class_nsd_std[i]}' for i in range((len(class_nsd)))},
            }
            # print(test_metrics)
            with open(self.test_info_path, 'w') as file:
                json.dump(test_metrics, file, indent=4)
        return class_dice, class_std, class_nsd, class_nsd_std, test_loss_mean

    def test_multi_device(self, test_loader):
        self.load_best_model()
        self.model.to(self.device)
        epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
        nsd_metric = {i: [] for i in range(1, self.number_of_classes)}
        dice_metric = {i: [] for i in range(1, self.number_of_classes)}
        model = self.model.eval()
        test_loss = []
        test_metrics = {}
        with torch.no_grad():
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if batch['label'].max() > 0:
                        t = batch['label']
                        needed_memory = (t.element_size() * t.numel() * (self.number_of_classes + 1)) / (1024 ** 3)
                        if needed_memory < 5.5:
                            test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                            self.model.to(self.device)
                        else:
                            test_inputs, test_labels = (batch["image"].cpu(), batch["label"].cpu())
                            self.model.to("cpu")
                            print(test_inputs.shape, 'cpu')
                        with torch.cuda.amp.autocast():
                            #                 test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), num_samples, model, overlap=0.8)
                            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples, model)
                            loss = self.loss_function(test_outputs.detach(), test_labels.detach()).item()

                        test_loss.append(loss)
                        mask_gt = np.array([np.array(f.detach(), dtype=bool) for f in
                                            [self.post_label(test_label_tensor).detach().cpu() for test_label_tensor in
                                             decollate_batch(test_labels)]])
                        mask_prep = np.array([np.array(f.detach(), dtype=bool) for f in
                                              [self.post_pred(test_pred_tensor).detach().cpu() for test_pred_tensor in
                                               decollate_batch(test_outputs)]])
                        test_inputs, test_outputs = test_inputs.detach().cpu(), test_outputs.detach().cpu()
                        del test_inputs, test_outputs
                        test_labels = None
                        surf, dice = self.evaluation(label=mask_gt[0], pred=mask_prep[0])
                        for k, v in surf.items():
                            nsd_metric[k].append(v)
                        for k, v in dice.items():
                            dice_metric[k].append(v)
                        # print(surf,  dice)
                        batch = batch["image"].detach().cpu(), batch["label"].detach().cpu()
                        del batch, mask_gt, mask_prep
                        # test_labels_convert, test_output_convert = None, None
                        # Free GPU memory
                        # print(torch.cuda.list_gpu_processes())
                        # gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()
                    else:
                        # print(batch['label'].max(), batch['label'].device)
                        batch['image'], batch['label'] = batch['image'].detach().cpu(), batch['label'].detach().cpu()
                        torch.cuda.empty_cache()
                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            class_nsd, class_nsd_std, class_dice, class_std = [], [], [], []
            # print(11)
            # n = len(dice_metric.keys())
            # print(nsd_metric, dice_metric)
            # for i in range(1, n + 1):
            for i in dice_metric.keys():
                # print(i)
                nsd_i = np.array(nsd_metric[i])
                dice_i = np.array(dice_metric[i])
                class_nsd.append(np.mean(nsd_i))
                class_nsd_std.append(np.nanstd(nsd_i, axis=0) * 100)
                class_dice.append(np.mean(dice_i))
                class_std.append(np.nanstd(dice_i, axis=0) * 100)
            # class_nsd = [np.mean(i) for i in nsd_metric.values()]
            # class_dice = [np.mean(i) for i in dice_metric.values()]
            # class_std = [np.nanstd(i, axis=0) * 100 for i in dice_metric.values()]
            # class_nsd_std = [np.nanstd(i, axis=0) * 100 for i in nsd_metric.values()]

            test_loss_mean = np.nanmean(np.nan_to_num(np.array(test_loss),
                                                      nan=np.nan, posinf=np.nan, neginf=np.nan))
            test_metrics = {
                'dice': {i + 1: f'{class_dice[i]} +- {class_std[i]}' for i in range((len(class_dice)))},
                'NSD': {i + 1: f'{class_nsd[i]} +- {class_nsd_std[i]}' for i in range((len(class_nsd)))},
            }
            # print(test_metrics)
            with open(self.test_info_path, 'w') as file:
                json.dump(test_metrics, file, indent=4)
        return class_dice, class_std, class_nsd, class_nsd_std, test_loss_mean

    def test_new(self, test_loader):
        self.load_best_model()
        self.model.to(self.device)
        epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
        nsd_metric = {i: [] for i in range(1, self.number_of_classes)}
        dice_metric = {i: [] for i in range(1, self.number_of_classes)}
        model = self.model.eval()
        test_loss = []
        test_metrics = {}
        with torch.no_grad():
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    t = batch['label']
                    needed_memory = (t.element_size() * t.numel() * (self.number_of_classes + 1)) / (1024 ** 3)
                    if batch['label'].max() > 0 and needed_memory < 4.5:
                        test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                        with torch.cuda.amp.autocast():
                            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples,
                                                                    model, overlap=0.5, mode='gaussian'
                                                                    )
                            loss = self.loss_function(test_outputs.detach(), test_labels.detach()).item()

                        test_loss.append(loss)
                        mask_gt = \
                        [self.post_label(test_label_tensor).detach().cpu().astype(bool) for test_label_tensor in
                         decollate_batch(test_labels)][0]
                        mask_prep = [self.post_pred(test_pred_tensor).detach().cpu().astype(bool) for test_pred_tensor in
                                     decollate_batch(test_outputs)][0]
                        test_inputs, test_outputs = test_inputs.detach().cpu(), test_outputs.detach().cpu()
                        del test_inputs, test_outputs
                        test_labels = None
                        surf, dice = self.evaluation(label=mask_gt, pred=mask_prep)
                        for k, v in surf.items():
                            nsd_metric[k].append(v)
                        for k, v in dice.items():
                            dice_metric[k].append(v)
                        batch = batch["image"].detach().cpu(), batch["label"].detach().cpu()
                        del batch, mask_gt, mask_prep
                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()
                    else:
                        test_inputs = [batch["image"][:, :, :, :, :301], batch["image"][:, :, :, :, 301:]]

                        test_labels = [batch["label"][:, :, :, :, :301], batch["label"][:, :, :, :, 301:]]
                        res = []
                        for ti, tl in zip(test_inputs, test_labels):
                            ti, tl = ti.cuda(), tl.cuda()
                            with torch.cuda.amp.autocast():
                                test_outputs = sliding_window_inference(ti, (96, 96, 96), self.num_samples,
                                                                        model, overlap=0.5, mode='gaussian'
                                                                        )
                            res.append(test_outputs.detach().cpu())
                            ti, tl = ti.detach().cpu(), tl.detach().cpu()
                            del ti, tl
                            # batch = batch["image"].detach().cpu(), batch["label"].detach().cpu()

                        test_labels, test_outputs = torch.cat((test_labels[0].to('cpu'), test_labels[1].to('cpu')),
                                                              dim=-1), \
                                                    torch.cat((res[0], res[1]), dim=-1)
                        loss = self.loss_function(test_outputs, test_labels).item()

                        test_loss.append(loss)
                        mask_gt = \
                        [self.post_label(test_label_tensor).detach().cpu().astype(bool) for test_label_tensor in
                         decollate_batch(test_labels)][0]
                        mask_prep = [self.post_pred(test_pred_tensor).detach().cpu().astype(bool) for test_pred_tensor in
                                     decollate_batch(test_outputs)][0]
                        surf, dice = self.evaluation(label=mask_gt, pred=mask_prep)
                        test_labels, test_outputs = None, None
                        for k, v in surf.items():
                            nsd_metric[k].append(v)
                        for k, v in dice.items():
                            dice_metric[k].append(v)
                        del mask_gt, mask_prep

                        torch.cuda.empty_cache()
                        torch.cuda.reset_max_memory_allocated()

                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            class_nsd, class_nsd_std, class_dice, class_std = [], [], [], []

            for i in dice_metric.keys():
                # print(i)
                nsd_i = np.array(nsd_metric[i])
                dice_i = np.array(dice_metric[i])
                class_nsd.append(np.mean(nsd_i))
                class_nsd_std.append(np.nanstd(nsd_i, axis=0) * 100)
                class_dice.append(np.mean(dice_i))
                class_std.append(np.nanstd(dice_i, axis=0) * 100)

            test_loss_mean = np.nanmean(np.nan_to_num(np.array(test_loss),
                                                      nan=np.nan, posinf=np.nan, neginf=np.nan))
            test_metrics = {
                'dice': {i + 1: f'{class_dice[i]} +- {class_std[i]}' for i in range((len(class_dice)))},
                'NSD': {i + 1: f'{class_nsd[i]} +- {class_nsd_std[i]}' for i in range((len(class_nsd)))},
            }
            with open(self.test_info_path, 'w') as file:
                json.dump(test_metrics, file, indent=4)
        return class_dice, class_std, class_nsd, class_nsd_std, test_loss_mean

    def test_old(self, test_loader):
        self.load_best_model()
        self.model.to(self.device)
        epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
        # dice_metric = DiceMetric(include_background=False, reduction='none', get_not_nans=False)
        # nsd_metric = SurfaceDiceMetric(class_thresholds=[3.0 for i in range(self.number_of_classes)],
        #                                include_background=False, reduction='none', get_not_nans=False)
        nsd_metric = {i: [] for i in range(1, 25)}
        dice_metric = {i: [] for i in range(1, 25)}
        model = self.model.eval()
        test_loss = []
        test_metrics = {}
        with torch.no_grad():
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if batch['image'].max() > 0:
                        test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                        with torch.cuda.amp.autocast():
                            #                 test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), num_samples, model, overlap=0.8)
                            test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples, model)
                        test_inputs = None
                        loss = self.loss_function(test_outputs, test_labels).item()
                        test_loss.append(loss)
                        # test_labels_convert = [self.post_label(test_label_tensor).detach() for test_label_tensor in
                        #                        decollate_batch(test_labels)]

                        # test_output_convert = [self.post_pred(test_pred_tensor).detach() for test_pred_tensor in
                        #                        decollate_batch(test_outputs)]
                        # print(test_labels_convert[0].shape, test_output_convert[0].shape)
                        # dice_metric(
                        #     y_pred=test_output_convert,
                        #     y=test_labels_convert)
                        # nsd_metric(y_pred=test_output_convert,
                        #            y=test_labels_convert)
                        mask_gt = np.array([np.array(f.detach().cpu(), dtype=bool) for f in
                                            [self.post_label(test_label_tensor).detach().cpu() for test_label_tensor in
                                             decollate_batch(test_labels)]])
                        mask_prep = np.array([np.array(f.detach(), dtype=bool) for f in
                                              [self.post_pred(test_pred_tensor).detach().cpu() for test_pred_tensor in
                                               decollate_batch(test_outputs)]])
                        test_labels = None
                        # # print(mask_gt.shape, test_output_convert[0].shape)
                        surf, dice = self.evaluation(label=mask_gt[0], pred=mask_prep[0])
                        # print(surf)
                        for k, v in surf.items():
                            nsd_metric[k].append(v)
                        for k, v in dice_metric.items():
                            dice_metric[k].append(v)
                        batch = batch["image"].cpu(), batch["label"].cpu()
                        batch = None
                        mask_gt = None
                        mask_prep = None
                        test_labels_convert, test_output_convert = None, None
                        # Free GPU memory
                        torch.cuda.empty_cache()
                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            class_nsd = [np.mean(i) for i in nsd_metric.values()]
            class_dice = [np.mean(i) for i in dice_metric.values()]
            class_std = [np.nanstd(i, axis=0) * 100 for i in dice_metric.values()]
            class_nsd_std = [np.nanstd(i, axis=0) * 100 for i in nsd_metric.values()]

            test_loss_mean = np.nanmean(np.nan_to_num(np.array(test_loss),
                                                      nan=np.nan, posinf=np.nan, neginf=np.nan))
            test_metrics = {
                'dice': {i + 1: f'{class_dice[i]} +- {class_std[i]}' for i in range((len(class_dice)))},
                'NSD': {i + 1: f'{class_nsd[i]} +- {class_nsd_std[i]}' for i in range((len(class_nsd)))},
            }
            with open(self.test_info_path, 'w') as file:
                json.dump(test_metrics, file, indent=4)
        return class_dice, class_std, class_nsd, class_nsd_std, test_loss_mean

    def multiple_test(self, file_list_test, task_name, val_transforms):
        self.load_best_model()
        self.model.to(self.device)
        test_ds = PersistentDataset(
            data=file_list_test[:35],
            transform=val_transforms,
            cache_dir=f'test_{task_name}'
            #     cache_dir='C:/Training/val'
        )

        test_loader = DataLoader(test_ds, num_workers=0, batch_size=1,
                                 collate_fn=lambda x: pad_list_data_collate(x, pad_to_shape=(96, 96, 96)))
        epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
        nsd_metric = {i: [] for i in range(1, 25)}
        dice_metric = {i: [] for i in range(1, 25)}
        model = self.model.eval()
        test_loss = []
        test_metrics = {}
        with torch.no_grad():
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                if batch['image'].max() > 0:
                    test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                    with torch.cuda.amp.autocast():
                        #                 test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), num_samples, model, overlap=0.8)
                        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples, model)
                    test_inputs = None
                    loss = self.loss_function(test_outputs, test_labels).item()
                    test_loss.append(loss)
                    mask_gt = np.array([np.array(f.cpu(), dtype=bool) for f in
                                        [self.post_label(test_label_tensor).cpu() for test_label_tensor in
                                         decollate_batch(test_labels)]])
                    mask_prep = np.array([np.array(f.cpu(), dtype=bool) for f in
                                          [self.post_pred(test_pred_tensor).cpu() for test_pred_tensor in
                                           decollate_batch(test_outputs)]])
                    test_labels = None
                    surf, dice = self.evaluation(label=mask_gt[0], pred=mask_prep[0])
                    for k, v in surf.items():
                        nsd_metric[k].append(v)
                    for k, v in dice_metric.items():
                        dice_metric[k].append(v)
                    batch = batch["image"].cpu(), batch["label"].cpu()
                    batch = None
                    mask_gt = None
                    mask_prep = None
                    test_labels_convert, test_output_convert = None, None
                    # Free GPU memory
                    torch.cuda.empty_cache()
                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            test_ds = PersistentDataset(
                data=file_list_test[35:],
                transform=val_transforms,
                cache_dir=f'test_{task_name}'
                #     cache_dir='C:/Training/val'
            )

            torch.cuda.empty_cache()
            test_loader = DataLoader(test_ds, num_workers=0, batch_size=1,
                                     collate_fn=lambda x: pad_list_data_collate(x, pad_to_shape=(96, 96, 96)))
            epoch_iterator_test = tqdm(test_loader, desc="Test", dynamic_ncols=True)
            for batch in epoch_iterator_test:
                # Free GPU memory
                torch.cuda.empty_cache()
                if batch['image'].max() > 0:
                    test_inputs, test_labels = (batch["image"].cuda(), batch["label"].cuda())
                    with torch.cuda.amp.autocast():
                        #                 test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), num_samples, model, overlap=0.8)
                        test_outputs = sliding_window_inference(test_inputs, (96, 96, 96), self.num_samples, model)
                    test_inputs = None
                    loss = self.loss_function(test_outputs, test_labels).item()
                    test_loss.append(loss)
                    mask_gt = np.array([np.array(f.cpu(), dtype=bool) for f in
                                        [self.post_label(test_label_tensor).cpu() for test_label_tensor in
                                         decollate_batch(test_labels)]])
                    mask_prep = np.array([np.array(f.cpu(), dtype=bool) for f in
                                          [self.post_pred(test_pred_tensor).cpu() for test_pred_tensor in
                                           decollate_batch(test_outputs)]])
                    test_labels = None
                    surf, dice = self.evaluation(label=mask_gt[0], pred=mask_prep[0])
                    for k, v in surf.items():
                        nsd_metric[k].append(v)
                    for k, v in dice_metric.items():
                        dice_metric[k].append(v)
                    batch = batch["image"].cpu(), batch["label"].cpu()
                    batch = None
                    mask_gt = None
                    mask_prep = None
                    test_labels_convert, test_output_convert = None, None
                    # Free GPU memory
                    torch.cuda.empty_cache()
                epoch_iterator_test.set_description(f"Test (loss={loss:2.5f})")

            class_nsd = [np.mean(i) for i in nsd_metric.values()]
            class_dice = [np.mean(i) for i in dice_metric.values()]
            class_std = [np.nanstd(i, axis=0) * 100 for i in dice_metric.values()]
            class_nsd_std = [np.nanstd(i, axis=0) * 100 for i in nsd_metric.values()]

            test_loss_mean = np.nanmean(np.nan_to_num(np.array(test_loss),
                                                      nan=np.nan, posinf=np.nan, neginf=np.nan))
            test_metrics = {
                'dice': {i + 1: f'{class_dice[i]} +- {class_std[i]}' for i in range((len(class_dice)))},
                'NSD': {i + 1: f'{class_nsd[i]} +- {class_nsd_std[i]}' for i in range((len(class_nsd)))},
            }
            with open(self.test_info_path, 'w') as file:
                json.dump(test_metrics, file, indent=4)
        return class_dice, class_std, class_nsd, class_nsd_std, test_loss_mean
