import torch
import os
from pathlib import Path
import pickle
import cv2
import torchvision.models as torch_models
import torch.nn.init as inits
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.nn import Sigmoid

from hyperparameters import *
from models import *
from Dataset import *

import wandb


def train(model, train_loader, validation_loader, optimizer, args):
    wandb.init(project=args.exp_name, dir='C:\\Users\\Mai\\CrackDetectionWandB')
    wandb.watch(model)

    min_val_loss = float('inf')
    for e in range(args.max_num_epochs):

        for data in train_loader:
            img = data['img'].to(args.device)
            y = data['y'].to(args.device)

            out = model.Sigmoid(model(img))
            loss = model.loss_func(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss})

        val_loss, val_acc = validation(model, validation_loader, args, e)
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_acc": val_acc})
        wandb.log({'#epoch': e})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if not os.path.exists(Path('./Checkpoints/%s' % args.exp_name)):
                os.makedirs(Path('./Checkpoints/%s' % args.exp_name))
            torch.save(model.state_dict(), Path('Checkpoints/%s/%s_%f.pt' % (args.exp_name, args.model_save_name, min_val_loss)))


def validation(model, validation_loader, args, e):
    average_loss = 0
    num_batches = 0
    acc = 0
    num_pairs = 0
    for data in validation_loader:
        img = data['img'].to(args.device)
        y = data['y'].to(args.device)

        out = model.Sigmoid(model(img).detach())
        loss = model.loss_func(out, y)

        pred = torch.where(out > args.positive_threshold, 1, 0)
        acc += sum(sum(pred * y.detach()))
        num_pairs += sum(sum(y.detach()))

        average_loss += loss.item()
        num_batches += 1

    return average_loss / num_batches, acc*1.0 / num_pairs


def test(model, test_loader, args):
    tp = torch.zeros((args.n_crack_types))
    fp = torch.zeros((args.n_crack_types))
    tn = torch.zeros((args.n_crack_types))
    fn = torch.zeros((args.n_crack_types))
    num_pairs = 0
    for data in test_loader:
        img = data['img'].to(args.device)
        y = data['y'].to(args.device)

        out = model.Sigmoid(model(img).detach())

        pred = torch.where(out > args.positive_threshold, 1, 0).cpu()
        gt = y.detach().cpu()
        pred_rev = torch.where(pred == 0, 1, 0)
        gt_rev = torch.where(gt == 0, 1, 0)

        tp += torch.sum(pred * gt, dim=0)
        fp += torch.sum(pred * gt_rev, dim=0)
        tn += torch.sum(pred_rev * gt_rev, dim=0)
        fn += torch.sum(pred_rev * gt, dim=0)

        # print(tp, fp, tn, fn)

    acc = (tp + tn) / (tn + fn + tp + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 / (1 / recall + 1 / precision)

    return acc, recall, precision, f1


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


if __name__ == '__main__':
    args = get_arguments()
    if args.train_mode:
        args_dict = vars(args)
        if not os.path.exists(Path(args.logs_dir + '/' + args.exp_name)):
            os.makedirs(Path(args.logs_dir + '/' + args.exp_name))
        with open(Path(args.logs_dir + '/' + args.exp_name + '/' + 'hp.pl'), 'wb') as f:
            pickle.dump(args_dict, f)

    model = Gradcam(
        encoder_model=torch_models.__dict__[args.arch],
        initializer=inits.__dict__[args.initializer],
        n_class=args.n_crack_types,
        input_channels=3
    )
    print("GPU count: ", torch.cuda.device_count())
    # if torch.cuda.device_count() > 1:
    #     model = MocoDataParallel(model)
    model.to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.__dict__[args.optimizer](
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    random_aug = torch.nn.Sequential(
        RandomApply(
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        RandomApply(
            transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        )
    )

    data_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((509, 625)),
        transforms.ToTensor(),
        # random_aug,
        # transforms.Normalize(mean=args.normalization_mean, std=args.normalization_std)
    ])
    dataset = CrackClassification(
        data_dir=str(Path('D:\\pickle_files')),
        transform=data_transform,
        mode='crack_type',
        augmented_data_dir=str(Path('D:\\image_augmentations')),
    )

    train_loader, validation_loader, test_loader = create_dataloader(
        args=args,
        dataset=dataset,
        test_split=0.05,
        validation_split=0.01,
        with_weight=True
    )

    if args.overfitting_test:
        of_train_loader, _, _ = train_loader, validation_loader, test_loader = create_dataloader(
            args=args,
            dataset=dataset,
            test_split=0.98,
            validation_split=0.01
        )
        train(model, of_train_loader, of_train_loader, optimizer, args)
        of_accuracy = test(model, of_train_loader, args)
        print("overfitting accuracy is : ", of_accuracy)
        exit(0)

    if args.load_pretrained_model:
        if os.path.isfile(Path(
                'Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))):
            model.load_state_dict(torch.load(Path(
                'Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))))
        else:
            print("No pretrained model found!")
            exit(0)

    if args.train_mode:
        train(model, train_loader, validation_loader, optimizer, args)

    if args.show_test_result:
        acc, recall, precision, f1 = test(model, test_loader, args)
        print("test accuracy is : ", acc)
        print("test recall is : ", recall)
        print("test precision is : ", precision)
        print("test f1 score is : ", f1)








