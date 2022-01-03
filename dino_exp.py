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

from hyperparameters import *
from models import *
from Dataset import *

import wandb

from dino import Dino
from vit import ViT


def train(model, learner, train_loader, validation_loader, optimizer, lr_scheduler, args):
    min_val_loss = float('inf')
    for e in range(args.max_num_epochs):

        current_teacher_temp = 0.04 + (0.001 * e if e < 30 else 0.03)
        for data in train_loader:
            img = data['img'].to(args.device)

            loss = learner(img, teacher_temp=current_teacher_temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

            wandb.log({"train_loss": loss.item()})
        lr_scheduler.step()

        val_loss = validation(learner, validation_loader)
        wandb.log({"val_loss": val_loss, "#epoch": e})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if not os.path.exists(Path('./Checkpoints/%s' % args.exp_name)):
                os.makedirs(Path('./Checkpoints/%s' % args.exp_name))
            torch.save(model.state_dict(), Path('Checkpoints/%s/%s_%f.pt' % (args.exp_name, args.model_save_name, min_val_loss)))


def validation(learner, validation_loader):
    average_loss = 0
    num_batches = 0
    for data in validation_loader:
        img = data['img'].to(args.device)
        loss = learner(img)

        average_loss += loss.item()
        num_batches += 1

    return average_loss / num_batches


def test(learner, test_loader):
    average_loss = 0
    num_batches = 0
    for data in test_loader:
        img = data['img'].to(args.device)
        loss = learner(img)

        average_loss += loss.item()
        num_batches += 1

    return average_loss / num_batches


if __name__ == '__main__':
    args = get_arguments()
    if args.train_mode:
        args_dict = vars(args)
        if not os.path.exists(Path(args.logs_dir + '/' + args.exp_name)):
            os.makedirs(Path(args.logs_dir + '/' + args.exp_name))
        with open(Path(args.logs_dir + '/' + args.exp_name + '/' + 'hp.pl'), 'wb') as f:
            pickle.dump(args_dict, f)

    print("GPU count: ", torch.cuda.device_count())

    model = ViT(
        image_size=256,
        patch_size=16,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048
    ).to(args.device)

    learner = Dino(
        model,
        image_size=256,
        hidden_layer='to_latent',  # hidden layer name or index, from which to extract the embedding
        projection_hidden_size=256,  # projector network hidden dimension
        projection_layers=4,  # number of layers in projection network
        num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
        student_temp=0.1,  # student temperature
        teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
        local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
        global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
        moving_average_decay=0.999,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
        center_moving_average_decay=0.999,
        # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
    )

    optimizer = torch.optim.__dict__[args.optimizer](
        params=model.parameters(),
        lr=args.lr * args.batch_size / 256,
        weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=200
    )

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.normalization_mean, std=args.normalization_std)
    ])
    dataset = CrackPatches(
        crack_dir=str(Path('D:\img_data_files\crack128')),
        pothole_dir=None,
        empty_dir=None,
        transform=data_transform
    )

    train_loader, validation_loader, test_loader = create_dataloader(
        args=args,
        dataset=dataset,
        test_split=0.2,
        validation_split=0.01
    )

    wandb.init(project=args.exp_name)
    wandb.watch(model)

    if args.train_mode:
        train(model, learner, train_loader, validation_loader, optimizer, lr_scheduler, args)
    elif os.path.isfile(Path('Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))):
        model.load_state_dict(torch.load(Path('Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))))
    else:
        print("No pretrained model found!")
        exit(0)

    if args.show_test_result:
        accuracy = test(learner, test_loader)
        print("test accuracy is : ", accuracy)

    if args.show_cam:
        if not os.path.exists(str(Path('./cam_results/%s/' % args.exp_name))):
            os.makedirs(str(Path('./cam_results/%s/' % args.exp_name)))

        img = next(iter(test_loader))['img'].to(args.device)
        y = next(iter(test_loader))['y'].to(args.device)

        for i in range(img.shape[0]):
            if not y[i]:
                continue

            plt.imshow(transforms.ToPILImage()(img[i]))
            plt.savefig(str(Path('./cam_results/%s/img_%04d' % (args.exp_name, i))))

            save_path = str(Path('./cam_results/%s/%d' % (args.exp_name, i)))
            calculate_cam(model.net, img[i], args, save_path)

    if args.generate_gb:
        test_image_path = str(Path('D:\\raw data\\pics\\DB-0001-05-0010-0170-200420-PAVEMENT\\MNM-0001-05-0010-0171 ACD 0001551.jpg'))
        rgb_img = cv2.imread(test_image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        img_tensor = data_transform(rgb_img)
        h, w = img_tensor.shape[1], img_tensor.shape[2]

        res_gb = np.zeros((h, w, 3))
        res_gb_cam = np.zeros((h, w, 3))
        i = 0
        # while i < args.num_patches:

        for i_patch in range(0, h-args.patch_size, int(args.patch_size/4)):
            for j_patch in range(0, w-args.patch_size, int(args.patch_size/4)):
                # i_patch = random.randint(0, h-args.patch_size)
                # j_patch = random.randint(0, w-args.patch_size)

                patch = img_tensor[:, i_patch:i_patch+args.patch_size, j_patch:j_patch+args.patch_size].to(args.device)
                out = model.softmax(model(patch.unsqueeze(dim=0)))
                pred = torch.argmax(out, dim=1)
                if not pred[0] == 2:
                    continue

                # print(i)

                cam, gb, gb_cam = calculate_cam(model.net, patch, args, file_path=None)

                res_gb_patch = res_gb[i_patch:i_patch+args.patch_size, j_patch:j_patch+args.patch_size, :]
                res_gb_cam_patch = res_gb_cam[i_patch:i_patch+args.patch_size, j_patch:j_patch+args.patch_size, :]

                # gb = np.where((gb > 210) | (gb < 80), 255, 0)
                # gb_cam = np.where((gb_cam > 210) | (gb_cam < 80), 255, 0)

                new_res_gb_patch = np.where(gb > res_gb_patch, gb, res_gb_patch)
                new_res_gb_cam_patch = np.where(gb_cam > res_gb_cam_patch, gb_cam, res_gb_cam_patch)

                res_gb[i_patch:i_patch + args.patch_size, j_patch:j_patch + args.patch_size, :] = new_res_gb_patch
                res_gb_cam[i_patch:i_patch + args.patch_size, j_patch:j_patch + args.patch_size, :] = new_res_gb_cam_patch

                i += 1

        if not os.path.exists(str(Path('./final_result/%s' % args.exp_name))):
            os.makedirs(str(Path('./final_result/%s' % args.exp_name)))

        cv2.imwrite(str(Path('./final_result/%s/gb.jpg' % args.exp_name)), res_gb)
        cv2.imwrite(str(Path('./final_result/%s/gb_cam.jpg' % args.exp_name)), res_gb_cam)

    '''
    gb_img = str(Path('./final_result/exp13/gb.jpg'))
    rgb_img = cv2.imread(gb_img, 1)[:, :, ::-1]
    rgb_img = rgb_img[:2480, :1980]

    img_norm = cv2.normalize(rgb_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    img_norm = np.where(img_norm < 100, 0, img_norm)
    cv2.imwrite(str(Path('./final_result/%s/gb_norm.jpg' % args.exp_name)), img_norm)
    '''







