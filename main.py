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
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def train(model, train_loader, validation_loader, optimizer, args):
    min_val_loss = float('inf')
    for e in range(args.max_num_epochs):

        for data in train_loader:
            img = data['img'].to(args.device)
            y = data['y'].to(args.device)

            out = model(img)
            loss = model.loss_func(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss})

        val_loss, val_acc = validation(model, validation_loader, args, e)
        wandb.log({"val_loss": val_loss})
        wandb.log({"val_acc": val_acc})

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

        out = model(img)
        loss = model.loss_func(out, y)

        pred = torch.argmax(model.softmax(out), dim=1)
        acc += sum(torch.eq(pred, y).item())
        num_pairs += len(pred.item())

        average_loss += loss.item()
        num_batches += 1

    return average_loss / num_batches, acc*1.0 / num_pairs


def test(model, test_loader, args):
    acc = 0
    num_pairs = 0
    for data in test_loader:
        img = data['img'].to(args.device)
        y = data['y'].to(args.device)

        out = model.softmax(model(img))

        pred = torch.argmax(out, dim=1)

        acc += sum(torch.eq(pred, y).item())
        num_pairs += len(pred.item())

    return acc / num_pairs


def calculate_cam(model, img, args, file_path):
    target_layers = [model.layer4[-1]]

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4[-1]]

    input_tensor = img.unsqueeze(0)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:
        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            target_category=target_category,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(img.detach().cpu().permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    if file_path is not None:
        cv2.imwrite(str(Path(file_path + '_cam.jpg')), cam_image)
        cv2.imwrite(str(Path(file_path + '_gb.jpg')), gb)
        cv2.imwrite(str(Path(file_path + '_cam_gb.jpg')), cam_gb)

    return cam_image, gb, cam_gb


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
        n_class=args.n_class
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

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=args.normalization_mean, std=args.normalization_std)
    ])
    dataset = CrackPatches(
        crack_dir=str(Path('D:\img_data_files\crack')),
        pothole_dir=str(Path('D:\img_data_files\pothole')),
        empty_dir=str(Path('D:\img_data_files\empty')),
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

    if args.train_mode:
        train(model, train_loader, validation_loader, optimizer, args)
    elif os.path.isfile(Path('Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))):
        model.load_state_dict(torch.load(Path('Checkpoints\\%s\\%s' % (args.load_exp_name, args.model_save_name + '_' + args.model_loss + '.pt'))))
    else:
        print("No pretrained model found!")
        exit(0)

    if args.show_test_result:
        accuracy = test(model, test_loader, args)
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
        test_image_path = str(Path('D:\\dataset\\images\\test\\MNM-0001-10-0010-0169 ACD 0002477.jpg'))
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
                if pred[0] == 0:
                    continue

                # print(i)

                cam, gb, gb_cam = calculate_cam(model.net, patch, args, file_path=None)

                res_gb_patch = res_gb[i_patch:i_patch+args.patch_size, j_patch:j_patch+args.patch_size, :]
                res_gb_cam_patch = res_gb_cam[i_patch:i_patch+args.patch_size, j_patch:j_patch+args.patch_size, :]

                new_res_gb_patch = np.where(gb > res_gb_patch, gb, res_gb_patch)
                new_res_gb_cam_patch = np.where(gb_cam > res_gb_cam_patch, gb_cam, res_gb_cam_patch)

                res_gb[i_patch:i_patch + args.patch_size, j_patch:j_patch + args.patch_size, :] = new_res_gb_patch
                res_gb_cam[i_patch:i_patch + args.patch_size, j_patch:j_patch + args.patch_size, :] = new_res_gb_cam_patch

                i += 1

        if not os.path.exists(str(Path('./final_result/%s' % args.exp_name))):
            os.makedirs(str(Path('./final_result/%s' % args.exp_name)))

        cv2.imwrite(str(Path('./final_result/%s/gb.jpg' % args.exp_name)), res_gb)
        cv2.imwrite(str(Path('./final_result/%s/gb_cam.jpg' % args.exp_name)), res_gb_cam)

    gb_img = str(Path('./final_result/exp13/gb.jpg'))
    rgb_img = cv2.imread(gb_img, 1)[:, :, ::-1]
    rgb_img = rgb_img[:2480, :1980]

    img_norm = cv2.normalize(rgb_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    img_norm = np.where(img_norm < 100, 0, img_norm)
    cv2.imwrite(str(Path('./final_result/%s/gb_norm.jpg' % args.exp_name)), img_norm)







