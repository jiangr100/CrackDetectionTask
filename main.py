import torch
import os
from pathlib import Path
import pickle
import torchvision.models as torch_models
import torch.nn.init as inits
import torch.nn as nn
import matplotlib.pyplot as plt

from hyperparameters import *
from models import *
from Dataset import *

import wandb


def train(model, train_loader, validation_loader, optimizer, args):
    min_val_loss = float('inf')
    for e in range(args.max_num_epochs):

        for (img, y) in train_loader:
            img = img.to(args.device)
            y = y.to(args.device)

            saliency_map, logit = model(img)
            loss = model.loss_func(model.softmax(logit))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # wandb.log({"train_loss": loss})

        val_loss = validation(model, validation_loader, args)
        # wandb.log({"val_loss": val_loss})

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            if not os.path.exists(Path('./Checkpoints/%s' % args.exp_name)):
                os.makedirs(Path('./Checkpoints/%s' % args.exp_name))
            torch.save(model.state_dict(), Path('Checkpoints/%s/%s_%f.pt' % (args.exp_name, args.model_save_name, min_val_loss)))


def validation(model, validation_loader, args):
    average_loss = 0
    num_batches = 0
    for (img, y) in validation_loader:
        img = img.to(args.device)
        y = y.to(args.device)

        saliency_map, logit = model(img)
        loss = model.loss_func(model.softmax(logit))

        average_loss += loss.item()
        num_batches += 1

    return average_loss / num_batches


if __name__ == '__main__':
    args = get_arguments()
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
        weight_decay=args.weight_decay,
        momentum=args.optimizer_momentum
    )

    dataset = CrackPatches(
        positive_root_dir=str(Path('D:\img_data_files\cropped_images1')),
        negative_root_dir=str(Path('D:\img_data_files\cropped_images_empty1')),
        transform=None
    )

    train_loader, validation_loader, test_loader = create_dataloader(
        args=args,
        dataset=dataset,
        test_split=0.2,
        validation_split=0.01
    )

    train(model, train_loader, validation_loader, optimizer, args)

    print(model.target_layer)
