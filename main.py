import torch
import os
from pathlib import Path
import pickle
import torchvision.models as torch_models
import torch.nn.init as inits
import torch.nn as nn

from hyperparameters import *
from models import *


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

    print(model.target_layer)
