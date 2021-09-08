import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description='Parameters for grad-cam.')

    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--initializer',
                        default='kaiming_uniform_')
    parser.add_argument('--n_class', default=2)

    parser.add_argument('--logs_dir', default='./logs')
    parser.add_argument('--exp_name', default='exp1')

    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer',
                        default='SGD')
    parser.add_argument('--lr',
                        default=0.03)
    parser.add_argument('--weight_decay',
                        default=0.0001)
    parser.add_argument('--optimizer_momentum',
                        default=0.9)

    args = parser.parse_args()
    return args