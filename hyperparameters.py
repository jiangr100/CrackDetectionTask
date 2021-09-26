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

    parser.add_argument('--max_num_epochs',
                        default=100)
    parser.add_argument('--batch_size',
                        default=32)
    parser.add_argument('--random_seed',
                        default=12345678)
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

    parser.add_argument('--train_mode',
                        default=True)
    parser.add_argument('--model_save_name',
                        default='Checkpoint')
    parser.add_argument('--model_loss',
                        default='')

    args = parser.parse_args()
    return args