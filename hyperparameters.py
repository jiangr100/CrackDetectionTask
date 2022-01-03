import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser(description='Parameters for grad-cam.')

    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--initializer',
                        default='kaiming_uniform_')
    parser.add_argument('--n_class', default=3)

    parser.add_argument('--logs_dir', default='./logs')
    parser.add_argument('--exp_name', default='multi_classification_exp5')

    parser.add_argument('--max_num_epochs',
                        default=200)
    parser.add_argument('--batch_size',
                        default=8)
    parser.add_argument('--random_seed',
                        default=12345678)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--optimizer',
                        default='Adam')
    parser.add_argument('--lr',
                        default=5e-4)
    parser.add_argument('--weight_decay',
                        default=0.0001)
    parser.add_argument('--optimizer_momentum',
                        default=0.9)
    parser.add_argument('--normalization_mean',
                        default=[0.5, 0.5, 0.5])
    parser.add_argument('--normalization_std',
                        default=[0.5, 0.5, 0.5])

    parser.add_argument('--train_mode',
                        default=False)
    parser.add_argument('--load_pretrained_model',
                        default=True)
    parser.add_argument('--overfitting_test',
                        default=False)
    parser.add_argument('--show_test_result',
                        default=True)
    parser.add_argument('--show_cam',
                        default=False)
    parser.add_argument('--model_save_name',
                        default='Checkpoint')
    parser.add_argument('--model_loss',
                        default='0.321078')
    parser.add_argument('--load_exp_name',
                        default='multi_classification_exp4')
    # exp17 is the best one for 64*64 patches
    # exp19 is the best one for 128*128 patches

    parser.add_argument('--generate_gb',
                        default=False)
    parser.add_argument('--num_patches',
                        default=100)
    parser.add_argument('--patch_size',
                        default=64)

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
             'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    # multi classification task
    parser.add_argument('--n_crack_types', default=5)
    parser.add_argument('--positive_threshold', default=0.5)

    args = parser.parse_args()
    return args