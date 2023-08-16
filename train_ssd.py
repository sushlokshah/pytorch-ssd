import argparse
import os
import logging
import sys
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.widerface import WIDERFace
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from ray.tune.search.optuna import OptunaSearch
# import raytune for hyperparameter tuning
import ray
from ray.air import session
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import cv2 as cv
from vision.utils import box_utils

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="wider_face", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', type=list,
                    default=["/sushlok/widerface"], help='Dataset directory path')
parser.add_argument('--validation_dataset', type=str,
                    default='/sushlok/widerface', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="mb1-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=1e-3, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=1e-3, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',  default='/sushlok/pytorch-ssd/models/mobilenet_v1_with_relu_69_5.pth',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd',
                    default='/sushlok/pytorch-ssd/models/mobilenet-v1-ssd-mp-0_675.pth', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="cosine", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=400, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')


parser.add_argument('--validation_epochs', default=1, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='./checkpoint/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--tensorboard_dir', default='/sushlok/pytorch-ssd/run/',
                    help='Directory for saving tensorboard info')


# logging.basicConfig(stream=sys.stdout, level=print,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    print('=================== training =====================')
    for i, data in enumerate(loader):
        # print(i)
        IMAGE_PATH, images, boxes, labels = data
        # print(boxes[0].shape)
        # print(boxes[0])
        # print(images[0].max(), images[0].min())
        # print(boxes[0].shape, labels[0].shape)
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # print(images.shape, boxes.shape, labels.shape)
        optimizer.zero_grad()
        confidence, locations = net(images)
        # print(locations[0].shape)
        # print(locations[0])

        # print(confidence[0][0], locations[0][0])
        # print(boxes.min(), boxes.max())
        regression_loss, classification_loss = criterion(
            confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        # print('loss', loss.item())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / (i + 1)
            avg_reg_loss = running_regression_loss / (i + 1)
            avg_clf_loss = running_classification_loss / (i + 1)
            print(
                f"Epoch: {epoch}, Step: {i}, " +
                f"training Loss: {avg_loss:.4f}, " +
                f"training Regression Loss {avg_reg_loss:.4f}, " +
                f"training Classification Loss: {avg_clf_loss:.4f}"
            )
            # running_loss = 0.0
            # running_regression_loss = 0.0
            # running_classification_loss = 0.0
    print('====================== losses ===================')
    print('training_loss', running_loss / len(loader), 'training_regression_loss',
          running_regression_loss / len(loader), 'training_classification_loss', running_classification_loss / len(loader))
    return running_loss / len(loader), running_regression_loss / len(loader), running_classification_loss / len(loader)


def test(loader, net, criterion, device, model_info, epoch):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    priors = model_info.priors.to(device)
    plot_image_id = np.random.randint(0, len(loader))
    for id, data in enumerate(loader):
        images_PATH, images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            # print(confidence[0][0])
            confidence = F.softmax(confidence, dim=2)
            # print(prob[0][0])
            boxes_pred = box_utils.convert_locations_to_boxes(
                locations, priors, model_info.center_variance, model_info.size_variance
            )
            boxes_pred = box_utils.center_form_to_corner_form(boxes_pred)
            regression_loss, classification_loss = criterion(
                confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

            # print(boxes_pred[0][0], boxes[0][0])
            if id == plot_image_id:
                print(
                    '====================== saving image {} ==================='.format(id))
                output_img = plot_image(images, boxes_pred, confidence)
                # create vis dir
                vis_dir = os.path.join('./vis')
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                cv.imwrite(os.path.join(
                    vis_dir, 'image_{}_{}.jpg'.format(epoch, id)), output_img)

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def plot_image(images, boxes, confidence):
    # convert image from tensor to numpy array
    image = images[0].permute(1, 2, 0).cpu().numpy()
    # image = (image - np.min(image))/(np.max(image) - np.min(image))
    image = (((image + 1) / 2) * 255).astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image_std = 128.0
    iou_threshold = 0.6
    center_variance = 0.1
    size_variance = 0.2
    prob_threshold = 0.3
    # draw boxes on image
    boxes = boxes[0]  # .cpu().numpy()
    confidence = confidence[0]
    print(confidence.shape)  # .cpu().numpy()
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidence.size(1)):
        probs = confidence[:, class_index]
        mask = probs > 0.2
        probs = probs[mask]
        print(probs.shape)
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        print(subset_boxes.shape)
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = box_utils.nms(box_probs, 'soft',
                                  score_threshold=prob_threshold,
                                  iou_threshold=iou_threshold,
                                  sigma=0.5,
                                  top_k=-1)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if len(picked_box_probs) == 0:
        return image
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= image.shape[1]
    picked_box_probs[:, 1] *= image.shape[0]
    picked_box_probs[:, 2] *= image.shape[1]
    picked_box_probs[:, 3] *= image.shape[0]
    if len(picked_labels) == 0:
        return image
    labels = torch.tensor(picked_labels)
    confidence = picked_box_probs[:, 4]
    boxes = picked_box_probs[:, :4]
    # print(boxes)
    # # normalized boxes to image size
    # boxes[:, 0] *= image.shape[1]
    # boxes[:, 1] *= image.shape[0]
    # boxes[:, 2] *= image.shape[1]
    # boxes[:, 3] *= image.shape[0]
    boxes = boxes.cpu().numpy()
    confidence = confidence.cpu().numpy()
    boxes = boxes.astype(np.int32)
    print('maximum confidence', np.max(confidence))
    # confidence = confidence[0].cpu().numpy()
    # print(boxes.shape, confidence.shape)
    for box, conf in zip(boxes, confidence):
        print(box, conf)
        if conf >= prob_threshold:
            box = box.astype(np.int32)
            image = cv.rectangle(cv.UMat(image), (box[0], box[1]),
                                 (box[2], box[3]), (0, 255, 0), 2)
            # print conf
            cv.putText(cv.UMat(image), '{:.2f}'.format(conf),
                       (box[0], box[1] - 2),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 0, 255),
                       2,
                       cv.LINE_AA)

    return image


def training_sweep(config=None, args=args, net=None, criterion=None, model_info=None):
    print(config)
    base_net_lr = config['base_net_lr'] if config['base_net_lr'] is not None else config['lr']
    extra_layers_lr = config['extra_layers_lr'] if config['extra_layers_lr'] is not None else config['lr']
    if args.freeze_base_net:
        print("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(
            net.regression_headers.parameters(), net.classification_headers.parameters())
        print("Freeze all the layers except prediction heads.")
    else:
        print("train all layers")
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        print(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        print(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        print("========================== pretrained ==============================")
        print(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    print(
        f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)
    criterion.to(DEVICE)

    optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print(f"Learning rate: {config['lr']}, Base net learning rate: {base_net_lr}, "
          + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        print("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        print("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(
            optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    print(f"Start training from epoch {last_epoch + 1}.")

    train_loader = DataLoader(train_dataset, config['batch_size'],
                              num_workers=args.num_workers,
                              shuffle=True)

    val_loader = DataLoader(val_dataset, 1,
                            num_workers=args.num_workers,
                            shuffle=False)

    # for data in val_loader:
    #     image_path, images, boxes, labels = data
    #     print(image_path[0])
    #     print(images[0][0][0][0], images[0][0][1][0])
    #     break
    # exit()

    for epoch in range(last_epoch + 1, config['num_epochs']):
        scheduler.step()
        print("================== epoch: ", epoch, "==================")
        print("lr: ", scheduler.get_last_lr(),
              'debug_steps: ', args.debug_steps)
        training_loss, training_regression_loss, training_classification_loss = train(train_loader, net, criterion, optimizer,
                                                                                      device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        # # print(
        #     f"Epoch: {epoch}, " +
        #     f"Training Loss: {training_loss:.4f}, " +
        #     f"Training Regression Loss {training_regression_loss:.4f}, " +
        #     f"Training Classification Loss: {training_classification_loss:.4f}"
        # )

        # tune.report(training_loss=training_loss, training_regression_loss=training_regression_loss,
        #             training_classification_loss=training_classification_loss)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            print('================== validation ==================')
            val_loss, val_regression_loss, val_classification_loss = test(
                val_loader, net, criterion, DEVICE, model_info, epoch)
            print('val_loss: ', val_loss, 'val_regression_loss: ', val_regression_loss,
                  'val_classification_loss: ',  val_classification_loss)
            # print(
            #     f"Epoch: {epoch}, " +
            #     f"Validation Loss: {val_loss:.4f}, " +
            #     f"Validation Regression Loss {val_regression_loss:.4f}, " +
            #     f"Validation Classification Loss: {val_classification_loss:.4f}"
            # )

            tune.report(training_loss=training_loss, training_regression_loss=training_regression_loss,
                        training_classification_loss=training_classification_loss, val_loss=val_loss, val_regression_loss=val_regression_loss,
                        val_classification_loss=val_classification_loss)

            if not os.path.exists(args.checkpoint_folder):
                os.mkdir(args.checkpoint_folder)

            model_path = os.path.join(
                args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            print(f"Saved model {model_path}")
        print("================== epoch: ", epoch,
              " complete ==================")


if __name__ == '__main__':
    timer = Timer()

    print(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        def create_net(num): return create_mobilenetv2_ssd_lite(
            num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-large-ssd-lite':
        def create_net(num): return create_mobilenetv3_large_ssd_lite(num)
        config = mobilenetv1_ssd_config
    elif args.net == 'mb3-small-ssd-lite':
        def create_net(num): return create_mobilenetv3_small_ssd_lite(num)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(
        config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(
        config.image_size, config.image_mean, config.image_std)

    print("Prepare training datasets.")
    datasets = []
    # print(dataset)
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(
                args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                                        transform=train_transform, target_transform=target_transform,
                                        dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(
                args.checkpoint_folder, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            print(dataset)
            num_classes = len(dataset.class_names)

        elif args.dataset_type == 'wider_face':
            dataset = WIDERFace(dataset_path,
                                transform=train_transform, target_transform=target_transform,
                                dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(
                args.checkpoint_folder, "wider-face-model-labels.txt")
            # store_labels(label_file, dataset.class_names)
            print(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(
                f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    print(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    print("Train dataset size: {}".format(len(train_dataset)))

    print("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
    elif args.dataset_type == 'wider_face':
        val_dataset = WIDERFace(dataset_path,
                                transform=test_transform, target_transform=target_transform,
                                dataset_type="val")
    else:
        raise ValueError(
            f"Dataset type {args.dataset_type} is not supported.")

    print(val_dataset)
    print("validation dataset size: {}".format(len(val_dataset)))

    print("Build network.")

    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)

    space_best_run = {
        'base_net_lr': 0.02,
        'lr': 0.012,
        'extra_layers_lr': 0.035,
        'num_epochs': 128,
        'batch_size': 256,
    }

    space = {
        'base_net_lr': tune.loguniform(1e-3, 1e-1),
        'lr': tune.loguniform(1e-3, 1e-1),
        'extra_layers_lr': tune.loguniform(1e-3, 1e-1),
        'num_epochs': tune.randint(128, 129),
        'batch_size': tune.randint(256, 257),
    }

    # training_sweep(config=space, args=args, net=net,
    #                criterion=criterion, model_info=config)
    # # hyperopt_search = HyperOptSearch(space, metric="mse", mode="min")
    algo = OptunaSearch(space=space, metric="val_loss",
                        mode="min", points_to_evaluate=[space_best_run])

    # # # configure reporter
    reporter = CLIReporter(
        parameter_columns=["base_net_lr", "lr",
                           "extra_layers_lr", "num_epochs", "batch_size"],
        metric_columns=['training_loss', 'training_regression_loss',
                        'training_classification_loss', 'val_loss', 'val_regression_loss', 'val_classification_loss'], max_report_frequency=120)

    tuner = tune.Tuner(
        tune.with_resources(partial(training_sweep, args=args, net=net, criterion=criterion, model_info=config), {
                            "cpu": 20, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=10,
            metric="val_loss",
            mode="min",
            search_alg=algo,
        ),
        run_config=air.RunConfig(
            storage_path=args.tensorboard_dir, name="ssd_{}_{}".format(args.dataset_type, args.net), progress_reporter=reporter),
        # param_space=space,
    )
    # tuner = tune.Tuner.restore(
    #     '/sushlok/pytorch-ssd/run/ssd_wider_face_mb1-ssd', trainable=partial(training_sweep, args=args, net=net, criterion=criterion, model_info=config), resume_errored=True, param_space=space)
    results = tuner.fit()
