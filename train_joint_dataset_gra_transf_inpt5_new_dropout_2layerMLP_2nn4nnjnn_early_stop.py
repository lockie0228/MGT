import argparse
import collections
import datetime
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


MGT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MGT_DIR)

from dataloader.JointDataset4dict_2nn4nnjnn import JointDataset_2nn4nnjnn
from network.gra_transf_inpt5_new_dropout_2layerMLP_3_adj_mtx import make_model
from utils.AverageMeter import AverageMeter
from utils.EarlyStopping import EarlyStopping
from utils.Logger import Logger
from utils.accuracy import accuracy


parser = argparse.ArgumentParser(description="MGT_stage_1_joint_dataset")
parser.add_argument(
    "--exp",
    type=str,
    default="train_joint_dataset_mgt_2nn4nnjnn_early_stop_001",
    help="experiment",
)
parser.add_argument("--batch_size", type=int, default=192, help="batch_size")
parser.add_argument("--num_workers", type=int, default=12, help="num_workers")
parser.add_argument("--gpu", type=str, default="1", help="choose GPU")
parser.add_argument(
    "--joint_data_root",
    type=str,
    default="./dataloader/joint_mgt_cache",
    help="exported joint MGT cache directory",
)
parser.add_argument("--num_epochs", type=int, default=100, help="num_epochs")
parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=10,
    help="early stopping patience",
)
parser.add_argument("--learning_rate", type=float, default=0.00005, help="learning rate")
parser.add_argument("--display_step", type=int, default=100, help="display step")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


basic_configs = collections.OrderedDict()
basic_configs["serial_number"] = args.exp
basic_configs["random_seed"] = int(time.time())
_seed = basic_configs["random_seed"]
random.seed(_seed)
np.random.seed(_seed)
torch.manual_seed(_seed)
torch.cuda.manual_seed(_seed)
torch.cuda.manual_seed_all(_seed)
os.environ["PYTHONHASHSEED"] = str(_seed)
basic_configs["learning_rate"] = args.learning_rate
basic_configs["num_epochs"] = args.num_epochs
basic_configs["early_stopping_patience"] = args.early_stopping_patience
basic_configs["lr_protocol"] = [
    (10, args.learning_rate),
    (20, args.learning_rate * 0.7),
    (30, args.learning_rate * 0.7 * 0.7),
    (40, args.learning_rate * 0.7 * 0.7 * 0.7),
    (50, args.learning_rate * 0.7 * 0.7 * 0.7 * 0.7),
    (60, args.learning_rate * 0.7 * 0.7 * 0.7 * 0.7 * 0.7),
    (70, args.learning_rate * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7),
    (80, args.learning_rate * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7 * 0.7),
    (85, args.learning_rate * (0.7 ** 8)),
    (90, args.learning_rate * (0.7 ** 9)),
    (95, args.learning_rate * (0.7 ** 10)),
    (100, args.learning_rate * (0.7 ** 11)),
]
basic_configs["display_step"] = args.display_step
lr_protocol = basic_configs["lr_protocol"]


dataloader_configs = collections.OrderedDict()
dataloader_configs["joint_data_root"] = args.joint_data_root
dataloader_configs["batch_size"] = args.batch_size
dataloader_configs["num_workers"] = args.num_workers


config_path = os.path.join(args.joint_data_root, "joint_mgt_config.json")
with open(config_path, "r") as config_file:
    dataset_config = json.load(config_file)


def _load_split_data(split):
    data_dict_path = os.path.join(args.joint_data_root, split + "_dataset_dict.pickle")
    sketch_list_path = os.path.join(args.joint_data_root, split + "_set.txt")
    with open(data_dict_path, "rb") as data_dict_file:
        data_dict = pickle.load(data_dict_file)
    return JointDataset_2nn4nnjnn(
        sketch_list=sketch_list_path,
        data_dict=data_dict,
        max_seq_len=dataset_config["max_seq_len"],
        pen_down_id=dataset_config["pen_down_id"],
        pen_up_id=dataset_config["pen_up_id"],
    )


train_dataset = _load_split_data("train")
val_dataset = _load_split_data("valid")
test_dataset = _load_split_data("test")

train_loader = DataLoader(
    train_dataset,
    batch_size=dataloader_configs["batch_size"],
    shuffle=True,
    num_workers=dataloader_configs["num_workers"],
)
val_loader = DataLoader(
    val_dataset,
    batch_size=dataloader_configs["batch_size"],
    shuffle=False,
    num_workers=dataloader_configs["num_workers"],
)
test_loader = DataLoader(
    test_dataset,
    batch_size=dataloader_configs["batch_size"],
    shuffle=False,
    num_workers=dataloader_configs["num_workers"],
)


exp_dir = os.path.join("./experimental_results", args.exp)

exp_log_dir = os.path.join(exp_dir, "log")
if not os.path.exists(exp_log_dir):
    os.makedirs(exp_log_dir)

exp_visual_dir = os.path.join(exp_dir, "visual")
if not os.path.exists(exp_visual_dir):
    os.makedirs(exp_visual_dir)

exp_ckpt_dir = os.path.join(exp_dir, "checkpoints")
if not os.path.exists(exp_ckpt_dir):
    os.makedirs(exp_ckpt_dir)

now_str = datetime.datetime.now().__str__().replace(" ", "_")
writer_path = os.path.join(exp_visual_dir, now_str)
writer = SummaryWriter(writer_path)

logger_path = os.path.join(exp_log_dir, now_str + ".log")
logger = Logger(logger_path).get_logger()


metadata_summary = dataset_config.get(
    "split_sizes",
    {
        "train": len(train_dataset),
        "valid": len(val_dataset),
        "test": len(test_dataset),
    },
)
class_list_path = os.path.join(exp_dir, "joint_class_list.json")
with open(class_list_path, "w") as class_list_file:
    json.dump(dataset_config["class_list"], class_list_file, indent=2)

run_config_path = os.path.join(exp_dir, "joint_mgt_config.json")
with open(run_config_path, "w") as run_config_file:
    json.dump(
        {
            "args": vars(args),
            "basic_configs": basic_configs,
            "dataloader_configs": dataloader_configs,
            "metadata_summary": metadata_summary,
            "source_dataset_config": dataset_config,
            "num_classes": dataset_config["num_classes"],
            "max_seq_len": dataset_config["max_seq_len"],
            "feat_dict_size": dataset_config["feat_dict_size"],
            "pen_down_id": dataset_config["pen_down_id"],
            "pen_up_id": dataset_config["pen_up_id"],
            "pad_id": dataset_config["pad_id"],
        },
        run_config_file,
        indent=2,
    )


logger.info("argument parser settings: {}".format(args))
logger.info("basic configuration settings: {}".format(basic_configs))
logger.info("dataloader configuration settings: {}".format(dataloader_configs))
logger.info("joint split sizes: {}".format(metadata_summary))
logger.info("joint class list path: {}".format(class_list_path))
logger.info("joint mgt config path: {}".format(run_config_path))


loss_function = nn.CrossEntropyLoss()
max_val_acc = 0.0
max_val_acc_epoch = -1


network_configs = collections.OrderedDict()
network_configs["output_dim"] = int(dataset_config["num_classes"])
network_configs["feat_dict_size"] = int(dataset_config["feat_dict_size"])
network_configs["max_seq_len"] = int(dataset_config["max_seq_len"])
network_configs["n_heads"] = 8
network_configs["embed_dim"] = 256
network_configs["n_layers"] = 4
network_configs["feed_forward_hidden"] = 4 * network_configs["embed_dim"]
network_configs["normalization"] = "batch"
network_configs["dropout"] = 0.25
network_configs["mlp_classifier_dropout"] = 0.25

logger.info("network configuration settings: {}".format(network_configs))


net = make_model(
    n_classes=network_configs["output_dim"],
    coord_input_dim=2,
    feat_input_dim=2,
    feat_dict_size=network_configs["feat_dict_size"],
    n_layers=network_configs["n_layers"],
    n_heads=network_configs["n_heads"],
    embed_dim=network_configs["embed_dim"],
    feedforward_dim=network_configs["feed_forward_hidden"],
    normalization=network_configs["normalization"],
    dropout=network_configs["dropout"],
    mlp_classifier_dropout=network_configs["mlp_classifier_dropout"],
)
net = net.cuda()


optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)


def _move_batch_to_cuda(batch):
    (
        coordinate,
        label,
        flag_bits,
        stroke_len,
        attention_mask,
        attention_mask_2,
        attention_mask_3,
        padding_mask,
        position_encoding,
    ) = batch

    coordinate = coordinate.cuda()
    label = label.cuda()
    flag_bits = flag_bits.cuda()
    stroke_len = stroke_len.cuda()
    attention_mask = attention_mask.cuda()
    attention_mask_2 = attention_mask_2.cuda()
    attention_mask_3 = attention_mask_3.cuda()
    padding_mask = padding_mask.cuda()
    position_encoding = position_encoding.cuda()

    flag_bits.squeeze_(2)
    position_encoding.squeeze_(2)
    stroke_len.unsqueeze_(1)

    return (
        coordinate,
        label,
        flag_bits,
        stroke_len,
        attention_mask,
        attention_mask_2,
        attention_mask_3,
        padding_mask,
        position_encoding,
    )


def train_function(epoch):
    training_loss = AverageMeter()
    training_acc = AverageMeter()
    net.train()

    lr = next((lr for (max_epoch, lr) in lr_protocol if max_epoch > epoch), lr_protocol[-1][1])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logger.info("set learning rate to: {}".format(lr))

    for idx, batch in enumerate(tqdm(train_loader, ascii=True)):
        (
            coordinate,
            label,
            flag_bits,
            stroke_len,
            attention_mask,
            attention_mask_2,
            attention_mask_3,
            padding_mask,
            position_encoding,
        ) = _move_batch_to_cuda(batch)

        optimizer.zero_grad()

        output = net(
            coordinate,
            flag_bits,
            position_encoding,
            attention_mask,
            attention_mask_2,
            attention_mask_3,
            padding_mask,
            stroke_len,
        )

        batch_loss = loss_function(output, label)
        batch_loss.backward()
        optimizer.step()

        training_loss.update(batch_loss.item(), coordinate.size(0))
        training_acc.update(accuracy(output, label, topk=(1,))[0].item(), coordinate.size(0))

        if (idx + 1) % basic_configs["display_step"] == 0:
            logger.info(
                "==> Iteration [{}][{}/{}]:".format(epoch + 1, idx + 1, len(train_loader))
            )
            logger.info("current batch loss: {}".format(batch_loss.item()))
            logger.info("average loss: {}".format(training_loss.avg))
            logger.info("average acc: {}".format(training_acc.avg))

    logger.info("Begin evaluating on validation set")
    validation_loss, validation_acc = validate_function(val_loader)
    logger.info("Begin evaluating on testing set")
    test_loss, test_acc = validate_function(test_loader)

    writer.add_scalars(
        "loss",
        {
            "training_loss": training_loss.avg,
            "validation_loss": validation_loss.avg,
            "test_loss": test_loss.avg,
        },
        epoch + 1,
    )
    writer.add_scalars(
        "acc",
        {
            "training_acc": training_acc.avg,
            "validation_acc": validation_acc.avg,
            "test_acc": test_acc.avg,
        },
        epoch + 1,
    )

    return validation_acc


def validate_function(data_loader):
    validation_loss = AverageMeter()
    validation_acc_1 = AverageMeter()
    validation_acc_5 = AverageMeter()
    validation_acc_10 = AverageMeter()

    net.eval()
    timelist = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader, ascii=True)):
            (
                coordinate,
                label,
                flag_bits,
                stroke_len,
                attention_mask,
                attention_mask_2,
                attention_mask_3,
                padding_mask,
                position_encoding,
            ) = _move_batch_to_cuda(batch)

            tic = time.time()
            output = net(
                coordinate,
                flag_bits,
                position_encoding,
                attention_mask,
                attention_mask_2,
                attention_mask_3,
                padding_mask,
                stroke_len,
            )
            timelist.append(time.time() - tic)

            batch_loss = loss_function(output, label)
            validation_loss.update(batch_loss.item(), coordinate.size(0))

            acc_1, acc_5, acc_10 = accuracy(output, label, topk=(1, 5, 10))
            validation_acc_1.update(acc_1, coordinate.size(0))
            validation_acc_5.update(acc_5, coordinate.size(0))
            validation_acc_10.update(acc_10, coordinate.size(0))

        logger.info("==> Evaluation Result: ")
        logger.info(
            "loss: {}  acc@1: {} acc@5: {} acc@10: {}".format(
                validation_loss.avg,
                validation_acc_1.avg,
                validation_acc_5.avg,
                validation_acc_10.avg,
            )
        )
        logger.info("Total inference time: {}s".format(sum(timelist)))

    return validation_loss, validation_acc_1


if __name__ == "__main__":
    logger.info("Begin evaluating on validation set before training")
    validate_function(val_loader)

    logger.info("training status: ")
    early_stopping = EarlyStopping(
        patience=basic_configs["early_stopping_patience"], delta=0
    )

    for epoch in range(basic_configs["num_epochs"]):
        logger.info("Begin training epoch {}".format(epoch + 1))
        validation_acc = train_function(epoch)

        if validation_acc.avg > max_val_acc:
            max_val_acc = validation_acc.avg
            max_val_acc_epoch = epoch + 1

        early_stopping(validation_acc.avg)
        logger.info("Early stopping counter: {}".format(early_stopping.counter))
        logger.info("Early stopping best_score: {}".format(early_stopping.best_score))
        logger.info("Early stopping early_stop: {}".format(early_stopping.early_stop))

        if early_stopping.early_stop is True:
            logger.info("Early stopping after Epoch: {}".format(epoch + 1))
            break

        net_checkpoint_name = args.exp + "_net_epoch" + str(epoch + 1)
        net_checkpoint_path = os.path.join(exp_ckpt_dir, net_checkpoint_name)
        net_state = {"epoch": epoch + 1, "network": net.state_dict()}
        torch.save(net_state, net_checkpoint_path)

    logger.info(
        "max_val_acc: {}  max_val_acc_epoch: {}".format(max_val_acc, max_val_acc_epoch)
    )
