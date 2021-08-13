import argparse
import os
import torch
from config import config
from core.augmentations import detection as detection_augmentations
from core.criterions import MultiBoxLoss, YoloV3Loss
from core.datasets import readers
from core.models import models
from core.logging import configure_logger
from core.logging import TrainingMonitor
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
import torch.distributed
from core.metrics import mean_average_precision


def main():
    logger = configure_logger()

    # decide whether you want to train on Multiple GPUs or not
    use_distributed_data_parallel = torch.cuda.is_available() and False
    use_data_parallel = torch.cuda.is_available() and True
    assert not (use_distributed_data_parallel and use_data_parallel)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", help="The pid to resume training.")
    parser.add_argument("--start", default=-1, type=int, help="The epoch to resume training.")
    if use_distributed_data_parallel:
        parser.add_argument("--local_rank", type=int, default=0)
        parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    local_rank = 0
    if use_distributed_data_parallel:
        local_rank = args.local_rank
        local_world_size = args.local_world_size
        print(f"Local Rank = {local_rank}, Local World Size = {local_world_size}")

    # Create directories
    output_dir = None
    checkpoints_dir = None

    if local_rank == 0:
        output_dir = f"{os.getpid()}" if args.pid is None else args.pid
        output_dir = os.path.join("models", output_dir)
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        logger.info(f"Saving outputs to {output_dir}")

    # Use GPU
    if use_distributed_data_parallel:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:6" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Create model
    logger.info(f"Configuring model ...")
    model = models[config.model.type][config.model.name](n_classes=config.dataset.n_classes)
    # model = TestModel()
    model = model.to(device)

    # Read datasets
    Reader = readers[config.dataset.reader]

    # Image augmentations
    train_transform = transforms.Compose([
        detection_augmentations.RandomBlur(p=0.2),
        detection_augmentations.RandomCutout(p=0.2),
        detection_augmentations.RandomFlip(p=0.2),
        detection_augmentations.Resize((model.config["image_size"], model.config["image_size"])),
        detection_augmentations.ToTensor(),
    ])
    val_transform = transforms.Compose([
        detection_augmentations.Resize((model.config["image_size"], model.config["image_size"])),
        detection_augmentations.ToTensor(),
    ])

    # Create dataset and dataloader
    logger.info(f"Reading training data from {config.dataset.train_path} ({config.dataset.reader}) ...")
    train_dataset = Reader(config.dataset.train_path, transform=train_transform)
    val_dataset = Reader(config.dataset.valid_path, transform=val_transform)
    logger.info(f"Classes: {', '.join(train_dataset.classes)}", extra={"type": "DATASET"})
    logger.info(f"Training samples: {len(train_dataset)}", extra={"type": "DATASET"})

    if use_distributed_data_parallel:
        torch.distributed.init_process_group('nccl', rank=local_rank, world_size=local_world_size)
        train_dist_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_dist_sampler)
        val_dataloader = DataLoader(val_dataset)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                      collate_fn=Reader.collate_fn, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size,
                                    collate_fn=Reader.collate_fn, shuffle=False)

    # Create optimizer
    logger.info(f"Configuring optimizer ...")
    if config.learning_algorithm == 'SGD':
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay,
                        nesterov=True)
    elif config.learning_algorithm == 'Adam':
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:  # default
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # Loss function
    if config.model.criterion == "multibox_loss":
        criterion = MultiBoxLoss(priors=model.priors, variances=model.config["variances"])
    elif config.model.criterion == "yolov3_loss":
        criterion = YoloV3Loss(model.config, model.anchors, device)
    elif config.model.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None

    training_monitor = TrainingMonitor(
        os.path.join(output_dir, "monitor.jpg"),
        os.path.join(output_dir, "monitor.json"),
        start=args.start
    )
    training_monitor.init()

    if args.start >= 0:
        logger.info(f"Resuming training from epoch {args.start} ...")
        checkpoint = torch.load(os.path.join(checkpoints_dir, f"ckpt-{args.start}.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if use_distributed_data_parallel:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        model.to(device)
    elif use_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=[6, 7])  # Encapsulate the model
        model.to(device)

    for epoch in range(args.start+1, config.epochs):
        # Training loop
        total_loss = 0
        image_count = 0
        model.train()
        for i, (images, targets) in enumerate(train_dataloader):
            image_count += len(images)
            images = images.to(device)
            predictions = model(images)
            if isinstance(targets, list):
                targets = [target.to(device) for target in targets]
            else:
                targets = targets.to(device)

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 300 == 0:
                logger.info(f"Epoch[{epoch:03d}:{local_rank}] Batch[{i:03d}]\t\tLoss: {loss.item():.5f}", extra={"type": "TRAINING"})

        total_loss /= image_count

        if local_rank == 0:
            model.eval()
            image_count = 0

            total_val_loss = 0
            for i, (images, targets) in enumerate(val_dataloader):
                with torch.no_grad():
                    image_count += len(images)
                    images = images.to(device)
                    predictions = model(images)

                    if isinstance(targets, list):
                        targets = [target.to(device) for target in targets]
                    else:
                        targets = targets.to(device)

                    val_loss = criterion(predictions, targets)
                    total_val_loss += val_loss.item()
                    # predictions = model.postprocess(images, predictions)
                    # map_val = mean_average_precision(targets, predictions, train_dataset.classes)
                    map_val = -1


            total_val_loss /= image_count
            print(f"Epoch: {epoch}, Tr_Loss: {total_loss:.5f} Val_Loss: {total_val_loss:.5f}, MAP: {map_val:.2f}")
            f = open("Training_Results.txt", "a")
            f.write(f"Epoch: {epoch}, Tr_Loss: {total_loss:.5f}, Val_Loss: {total_val_loss:.5f}, MAP: {map_val:.2f}\n")
            f.close()

            # Update training monitor
            training_monitor.update({"loss": total_loss})
            # logger.info(f"Epoch[{epoch:03d}]\t\t\tLoss: {total_loss:.5f}", extra={ "type": "TRAINING" })

            # Save checkpoint
            if use_distributed_data_parallel or use_data_parallel:
                torch.save({
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(checkpoints_dir, f"ckpt-{epoch}.tar"))
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, os.path.join(checkpoints_dir, f"ckpt-{epoch}.tar"))
        if use_distributed_data_parallel:
            torch.distributed.barrier()


if __name__ == '__main__':
    main()
