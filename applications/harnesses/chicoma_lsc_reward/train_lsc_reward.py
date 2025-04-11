"""Training script for reward/value network using hybrid2vectorCNN.

This script loads the LSC_hfield_reward_DataSet and trains a reward prediction network 
using the hybrid2vectorCNN architecture.

Includes learning rate scheduling and support for normalized inputs.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from yoke.models.hybridCNNmodules import hybrid2vectorCNN
from yoke.datasets.lsc_dataset import LSC_hfield_reward_DataSet
import yoke.torch_training_utils as tr  # Assuming this exists
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli  # Assuming this exists

#############################################
# Inputs
#############################################
descr_str = (
    "Trains reward network architecture to calculate error between current and target density fields."
)
parser = argparse.ArgumentParser(
    prog="reward network training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Change some default filepaths.
parser.set_defaults(design_file="design_lsc240420_MASTER.csv")

#############################################
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Data Paths
    design_file = os.path.abspath(args.LSC_DESIGN_DIR + args.design_file)
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    anchor_lr = args.anchor_lr
    num_cycles = args.num_cycles
    min_fraction = args.min_fraction
    terminal_steps = args.terminal_steps
    warmup_steps = args.warmup_steps

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    num_workers = args.num_workers

    batch_size = args.batch_size
    # Leave one CPU out of the worker queue. Not sure if this is necessary.
    train_per_val = args.TRAIN_PER_VAL

    # Epoch Parameters
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])


    #############################################
    # Model
    #############################################
    model = hybrid2vectorCNN(img_size = (1, 1120, 400),
        input_vector_size = 28,
        output_dim = 1,
        features = 12,
        depth = 12,
        kernel = 3,
        img_embed_dim = 32,
        vector_embed_dim = 32,
        size_reduce_threshold = (8, 8),
        vector_feature_list = [32, 32, 64, 64],
        output_feature_list = [64, 128, 128, 64],
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm
        )
    #model.to(args.device)

    #############################################
    # Training Setup
    #############################################
    loss_fn = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01
    )
    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    if args.multigpu:
        model = nn.DataParallel(model)

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


    #############################################
    # LR scheduler
    #############################################
    # We will take a scheduler step every back-prop step so the number of steps
    # is the number of previous batches.
    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)


    LRsched = CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=anchor_lr,
        terminal_steps=terminal_steps,
        warmup_steps=warmup_steps,
        num_cycles=num_cycles,
        min_fraction=min_fraction,
        last_epoch=last_epoch,
    )


    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_hfield_reward_DataSet(
        args.LSC_NPZ_DIR, train_filelist, design_file, #normalization_file
    )
    val_dataset = LSC_hfield_reward_DataSet(
        args.LSC_NPZ_DIR, validation_filelist, design_file,
    )
    test_dataset = LSC_hfield_reward_DataSet(
        args.LSC_NPZ_DIR, test_filelist, design_file,
    )

    print("Datasets initialized.")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset, batch_size, train_batches, num_workers=num_workers
    )
    val_dataloader = tr.make_dataloader(
        val_dataset, batch_size, val_batches, num_workers=num_workers
    )

    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        # Train an Epoch
        tr.train_array_csv_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=compiled_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
        )

        # Increment LR scheduler
        #stepLRsched.step()

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...")
        print("Epoch time:", epoch_time)

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    compiled_model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        compiled_model, optimizer, epochIDX, new_h5_path, compiled=True
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
