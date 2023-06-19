import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from avalanche_extend.benchmarks.datasets.cloc import (
    CLOCDataset
)

# This code is used only for the ablation study section of the paper. Note that it is not developed using the Avalanche framework.

def compute_backward_transfer(args, model, scenario, device, tensorboardLogger):
    """
    Computes the backward transfer metric for a neural network model on a given test dataset.
    
    Args:
    - args (namespace): A namespace object with the command-line arguments.
    - model (nn.Module): The neural network model to evaluate.
    - scenario (ContinualScenario): The scenario object that contains the test data.
    - device (str): The device to use for evaluating the model (e.g., "cuda" or "cpu").
    - tensorboardLogger (TensorBoardLogger): The logger object to use for recording the evaluation results.
    
    Returns:
    - None
    """
    print("Starting Backward Transfer evaluation...")

    # Set the name of the metric
    metric_name = "Backward_Transfer"

    # Set the model to evaluation mode
    model.eval()


    # Get the time_taken attribute from the CLOC test set
    cloc_dataset_test = CLOCDataset(
        split="test", dataset_root=args.dataset_root
    )
    time_taken = cloc_dataset_test.time_taken
    starting_test_time = time_taken[0]


    # Get the test set and create a DataLoader for the test dataset
    test_stream = scenario.test_stream[0]
    test_dataset = test_stream.dataset.eval()
    dataloader = DataLoader(
            test_dataset,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True)

    # Initialize lists to store predictions, seen samples, and x-axis values for plotting
    prediction_list = [] 
    seen_samples = []
    x_plot_values = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Iterate over the test data
        for mbatch in tqdm(dataloader):
            # Move the batch to the specified device
            for i in range(len(mbatch)):
                mbatch[i] = mbatch[i].to(device)
            
            mb_x, mb_target, mb_index, mb_label = mbatch

            # Forward pass
            mb_output = model(mb_x)

            # Compute true and predicted labels
            true_y = torch.as_tensor(mb_target)
            predicted_y = torch.as_tensor(mb_output)
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

            # Compute the number of correct predictions in the batch
            true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
            total_patterns = len(true_y)
            prediction_list.append(true_positives)
            
            # Compute the number of seen samples so far
            offest = 0
            if len(seen_samples) > 0:
                offest = seen_samples[-1]
            seen_samples.append(total_patterns + offest) 

            # Compute the x-axis values for plotting
            x_plot_values.append(max(0, time_taken[mb_index[-1].item()] - starting_test_time))
    
    # Reverse the prediction list array
    prediction_list = prediction_list[::-1]

    # Compute the metric for each batch and log the results to TensorBoard
    acc_sum = 0
    for i in range(len(prediction_list)):
        acc_sum += prediction_list[i] 
        tensorboardLogger.log_single_metric(metric_name, 100*acc_sum/seen_samples[i], x_plot_values[i])


def compute_forward_transfer(args, model, scenario, device, tensorboardLogger, checkpoint_path, time_percent):
    """
    Computes the forward transfer metric for a neural network model on a given test dataset.
    
    Args:
    - args (namespace): A namespace object with the command-line arguments.
    - model (nn.Module): The neural network model to evaluate.
    - scenario (ContinualScenario): The scenario object that contains the test data.
    - device (str): The device to use for evaluating the model (e.g., "cuda" or "cpu").
    - tensorboardLogger (TensorBoardLogger): The logger object to use for recording the evaluation results.
    - checkpoint_path (str): The model checkpoint to compute the forward transfer for.
    - time_percent (int): The percentage of the stream at which the forward transfer is computed. 
    
    Returns:
    - None
    """
    print("Starting Forward Transfer evaluation...")

    # Set the name of the metric
    metric_name = f"Forward_Transfer_{time_percent}"

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    next_training_index = checkpoint['next_index'].item()


    # Get the timestamp of the last datapoint the model was trained on
    cloc_dataset_train = CLOCDataset(
        split="train", dataset_root=args.dataset_root
    )
    train_time_taken = cloc_dataset_train.time_taken
    last_train_time = train_time_taken[next_training_index]


    # Get the time_taken attribute from the CLOC test set
    cloc_dataset_test = CLOCDataset(
        split="test", dataset_root=args.dataset_root
    )
    test_time_taken = cloc_dataset_test.time_taken

    # Find the index at which to split the test dataset
    test_set_index = -1
    for j, test_t in enumerate(test_time_taken):
        if test_t > last_train_time:
            test_set_index = j
            break
    starting_test_time = test_time_taken[test_set_index]


    # Get the test set and create a DataLoader for the test dataset
    test_stream = scenario.test_stream[0]
    test_dataset = test_stream.dataset.eval()
    future_test_dataset = torch.utils.data.Subset(test_dataset, range(test_set_index, len(test_dataset)))
    dataloader = DataLoader(
            future_test_dataset,
            num_workers=args.workers,
            batch_size=args.batch_size,
            pin_memory=True)

    # Initialize lists to store predictions, seen samples, and x-axis values for plotting
    prediction_list = [] 
    seen_samples = []
    x_plot_values = []

    # Disable gradient computation during evaluation
    with torch.no_grad():
        # Iterate over the test data
        for mbatch in tqdm(dataloader):
            # Move the batch to the specified device
            for i in range(len(mbatch)):
                mbatch[i] = mbatch[i].to(device)
            
            mb_x, mb_target, mb_index, mb_label = mbatch

            # Forward pass
            mb_output = model(mb_x)

            # Compute true and predicted labels
            true_y = torch.as_tensor(mb_target)
            predicted_y = torch.as_tensor(mb_output)
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

            # Compute the number of correct predictions in the batch
            true_positives = float(torch.sum(torch.eq(predicted_y, true_y)))
            total_patterns = len(true_y)
            prediction_list.append(true_positives)
            
            # Compute the number of seen samples so far
            offest = 0
            if len(seen_samples) > 0:
                offest = seen_samples[-1]
            seen_samples.append(total_patterns + offest) 

            # Compute the x-axis values for plotting
            x_plot_values.append(max(0, test_time_taken[mb_index[-1].item()] - starting_test_time))

    # Compute the metric for each batch and log the results to TensorBoard
    acc_sum = 0
    for i in range(len(prediction_list)):
        acc_sum += prediction_list[i]
        tensorboardLogger.log_single_metric(metric_name, 100*acc_sum/seen_samples[i], x_plot_values[i])
