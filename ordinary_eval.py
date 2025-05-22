from model import build_model
from datasets import Dataset as HFDataset
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
from scipy.stats import spearmanr         # add spearman
from lifelines.utils import concordance_index  # add c-index

def ordinary_eval(data_path = "./data/ordinary_dataset",
                  model_path = "./ckpt/best_model.pth",
                  batch_size = 64,
                  num_workers = 4,
                  ):
    # Load the dataset
    test_dataset = HFDataset.load_from_disk(os.path.join(data_path, "test"))
    test_dataset.set_format(type="torch", columns=["grid", "label"])
    # Load the model
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print("Starting evaluation...")
    print(f"Using device: {device}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            grids = batch["grid"].to(device).float()
            # NOTE: NORMALIZE
            labels = batch["label"].to(device) / 15.
            labels = labels.unsqueeze(1)
            outputs = model(grids)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels) 
    # Calculate MSE
    mse = np.mean((all_predictions - all_labels) ** 2)
    print(f"Mean Squared Error: {mse}")
    # Calculate MAE
    mae = np.mean(np.abs(all_predictions - all_labels))
    print(f"Mean Absolute Error: {mae}")
    # Calculate R^2
    ss_total = np.sum((all_labels - np.mean(all_labels)) ** 2)
    ss_residual = np.sum((all_labels - all_predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    print(f"R^2 Score: {r2}")
    # Calculate Pearson correlation coefficient
    pearson_corr = np.corrcoef(all_predictions.flatten(), all_labels.flatten())[0, 1]
    print(f"Pearson Correlation Coefficient: {pearson_corr}")
    spearman_corr, _ = spearmanr(all_predictions.flatten(), all_labels.flatten())
    print(f"Spearman Correlation Coefficient: {spearman_corr}")
    # Calculate C-index
    c_index = concordance_index(all_labels.flatten(), all_predictions.flatten())
    print(f"C-index: {c_index}")
    print("Evaluation completed.")
    return pearson_corr
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--data_path", type=str, default="./data/ordinary_dataset", help="Path to the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model", required=True)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    args = parser.parse_args()
    pearson_corr = ordinary_eval(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )