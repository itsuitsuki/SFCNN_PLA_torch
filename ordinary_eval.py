from model import build_model
from datasets import Dataset as HFDataset
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse

def ordinary_eval(data_path = "./data/ordinary_dataset",
                  model_path = "./ckpt/best_model.pth",
                  batch_size = 64,
                  ):
    # Load the dataset
    test_dataset = HFDataset.load_from_disk(os.path.join(data_path, "test"))
    # Load the model
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    def collate_fn(batch):
        grids = torch.stack([torch.tensor(item["grid"], dtype=torch.float32) for item in batch]) # (64, 20, 20, 20, 28)
        # -> (64, 28, 20, 20, 20)
        grids = grids.permute(0, 4, 1, 2, 3)
        return {
            "grid": grids,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1) # (64, 1)
        }
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            grids = batch["grid"].to(device)
            labels = batch["label"].to(device)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--data_path", type=str, default="./data/ordinary_dataset", help="Path to the dataset")
    parser.add_argument("--model_path", type=str, default="./ckpt/best_model.pth", help="Path to the model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    args = parser.parse_args()
    ordinary_eval(
        data_path=args.data_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )