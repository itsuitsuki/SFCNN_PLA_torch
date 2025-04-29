from model import build_model
import os
from datasets import Dataset as HFDataset
import torch
from tqdm import tqdm
import argparse
import numpy as np
import wandb
from scipy.stats import spearmanr
from lifelines.utils import concordance_index
# torch.set_printoptions(edgeitems=114514)
def train(data_path = "./data/ordinary_dataset",
          lr = 5e-4,
          dropout = 0.1,
          last_dense_wd = 0.01,
          n_epochs = 200,
          batch_size = 64,
          pearson_coeff = 0.1,
          max_grad_norm = 1.5,
        ):
    train_dataset = HFDataset.load_from_disk(os.path.join(data_path, "train"))
    valid_dataset = HFDataset.load_from_disk(os.path.join(data_path, "valid"))
    
    model = build_model(dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Apply weight decay only to the last linear layer
    last_layer_params = list(model.fc2.parameters())
    last_layer_param_names = [n for n, p in model.named_parameters() if ("fc2" in n)]
    # print(f"Last layer parameters: {last_layer_param_names}")
    other_params = [p for n, p in model.named_parameters() if n not in last_layer_param_names]
    
    # total num of params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    optimizer = torch.optim.RMSprop(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": last_layer_params, "weight_decay": last_dense_wd}
        ],
        lr=lr
    )
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = torch.nn.DataParallel(model)
    model.train()
    # loss function: mse (regress)
    loss_fn = torch.nn.MSELoss()
    def collate_fn(batch):
        # -> tensor
        # batch is a list of dictionaries
        # each dictionary has "grid" and "label" keys
        # grid is a tensor of shape (1, 20, 20, 20, 28)
        # print(type(batch[0]["grid"]))
        # exit()
        grids = torch.stack([torch.tensor(item["grid"], dtype=torch.float32) for item in batch]) # (64, 20, 20, 20, 28)
        # -> (64, 28, 20, 20, 20)
        grids = grids.permute(0, 4, 1, 2, 3)
        return {
            "grid": grids,
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1) # (64, 1)
        }
    # DataLoader for batching
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
    )
    
    # best_valid_loss = float("inf")
    best_metric = float("-inf")
    best_model = None
    # training loop
    global_step = 0
    for epoch in range(n_epochs):
        print("-" * 50)
        print(f"Epoch {epoch+1}/{n_epochs}")
        pbar = tqdm(total=len(train_loader), desc="Training")
        for batch in train_loader:
            grids = batch["grid"].to(device)
            # print("Sum of grids:", grids.sum())
            # NOTE: NORMALIZE
            labels = batch["label"].to(device) / 15.
            optimizer.zero_grad()
            outputs = model(grids) # shape (64, 1)
            loss_mse = loss_fn(outputs, labels)
            # loss pearson
            # pearson penalty
            pearson_penalty = torch.corrcoef(torch.stack([outputs.squeeze(), labels.squeeze()]))[0, 1]
            loss = loss_mse - pearson_coeff * pearson_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
            wandb.log({
                "train_loss": loss.item(),
                "train_mse_loss": loss_mse.item(),
                "train_pearson_penalty": pearson_penalty.item(),
                "epoch": epoch,
                "global_step": global_step,
            }, step=global_step)
            global_step += 1
        pbar.close()
        print(f"Epoch {epoch+1}/{n_epochs} completed.")
        # Validation
        model.eval()
        valid_losses = []
        valid_pred = []
        valid_labels = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                grids = batch["grid"].to(device)
                # NOTE: NORMALIZE
                labels = batch["label"].to(device) / 15.
                outputs = model(grids)
                valid_pred.append(outputs.cpu().numpy())
                valid_labels.append(labels.cpu().numpy())
                loss = loss_fn(outputs, labels)
                valid_losses.append(loss.item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        # pearson
        valid_pred = np.concatenate(valid_pred)
        valid_labels = np.concatenate(valid_labels)
        # Calculate Pearson correlation coefficient
        valid_pearson = np.corrcoef(valid_pred.flatten(), valid_labels.flatten())[0, 1]
        valid_spearman = spearmanr(valid_pred.flatten(), valid_labels.flatten()).correlation
        valid_c_index = concordance_index(valid_labels.flatten(), valid_pred.flatten())
        wandb.log({
            "valid_loss": avg_valid_loss,
            "valid_pearson_corr": valid_pearson,
            "valid_spearman_corr": valid_spearman,
            "valid_c_index": valid_c_index,
            "epoch": epoch,
        }, step=global_step)
        model.train()
        # Save the best model
        
        # DONE: CHANGED TO PEARSON COMPARISON
        # if avg_valid_loss < best_valid_loss:
        #     best_valid_loss = avg_valid_loss
        print("Pearson Correlation Coefficient:", valid_pearson)
        print("Spearman Correlation Coefficient:", valid_spearman)
        print("Concordance Index:", valid_c_index)
        if valid_c_index > best_metric:
            best_metric = valid_c_index
            best_model = model.state_dict()
            print("Best model updated w/ C-index:", best_metric)
    # Save the best model
    if best_model is not None:
        os.makedirs("ckpt", exist_ok=True)
        torch.save(best_model, "ckpt/best_model.pth")
        print("Best model saved as best_model.pth")
    else:
        raise ValueError("No model was saved. Check the training process.")
    return best_model
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--data_path", type=str, default="./data/ordinary_dataset", help="Path to the dataset")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--last_dense_wd", type=float, default=0.01, help="Weight decay for the last dense layer")
    # pearson corr penalty
    parser.add_argument("--pearson_penalty_coeff", type=float, default=0.1, help="Pearson correlation penalty")
    args = parser.parse_args()
    # train the model
    wandb.init(
        project="sfcnn-protein-ligand-binding-affinity",
        config={
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "last_dense_wd": args.last_dense_wd,
            "pearson_penalty_coeff": args.pearson_penalty_coeff,
        },
        name=f"lr{args.lr}_dropout{args.dropout}_batch{args.batch_size}_epochs{args.n_epochs}_wd{args.last_dense_wd}_pearson_coeff{args.pearson_penalty_coeff}",
    )
    model = train(
        data_path = args.data_path,
        lr = args.lr,
        dropout = args.dropout,
        last_dense_wd = args.last_dense_wd,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
        pearson_coeff = args.pearson_penalty_coeff,
    )
    
