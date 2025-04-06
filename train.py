from model import build_model
import os
from datasets import Dataset as HFDataset
import torch
from tqdm import tqdm
import argparse
torch.set_printoptions(edgeitems=114514)
def train(data_path = "./data/ordinary_dataset",
          lr = 0.001,
          dropout = 0.5,
          last_dense_wd = 0.01,
          n_epochs = 5,
          batch_size = 64,
        ):
    train_dataset = HFDataset.load_from_disk(os.path.join(data_path, "train"))
    valid_dataset = HFDataset.load_from_disk(os.path.join(data_path, "valid"))
    
    model = build_model(dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    model.train()
    
    # Apply weight decay only to the last linear layer
    last_layer_params = list(model.fc2.parameters())
    last_layer_param_names = [n for n, p in model.named_parameters() if ("fc2" in n)]
    # print(f"Last layer parameters: {last_layer_param_names}")
    other_params = [p for n, p in model.named_parameters() if n not in last_layer_param_names]
    
    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": last_layer_params, "weight_decay": last_dense_wd}
        ],
        lr=lr
    )
    
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
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    best_valid_loss = float("inf")
    best_model = None
    # training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        pbar = tqdm(total=len(train_loader), desc="Training")
        for batch in train_loader:
            grids = batch["grid"].to(device)
            # print("Sum of grids:", grids.sum())
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(grids)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
        pbar.close()
        print(f"Epoch {epoch+1}/{n_epochs} completed.")
        # Validation
        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                grids = batch["grid"].to(device)
                labels = batch["label"].to(device)
                outputs = model(grids)
                loss = loss_fn(outputs, labels)
                valid_losses.append(loss.item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        model.train()
        # Save the best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model = model.state_dict()
            print("Best model updated.")
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
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--last_dense_wd", type=float, default=0.01, help="Weight decay for the last dense layer")
    args = parser.parse_args()
    # train the model
    model = train(
        data_path = args.data_path,
        lr = args.lr,
        dropout = args.dropout,
        last_dense_wd = args.last_dense_wd,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
    )
    