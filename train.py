from model import build_model
import os
from datasets import Dataset as HFDataset
import torch
from tqdm import tqdm
import argparse
import numpy as np
import wandb
from copy import deepcopy
from scipy.stats import spearmanr
from lifelines.utils import concordance_index
DEBUG = False

def train(data_path = "./data/ordinary_dataset",
          init_lr = 1e-4,
          lr = 4e-3,
          warmup_steps = 200,
          dropout = 0.1,
          last_dense_wd = 0.01,
          n_epochs = 200,
          batch_size = 64,
          max_grad_norm = 3.0,
          num_workers = 16,
        ):
    train_dataset = HFDataset.load_from_disk(os.path.join(data_path, "train"))
    train_dataset.set_format(type="torch", columns=["grid", "label"])
    valid_dataset = HFDataset.load_from_disk(os.path.join(data_path, "valid"))
    valid_dataset.set_format(type="torch", columns=["grid", "label"])
    # torch.backends.cudnn.benchmark = True
    model = build_model(dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Apply weight decay only to the last linear layer
    last_layer_params = [p for n,p in model.named_parameters() if n=="fc2.weight"]
    # last_layer_param_names = [n for n, p in model.named_parameters() if n=="fc2.weight"]
    # print(f"Last layer parameters: {last_layer_param_names}")
    other_params = [p for n, p in model.named_parameters() if n!="fc2.weight"]
    # total num of params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    optimizer = torch.optim.RMSprop(
        [
            {"params": other_params, "weight_decay": 0.0},
            {"params": last_layer_params, "weight_decay": last_dense_wd}
        ],
        lr=lr,
    )
    # warmup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (float(step) / warmup_steps * (lr - init_lr) + init_lr) / lr),
    )
    # if DEBUG:
    #     for n, m in model.named_modules():
    #         if isinstance(m, torch.nn.Conv3d):
    #             m.register_forward_hook(_shape_hook(n))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        model = torch.nn.DataParallel(model)
    model.train()

    # loss function: mse (regress)
    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()
    # def collate_fn(batch):
    #     # -> tensor
    #     # batch is a list of dictionaries
    #     # each dictionary has "grid" and "label" keys
    #     # grid is a tensor of shape (1, 20, 20, 20, 28)
    #     # print(type(batch[0]["grid"]))
    #     # exit()
    #     grids = torch.stack([torch.tensor(item["grid"], dtype=torch.float32) for item in batch]) # (64, 20, 20, 20, 28)
    #     # -> (64, 28, 20, 20, 20)
    #     grids = grids.permute(0, 4, 1, 2, 3)
    #     return {
    #         "grid": grids,
    #         "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1) # (64, 1)
    #     }
    # DataLoader for batching
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=lambda batch: {
        #     "grid": torch.stack([torch.FloatTensor(item["grid"]) for item in batch]), # (64, 20, 20, 20, 28)
        #     "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1) # (64, 1)
        # },
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        # collate_fn=lambda batch: {
        #     "grid": torch.stack([item["grid"] for item in batch]), # (64, 20, 20, 20, 28)
        #     "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1) # (64, 1)
        # },
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )
    
    best_valid_mae = float("inf")
    best_valid_pearson = -1.0
    best_model = None
    # training loop
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(n_epochs):
        print("-" * 50)
        print(f"Epoch {epoch+1}/{n_epochs}")
        pbar = tqdm(total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}", disable=False)
        import time
        starttime = time.time()
        for batch in train_loader:
            print(f"Load time: {time.time() - starttime:.5f}s")
            starttime = time.time()
            
            model.train()
            grids = batch["grid"].to(device) # shape (64, 20, 20, 20, 28)
            # normalize the grid
            grids = (grids - grids.mean()) / grids.std()
            # print(f"batch size this step = {grids.size(0)}")
            # print("Sum of grids:", grids.sum())
            # NOTE: NORMALIZE
            labels = batch["label"].to(device) / 15.
            
            starttime = time.time()
            outputs = model(grids.permute(0, 4, 1, 2, 3)) # shape (64, 1)
            print(f"Forward time: {time.time() - starttime:.5f}s")
            starttime = time.time()
            loss = mse_loss_fn(outputs, labels)
            loss.backward()

            norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # if global_step < 5:
            #     with torch.no_grad():
            #         print(f"\n=== Step {global_step} debug ===")
            #         print("  labels  :", labels.view(-1).cpu().tolist()[:10])
            #         print("  outputs :", outputs.view(-1).cpu().tolist()[:10])
            #         print("  grid.min/max:", grids.min().item(), grids.max().item())
            #         print("  grad_norm_before_clip:", norm_before)
            #         print("  lr:", optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)
            print(f"Backward time: {time.time() - starttime:.5f}s")
            print("Loss:", loss.item())
            pbar.set_postfix(loss=loss.item())
            wandb.log({
                "train_mse_loss": loss.item(),
                "epoch": epoch,
                "global_step": global_step,
                "original_grad_norm": norm_before,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)
            global_step += 1
        pbar.close()
        print(f"Epoch {epoch+1}/{n_epochs} completed.")
        # Validation

        with torch.no_grad():
            model.eval()
            valid_maes = []
            valid_pred = []
            valid_labels = []
            for batch in tqdm(valid_loader, desc="Validating"):
                grids = batch["grid"].to(device) # should permute (0, 4, 1, 2, 3)
                # NOTE: NORMALIZE
                labels = batch["label"].to(device) / 15.
                outputs = model(grids.permute(0, 4, 1, 2, 3)) # shape (64, 1)
                valid_pred.append(outputs.cpu())
                valid_labels.append(labels.cpu())
                mse = mse_loss_fn(outputs, labels)
                mae = mae_loss_fn(outputs, labels)
                valid_maes.append(mae.item())
            valid_mae = sum(valid_maes) / len(valid_maes)
            print(f"Validation Loss: {valid_mae:.4f}")
            # pearson
            valid_pred = np.concatenate(valid_pred)
            valid_labels = np.concatenate(valid_labels)
            # Calculate Pearson correlation coefficient
            
            if valid_pred.std() < 1e-6 or valid_labels.std() < 1e-6:
                print("Warning: std of prediction or labels is too small. Setting Pearson correlation to 0.")
                valid_pearson = 0.0
            else:
                valid_pearson = np.corrcoef(valid_pred.flatten(), valid_labels.flatten())[0, 1]
            valid_spearman = spearmanr(valid_pred.flatten(), valid_labels.flatten()).correlation
            valid_c_index = concordance_index(valid_labels.flatten(), valid_pred.flatten())
            
            # Save the best model
            print("Pearson Correlation:", valid_pearson)
            print("Spearman Correlation:", valid_spearman)
            print("Concordance Index:", valid_c_index)
            # if valid_mae < best_valid_mae:
            #     best_valid_mae = valid_mae
            # if valid_c_index > best_metric:
            #     best_metric = valid_c_index
            if valid_pearson > best_valid_pearson:
                best_valid_pearson = valid_pearson
                best_model = deepcopy(model.state_dict()) if not isinstance(model, torch.nn.DataParallel) else deepcopy(model.module.state_dict())
                print("Best model updated w/ Valid MAE:", best_valid_mae)
            wandb.log({
                "valid_mse_loss": valid_mae,
                "valid_pearson_corr": valid_pearson,
                "valid_spearman_corr": valid_spearman,
                "valid_c_index": valid_c_index,
                "valid_mae_loss": valid_mae,
                "epoch": epoch,
            }, step=(epoch + 1) * len(train_loader))
    # Save the best model
    if best_model is not None:
        os.makedirs(f"ckpt/sfcnn_lr{lr}_dropout{dropout}_wd{last_dense_wd}", exist_ok=True)
        torch.save(best_model, "ckpt/best_model.pth")
        print("Best model saved as best_model.pth")
    else:
        raise ValueError("No model was saved. Check the training process.")
    return best_model
    
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--data_path", type=str, default="./data/ordinary_dataset", help="Path to the dataset")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--last_dense_wd", type=float, default=0.01, help="Weight decay for the last dense layer")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
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
        },
        name=f"lr{args.lr}_dropout{args.dropout}_batch{args.batch_size}_epochs{args.n_epochs}_wd{args.last_dense_wd}",
    )
    model = train(
        data_path = args.data_path,
        lr = args.lr,
        dropout = args.dropout,
        last_dense_wd = args.last_dense_wd,
        n_epochs = args.n_epochs,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )
    
