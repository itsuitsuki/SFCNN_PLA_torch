import os
import torch
from datasets import Dataset as HFDataset
from model import build_model
from tqdm import tqdm


def eval_on_bootstrap_dataset(model_path,
                            data_path ="data/heuristic_pred_test",
                            batch_size=64,
                            num_workers=4,
                            ):
    test_dataset = HFDataset.load_from_disk(data_path)
    test_dataset.set_format(type="torch", columns=["grid", "label", "name", "pLDDT"])
    # Load the model
    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    eval_dict = {}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            grids = batch["grid"].to(device).float()
            # NOTE: NORMALIZE
            labels = batch["label"].to(device) / 15.
            labels = labels.unsqueeze(1)
            outputs = model(grids)
            names = batch["name"]
            plddt = batch["pLDDT"] if "pLDDT" in batch else None
            for name, pred, label, plddt in zip(names, outputs.cpu().numpy(), labels.cpu().numpy(), plddt.cpu().numpy() if plddt is not None else None):
                eval_dict[name] = { # denormalize
                    "pred": pred * 15,
                    "label": label * 15,
                    "pLDDT": plddt,
                }
    for name in eval_dict:
        pred = eval_dict[name]["pred"]
        label = eval_dict[name]["label"]
        mse = ((pred - label) ** 2).mean()
        mae = (abs(pred - label)).mean()
        eval_dict[name].update({
            "mse": mse,
            "mae": mae,
        })
    return eval_dict

def plot_plddt_and_mse(eval_dict, visual_path="./plots/"):
    import matplotlib.pyplot as plt
    import numpy as np

    plddt = []
    mse = []
    names = []

    for name, metrics in eval_dict.items():
        plddt.append(metrics["pLDDT"])
        mse.append(metrics["mse"])
        names.append(name)

    # Convert to numpy arrays
    plddt = np.array(plddt)
    mse = np.array(mse)

    # Plot pLDDT vs MSE
    plt.figure(figsize=(10, 6))
    plt.scatter(plddt, mse, alpha=0.5)
    plt.title("pLDDT vs MSE in Bootstrapped Evaluation")
    plt.xlabel("pLDDT")
    plt.ylabel("MSE")
    
    # Save the plot
    os.makedirs(visual_path, exist_ok=True)
    plt.savefig(os.path.join(visual_path, "plddt_vs_mse.png"))
    plt.close()

    # Plot label gap (absolute difference between prediction and label)
    label_gap = [(metrics["pred"] - metrics["label"]).mean() for metrics in eval_dict.values()]
    label_gap = np.array(label_gap)

    plt.figure(figsize=(10, 6))
    plt.scatter(plddt, label_gap, alpha=0.5, color='orange')
    plt.title("pLDDT vs Label Gap on Bootstrapped Dataset")
    plt.xlabel("pLDDT")
    plt.ylabel("Label Gap (Signed MAE)")
    
    # Save the plot
    plt.savefig(os.path.join(visual_path, "plddt_vs_label_gap.png"))
    plt.close()
if __name__ == "__main__":
    model_name = "sfcnn_lr0.0016_dropout0.3_wd0.015_bs16"
    model_path = f"ckpt/{model_name}/best_model.pth"  # Update with your model path
    data_path = "data/heuristic_pred_test"  # Update with your data path
    eval_dict = eval_on_bootstrap_dataset(model_path, data_path)
    os.makedirs(f"./plots/{model_name}/", exist_ok=True)
    plot_plddt_and_mse(eval_dict, visual_path=f"./plots/{model_name}/")
    print("Evaluation and plotting completed.")