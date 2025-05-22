import os
import torch
from datasets import Dataset as HFDataset
from model import build_model
from tqdm import tqdm

def eval_on_named_dataset(model_path,
    data_path = "data/ordinary_dataset/test",
    batch_size = 64,
    num_workers = 4,
    ):
    test_dataset = HFDataset.load_from_disk(data_path)
    test_dataset.set_format(type="torch", columns=["grid", "label", "name"])
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
            for name, pred, label in zip(names, outputs.cpu().numpy(), labels.cpu().numpy()):
                eval_dict[name] = { # denormalize
                    "pred": pred * 15,
                    "label": label * 15,
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

def plot_2_dicts(eval_dict1, eval_dict2, predplot_path, mse_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # Find common keys
    common_names = set(eval_dict1.keys()) & set(eval_dict2.keys())
    common_names = list(common_names)

    pred1, label12, pred2 = [], [], []
    for name in sorted(common_names):
        pred1.append(eval_dict1[name]["pred"])
        label12.append(eval_dict1[name]["label"])
        pred2.append(eval_dict2[name]["pred"])
        assert eval_dict1[name]["label"] == eval_dict2[name]["label"]
        # label2.append(eval_dict2[name]["label"])

    pred1 = np.array(pred1)
    label12 = np.array(label12)
    pred2 = np.array(pred2)
    # label2 = np.array(label2)

    # Plot the predictions
    plt.figure(figsize=(10, 10))
    plt.scatter(pred1, label12, alpha=0.75, label="Ordinary Dataset")
    plt.scatter(pred2, label12, alpha=0.75, label="Heuristic Dataset")
    plt.scatter(label12, label12, alpha=0.75, label="Labels")
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title("Predictions vs True Labels (n = {})".format(len(common_names)))
    plt.legend()
    plt.savefig(predplot_path)
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    # Plot the MSE of Ordinary Dataset & Heuristic Dataset
    mse1 = np.array([eval_dict1[name]["mse"] for name in common_names])
    mse2 = np.array([eval_dict2[name]["mse"] for name in common_names])
    plt.figure(figsize=(10, 10))
    plt.scatter(mse1, mse2, alpha=0.75)
    plt.xlabel("Ordinary Dataset MSE")
    plt.ylabel("Heuristic Dataset MSE")
    plt.title("MSE of Ordinary Dataset vs Heuristic Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path)
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    
    # violin graph about mse1 and mse2
    plt.figure(figsize=(10, 10))
    plt.violinplot([mse1, mse2], showmeans=True)
    plt.xticks([1, 2], ["Ordinary Dataset", "Heuristic Dataset"])
    plt.ylabel("MSE")
    plt.title("MSE of Ordinary Dataset vs Heuristic Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path.replace(".png", "_violin.png"))
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    
    # violin graph of label-pred gaps
    plt.figure(figsize=(10, 10))

    plt.violinplot([abs(pred1.squeeze(1) - label12.squeeze(1)), abs(pred2.squeeze(1) - label12.squeeze(1))], showmeans=True)
    plt.xticks([1, 2], ["Ordinary Dataset", "Heuristic Dataset"])
    plt.ylabel("Label-Pred Gap")
    plt.title("Label-Pred Gap of Ordinary Dataset vs Heuristic Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path.replace(".png", "_label_pred_gap_violin.png"))
    plt.tight_layout()
    
def main():
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/sfcnn_lr0.004_dropout0.1_wd0.01_bs32/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    real_eval_dict = eval_on_named_dataset(
        model_path=args.model_path,
        data_path="data/ordinary_dataset/test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pred_eval_dict_heu = eval_on_named_dataset(
        model_path=args.model_path,
        data_path="data/heuristic_pred_test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    plot_2_dicts(
        real_eval_dict,
        pred_eval_dict_heu,
        predplot_path="visual/real_vs_heu_preds.png",
        mse_path="visual/real_vs_heu_mse.png",
    )
    
if __name__ == "__main__":
    main()