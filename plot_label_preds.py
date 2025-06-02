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

def plot_bootstrap_ordinary_preds(eval_dict1, eval_dict2, predplot_path, mse_path):
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
    
    # 3 pearson
    from scipy.stats import pearsonr
    boostrapped_label_pred_corr = pearsonr(pred2.squeeze(1), label12.squeeze(1))
    ordinary_label_pred_corr = pearsonr(pred1.squeeze(1), label12.squeeze(1))
    mutual_pred_corr = pearsonr(pred1.squeeze(1), pred2.squeeze(1))
    # print
    print(f"Groundtruth Dataset Label-Prediction Correlation: {ordinary_label_pred_corr[0]:.4f}, p-value: {ordinary_label_pred_corr[1]:.4f}")
    print(f"Bootstrapped Dataset Label-Prediction Correlation: {boostrapped_label_pred_corr[0]:.4f}, p-value: {boostrapped_label_pred_corr[1]:.4f}")
    print(f"Mutual Prediction Correlation: {mutual_pred_corr[0]:.4f}, p-value: {mutual_pred_corr[1]:.4f}")
    # Plot the predictions
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    # Ordinary Dataset subplot
    axes[0].scatter(pred1, label12, alpha=0.75, label="Groundtruth Dataset")
    axes[0].plot([0, 15], [0, 15], color='black', linestyle='--', label="Labels")
    axes[0].set_title("Groundtruth Dataset")
    axes[0].set_xlabel("Predictions")
    axes[0].set_ylabel("True Labels")
    axes[0].legend()

    # Heuristic Dataset subplot
    axes[1].scatter(pred2, label12, alpha=0.75, label="Bootstrapped Dataset")
    axes[1].plot([0, 15], [0, 15], color='black', linestyle='--', label="Labels")
    axes[1].set_title("Bootstrapped Dataset")
    axes[1].set_xlabel("Predictions")
    axes[1].legend()

    plt.suptitle("Predictions vs True Labels (n = {})".format(len(common_names)))
    plt.legend()
    os.makedirs(os.path.dirname(predplot_path), exist_ok=True)
    plt.savefig(predplot_path)
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    # Plot the MSE of Ordinary Dataset & Heuristic Dataset
    mse1 = np.array([eval_dict1[name]["mse"] for name in common_names])
    mse2 = np.array([eval_dict2[name]["mse"] for name in common_names])
    mse_gap = mse2 - mse1 # heuristic - ordinary
    # find big mse2 but small mse1 indices
    big_mse2_indices = np.where((mse2 > 10) & (mse1 < 2.5) & (mse_gap > 10))[0]
    plt.figure(figsize=(10, 10))
    plt.scatter(mse1, mse2, alpha=0.75, label="Normal Points")
    # big mse2 indices then plot them in red and their structure names on the graph
    for i in big_mse2_indices:
        plt.text(mse1[i], mse2[i], common_names[i], fontsize=8, color='black')
    plt.scatter(mse1[big_mse2_indices], mse2[big_mse2_indices], color='red', label='Big MSE Gap')
    # legend for red
    plt.legend()
    plt.xlabel("Groundtruth Dataset MSE")
    plt.ylabel("Bootstrapped Dataset MSE")
    plt.title("MSE of Groundtruth Dataset vs Bootstrapped Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path)
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    
    # plot the 10 largest mse2-mse1 indices and names
    # Sort indices by mse_gap in descending order
    largest_gap_indices = np.argsort(mse_gap)[-10:]
    largest_gap_names = [common_names[i] for i in largest_gap_indices]
    largest_gap_values = mse_gap[largest_gap_indices]

    # Plot the 10 largest mse gaps
    plt.figure(figsize=(10, 6))
    plt.barh(largest_gap_names, largest_gap_values, color='orange')
    plt.xlabel("MSE Gap (Bootstrapped - Groundtruth)")
    plt.ylabel("Structure Names")
    plt.title("Top 10 Largest MSE Gaps")
    plt.tight_layout()
    plt.savefig(mse_path.replace(".png", "_largest_mse_gap.png"))
    plt.clf()
    plt.close("all")
    
    # violin graph about mse1 and mse2
    plt.figure(figsize=(10, 10))
    plt.violinplot([mse1, mse2], showmeans=True)
    plt.xticks([1, 2], ["Groundtruth Dataset", "Bootstrapped Dataset"])
    plt.ylabel("MSE")
    plt.title("MSE of Groundtruth Dataset vs Bootstrapped Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path.replace(".png", "_violin.png"))
    plt.tight_layout()
    plt.clf()
    plt.close("all")
    
    # violin graph of label-pred gaps
    plt.figure(figsize=(10, 10))

    plt.violinplot([abs(pred1.squeeze(1) - label12.squeeze(1)), abs(pred2.squeeze(1) - label12.squeeze(1))], showmeans=True)
    plt.xticks([1, 2], ["Groundtruth Dataset", "Bootstrapped Dataset"])
    plt.ylabel("Label-Pred Gap")
    plt.title("Label-Pred Gap of Groundtruth Dataset vs Bootstrapped Dataset (n = {})".format(len(common_names)))
    plt.savefig(mse_path.replace("_mse.png", "_label_pred_gap_violin.png"))
    plt.tight_layout()
    
def main():
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ckpt/sfcnn_lr0.004_dropout0.5_wd0.01_bs32/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    # ckpt/{model_name}/best_model.pth
    model_name = os.path.basename(os.path.dirname(args.model_path))
    os.makedirs(f"plots/{model_name}", exist_ok=True)
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
    
    plot_bootstrap_ordinary_preds(
        real_eval_dict,
        pred_eval_dict_heu,
        predplot_path=f"plots/{model_name}/real_vs_heu_preds.png",
        mse_path=f"plots/{model_name}/real_vs_heu_mse.png",
    )
    
if __name__ == "__main__":
    main()