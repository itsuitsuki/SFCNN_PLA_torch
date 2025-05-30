import os
import torch
from datasets import Dataset as HFDataset
from model import build_model
from tqdm import tqdm
import numpy as np

def eval_on_named_dataset(model_path,
    data_path = "data/ordinary_dataset/test",
    batch_size = 8,
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
            grids = batch["grid"].to(device).float() # (bs, 10, 28, 20, 20, 20)
            # NOTE: NORMALIZE
            # flatten 0,1 dim
            grids = grids.view(grids.size(0) * grids.size(1), *grids.size()[2:])
            labels = batch["label"].to(device) / 15.
            labels = labels.unsqueeze(1)
            outputs = model(grids) # (bs * 10, 1)
            outputs = outputs.view(batch["grid"].size(0), -1) # (bs, 10)
            # std of outputs
            std_outputs = outputs.std(dim=1, keepdim=True) # (bs, 1)
            # outputs_mean = outputs.mean(dim=1).unsqueeze(1) # (bs, 1)
            plddt = batch.get("pLDDT", None)
            names = batch["name"]
            if plddt is not None:
                for name, pred, label, std, plddt in zip(names, outputs.cpu().numpy(), labels.cpu().numpy(), std_outputs.cpu().numpy(), plddt.cpu().numpy()):
                    eval_dict[name] = { # denormalize
                        "preds_list": pred * 15,
                        "pred": np.mean(pred, keepdims=True) * 15,  # denormalize mean 
                        "label": label * 15,
                        "std": std * 15,  # denormalize std
                        "plddt": plddt,  # pLDDT value
                    }
            else:
                for name, pred, label, std in zip(names, outputs.cpu().numpy(), labels.cpu().numpy(), std_outputs.cpu().numpy()):
                    eval_dict[name] = { # denormalize
                        "preds_list": pred * 15,
                        "pred": np.mean(pred, keepdims=True) * 15,  # denormalize mean
                        "label": label * 15,
                        "std": std * 15,
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

if __name__ == "__main__":
    model_name = "sfcnn_lr0.004_dropout0.1_wd0.01_bs32"
    model_path = f"ckpt/{model_name}/best_model.pth"
    data_path_real = "data/aug/real_test_aug"
    data_path_pred = "data/aug/pred_test_aug"
    real_edict = eval_on_named_dataset(model_path, data_path_real, batch_size=32, num_workers=4)
    pred_edict = eval_on_named_dataset(model_path, data_path_pred, batch_size=32, num_workers=4)
    from plot_label_preds import plot_bootstrap_ordinary_preds
    plot_bootstrap_ordinary_preds(real_edict, pred_edict, 
        predplot_path=f"plots/aug/{model_name}/real_vs_heu_preds.png",
        mse_path=f"plots/aug/{model_name}/real_vs_heu_mse.png"
    )
    # plot std violin
    import matplotlib.pyplot as plt
    import numpy as np
    real_stds = [v["std"] for v in real_edict.values() if "std" in v]
    real_stds = np.array(real_stds).squeeze()
    
    pred_stds = [v["std"] for v in pred_edict.values() if "std" in v]
    pred_stds = np.array(pred_stds).squeeze()
    # print(real_stds.shape, pred_stds.shape)
    plt.figure(figsize=(10, 10))
    plt.violinplot([real_stds, pred_stds], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["Real Test", "Heuristic Test"])
    plt.ylabel("Standard Deviation of Predictions")
    plt.title(f"Distribution of Standard Deviations of Predictions for {model_name}")
    plt.savefig(f"plots/aug/{model_name}/std_violin.png")
    plt.tight_layout()
    plt.clf()
    plt.close("all")