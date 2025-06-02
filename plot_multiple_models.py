from plot_label_preds_ttaug import eval_on_named_dataset

def multiple_evaldict_variance_plot(model_names: list, plot_path: str):
    l_of_dicts = []
    for model_name in model_names:
        real_dict = eval_on_named_dataset(
            f"ckpt/{model_name}/best_model.pth",
            "data/aug/real_test_aug",
            batch_size=32,
            num_workers=4
        )
        pred_dict = eval_on_named_dataset(
            f"ckpt/{model_name}/best_model.pth",
            "data/aug/pred_test_aug",
            batch_size=32,
            num_workers=4
        )
        l_of_dicts.append((real_dict, pred_dict))
        
    real_set_pla_dict = {} # key: name, value: list of preds
    pred_set_pla_dict = {} # key: name, value: list of preds
    label_dict = {}
    # [0] and [1]
    intersect_names = set(l_of_dicts[0][0].keys()).intersection(set(l_of_dicts[0][1].keys()))
    for name in intersect_names:
        label_dict[name] = l_of_dicts[0][0][name]["label"]
        # real_set_pla_dict[name] = [l_of_dicts[i][0][name]["preds"] for i in range(len(l_of_dicts))]
        # concat the preds
        real_set_pla_dict[name] = [item for sublist in [l_of_dicts[i][0][name]["preds_list"] for i in range(len(l_of_dicts))] for item in sublist]
        pred_set_pla_dict[name] = [item for sublist in [l_of_dicts[i][1][name]["preds_list"] for i in range(len(l_of_dicts))] for item in sublist]
    # standard deviation of preds
    import numpy as np
    real_stds = [np.std(np.array(real_set_pla_dict[name]), axis=0) for name in intersect_names]
    real_stds = np.array(real_stds).squeeze()
    pred_stds = [np.std(np.array(pred_set_pla_dict[name]), axis=0) for name in intersect_names]
    pred_stds = np.array(pred_stds).squeeze()
    # violin plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.violinplot([real_stds, pred_stds], showmeans=True, showmedians=True)
    plt.xticks([1, 2], ["Groundtruth Structure Evaluation", "Bootstrapped Evaluation"])
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation of Predictions for Multiple Models")
    plt.savefig(plot_path)
    plt.close()
    
if __name__ == "__main__":
    model_names = [
        "sfcnn_lr0.004_dropout0.1_wd0.01_bs32",
        "sfcnn_lr0.004_dropout0.5_wd0.01_bs32",
        "sfcnn_lr0.0016_dropout0.3_wd0.015_bs16"
    ]
    plot_path = "plots/aug/multiple_models_std_violin.png"
    multiple_evaldict_variance_plot(model_names, plot_path)