from train import train
from copy import deepcopy
import optuna
import wandb
import argparse
import datetime
import functools

def objective(trial: optuna.Trial,
              total_cpus, n_jobs):
    # Suggest hyperparameters
    # lr: 对数均匀分布，例如 1e-5 到 1e-2
    # lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    # # dropout: 均匀分布，例如 0.0 到 0.5
    # dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    # # last_dense_wd: 对数均匀分布，例如 1e-4 到 1e-1
    # last_dense_wd = trial.suggest_loguniform("last_dense_wd", 1e-4, 1e-1)

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    last_dense_wd = trial.suggest_float("last_dense_wd", 1e-4, 1e-1, log=True)
    # batch_size = trial.suggest_int("batch_size", 16, 128, step=16) 
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # Calculate num_workers per trial
    # 确保至少分配 1 个 worker
    num_workers_for_trial = max(1, total_cpus // n_jobs)
    print(f"Trial {trial.number}: Using {num_workers_for_trial} num_workers per trial (Total CPUs: {total_cpus}, Optuna Jobs: {n_jobs})")

    # Initialize WandB for this specific trial
    # Use trial number in the run name to distinguish
    # today = datetime.datetime.today()
    # now_minute = today.strftime("%Y-%m-%d %H:%M")
    # wandb.init(
    #     project="sfcnn-protein-ligand-binding-affinity-optuna",
    #     config={
    #         "learning_rate": lr,
    #         "dropout": dropout,
    #         "batch_size": 64, # Keep batch_size fixed for tuning speed? Or make it tunable?
    #         "n_epochs": 200, # Keep n_epochs fixed for fair comparison? Or reduce for tuning speed?
    #         "last_dense_wd": last_dense_wd,
    #         "num_workers_per_trial": num_workers_for_trial, # Log allocated workers
    #         "optuna_trial_number": trial.number, # Log trial number
    #     },
    #     # Use a unique name for each trial
    #     name=f"trial_{trial.number}_lr{lr:.0e}_do{dropout:.2f}_wd{last_dense_wd:.0e}-{now_minute}",
    #     reinit=True # Allows reinitialization in the same process if needed (though separate processes are typical for n_jobs > 1)
    # )

    # Call the train function with suggested hyperparameters and calculated num_workers
    # You might want to reduce n_epochs or batch_size here for faster tuning if needed
    _, pearson_correlation = train(
        data_path="./data/ordinary_dataset", # Fixed parameters
        init_lr=1e-4, # Fixed init_lr
        warmup_steps=200, # Fixed warmup_steps
        max_grad_norm=3.0, # Fixed max_grad_norm
        # Tunable parameters
        lr=lr,
        dropout=dropout,
        last_dense_wd=last_dense_wd,
        # Calculated parameters
        num_workers=num_workers_for_trial,
        # Trial identifier
        do_optuna=True,
        trial_number=trial.number,
        # Consider reducing epochs/batch size for tuning speed:
        n_epochs=args.n_epochs, # Example: Reduce epochs for tuning
        batch_size=batch_size, # Example: Use batch size from trial
    )

    # The train function saves the best model internally based on its own run.
    # Optuna's goal is just to find the best *hyperparameters*.
    # We return the metric Optuna should optimize.
    
    print(f"Trial {trial.number} completed: params={trial.params}, pearson={pearson_correlation:.4f}")
    return pearson_correlation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter tuning.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials to run.")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel Optuna jobs (processes). Use > 1 for multiprocessing.")
    parser.add_argument("--total_cpus", type=int, default=8, help="Total number of CPU cores allocated by Slurm for this task.")
    parser.add_argument("--n_epochs", type=int, default=120, help="Number of epochs for training.")
    args = parser.parse_args()

    # Define total available CPUs from Slurm allocation
    total_cpus_available = args.total_cpus
    # Define the number of parallel Optuna trials (jobs) to run
    n_optuna_jobs = args.n_jobs

    # Calculate num_workers per trial based on total CPUs and n_jobs
    # Each Optuna job (trial) will get a share of the total CPUs for its DataLoader workers
    if n_optuna_jobs > total_cpus_available:
         print(f"Warning: n_jobs ({n_optuna_jobs}) is greater than total_cpus ({total_cpus_available}). Setting n_jobs = total_cpus.")
         n_optuna_jobs = total_cpus_available

    if n_optuna_jobs <= 0:
         n_optuna_jobs = 1 # Ensure at least one job

    # The calculation of num_workers per trial is done inside the objective function
    # using total_cpus_available and n_optuna_jobs passed via functools.partial


    # Create the Optuna study
    # direction="maximize" because we want to maximize Pearson correlation
    study = optuna.create_study(direction="maximize", study_name="sfcnn_hpo_pearson")

    # Use functools.partial to pass fixed arguments (total_cpus, n_jobs) to the objective function
    # The `trial` object will be passed by Optuna during optimization
    objective_with_args = functools.partial(objective,
                                             total_cpus=total_cpus_available,
                                             n_jobs=n_optuna_jobs)


    print(f"Running Optuna study with {args.n_trials} trials and {n_optuna_jobs} parallel jobs.")
    print(f"Total CPUs allocated: {total_cpus_available}")
    print(f"Each trial will use approximately {max(1, total_cpus_available // n_optuna_jobs)} num_workers for DataLoader.")

    # Run the optimization
    # n_jobs > 1 enables multiprocessing
    study.optimize(objective_with_args, n_trials=args.n_trials, n_jobs=n_optuna_jobs)

    print("\n" + "="*50)
    print("Optuna Study finished!")
    print(f">> Best Pearson Correlation: {study.best_value:.4f}")
    print(">> Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)

    # You can also save the study results
    # study.trials_dataframe().to_csv("optuna_results.csv")