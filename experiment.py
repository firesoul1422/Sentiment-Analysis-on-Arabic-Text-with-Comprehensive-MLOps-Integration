import mlflow
import optuna
from transformers import Trainer, TrainingArguments
from evaluater import evaluation
from optuna.pruners import HyperbandPruner  # Add this import
from config_loader import load_config

configs = load_config()


def train(model, tokenizer, train_dataset, test_dataset,  lr, weight_decay):
    
    trainerarg = TrainingArguments(
        output_dir=configs["checkpoint_dir"],
        evaluation_strategy = "epoch",
        save_strategy="steps",
        num_train_epochs = configs["num_train_epochs"],
        logging_dir = 'logging/',
        logging_steps = configs["logging_steps"],
        save_steps= configs["save_steps"],
        learning_rate = lr,
        weight_decay = weight_decay,
        # lr_scheduler_type= "cosine",
    #     warmup_ratio=1e-4,
    #     warmup_steps=400,
        per_device_train_batch_size = configs["per_device_train_batch_size"],
        per_device_eval_batch_size  = configs["per_device_eval_batch_size"],
 
    )

    trainer = Trainer(
        model = model,
        args = trainerarg,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = evaluation,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.evaluate(test_dataset)

def objective(trial, model, tokenizer, train_dataset, test_dataset):

    weight_decay = trial.suggest_float('weight_decay', configs["weight_decay_range"][0], configs["weight_decay_range"][1])

    lr = trial.suggest_categorical('lr', configs["lr_values"])
    
    with mlflow.start_run(       
                run_name="Sentimintal_analysis_project_experiment 0",
                experiment_id=0) as exp:
        
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("weight_decay", weight_decay)
            
        results = train(model, tokenizer, train_dataset, test_dataset, lr, weight_decay)
        
        mlflow.log_metric("F1 score", results["eval_f1_score"])
        mlflow.log_metric("accuercy", results["eval_accuercy"])
        mlflow.log_metric("recall score", results["eval_recall_score"])
        mlflow.log_metric("precision_score", results["eval_precision_score"])
        
        return results["eval_f1_score"]


def experiment_initilization(model, tokenizer, train_dataset, test_dataset):
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='maximize', pruner=HyperbandPruner())
    study.optimize(lambda trial: objective(trial, model, tokenizer, train_dataset, test_dataset), n_trials=configs["10"])
    return study.best_params