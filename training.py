from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from data_prepration import tweets_preprosesing
import click
import torch
import torch.nn as nn
from experiment import experiment_initilization
from evaluater import evaluation
from config_loader import load_config

configs = load_config()



@click.command()
@click.argument("data_path")
def training(data_path):
    with open(f"{configs['log_dir']}/prediction_history.txt", "w") as f:
        f.write("inputs\tprediction\tconfident\tlabel\ttimestamp")
    
    
    
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-base-arabic")
    model.classifier = nn.Linear(model.classifier.in_features, 4)
    train_dataset, test_dataset = tweets_preprosesing(data_path, tokenizer)
    

    best_param = experiment_initilization(model, tokenizer, train_dataset, test_dataset)
    
    
    trainerarg = TrainingArguments(
        output_dir=configs["checkpoint_dir"],
        evaluation_strategy = "epoch",
        save_strategy="steps",
        num_train_epochs = configs["num_train_epochs"],
        logging_dir = 'logging/',
        logging_steps = configs["logging_steps"],
        save_steps= configs["save_steps"],
        learning_rate = best_param["lr"],
        weight_decay = best_param["weight_decay"],
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
    trainer.save_model(output_dir=configs["model_dir"])


if __name__ == "__main__":
    training()