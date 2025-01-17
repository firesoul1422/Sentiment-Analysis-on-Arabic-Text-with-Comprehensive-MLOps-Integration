import pandas as pd
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt
from datetime import datetime
from config_loader import load_config

configs = load_config()


def log_training_data(path):
    df = pd.read_csv(path, sep="\t", header= None, names=["tweets", "tag"])
    words_per_sentence = df["tweets"].apply(lambda tweet: len(tweet))
    average_words_per_sentence = words_per_sentence.mean()
    std_words_per_sentence = words_per_sentence.std()
    labels_count = df["tag"].value_counts()
    
    
    return average_words_per_sentence, std_words_per_sentence, labels_count, words_per_sentence



def log_predection_data(path):
    df = pd.read_csv(path, sep="\t")
    pred_words_per_sentence = df["inputs"].apply(lambda tweet: len(tweet))
    pred_average_words_per_sentence = pred_words_per_sentence.mean()
    pred_std_words_per_sentence = pred_words_per_sentence.std()
    pred_labels_count = df["labels"].value_counts()
    
    confident_mean = df["confident"].mean()
    confidents = df["confident"]
    
    return pred_average_words_per_sentence, pred_std_words_per_sentence, pred_labels_count, confident_mean, confidents, pred_words_per_sentence



def monitoring_calculation():
    average_words_per_sentence, std_words_per_sentence, labels_count, words_per_sentence = log_training_data(configs["training_data_path"])
    pred_average_words_per_sentence, pred_std_words_per_sentence, pred_labels_count, confident_mean, confidents, pred_words_per_sentence = log_predection_data(configs["prediction_log_path"])
    
    
    ks_statistic, p_value = ks_2samp(words_per_sentence, pred_words_per_sentence)
    emd = wasserstein_distance(words_per_sentence, pred_words_per_sentence)
    
    plt.hist(confidents)
    plt.xlabel("confidents")
    plt.savefig(f"{configs['log_dir']}\{configs['graph_dir']}\confident_{datetime.now()}.png")
    
    plt.hist(words_per_sentence)
    plt.xlabel("words_per_sentence")
    plt.savefig(f"{configs['log_dir']}\{configs['graph_dir']}\words_per_sentence_{datetime.now()}.png")
    
    
    plt.hist(pred_words_per_sentence)
    plt.xlabel("pred_words_per_sentence")
    plt.savefig(f"{configs['log_dir']}\{configs['graph_dir']}\pred_words_per_sentence_{datetime.now()}.png")


    
    
    pd.DataFrame(data= {'ks_statistic': ks_statistic, 'p_value': p_value, "wasserstein_distance": emd, "average_words_per_sentence": average_words_per_sentence, "pred_average_words_per_sentence": pred_average_words_per_sentence, "std_words_per_sentence": std_words_per_sentence, "pred_std_words_per_sentence": pred_std_words_per_sentence, "confident_mean": confident_mean}).to_csv(f"{configs['log_dir']}\{configs['report_dir']}\_full_report_{datetime.now()}.csv")
    