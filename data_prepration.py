import pandas as pd 
import re
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X.iloc[idx], self.y.iloc[idx]
        encoding = self.tokenizer(
            x,
            padding="max_length",  # Pad to max_length
            truncation=True,       # Truncate to max_length
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Convert to tensors
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)  # Remove batch dimension
        label = torch.tensor(y, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

  
    
def punctuation_removal(text):
    pattern = r"[^\w\s]"
    return re.sub(pattern, "", text)

def tweets_preprosesing(tweets_dir, tokenizer):
    df = pd.read_csv(tweets_dir, sep="\t", header= None, names=["tweets", "tag"])
    df.dropna(inplace = True)
    df["tweets"] = df["tweets"].apply(punctuation_removal)
    labels_mapping = {"OBJ": 0, "POS": 1, "NEG": 2, "NEUTRAL": 3}
    df["tag"] = df["tag"].apply(lambda tag: labels_mapping[tag])
    X = df.iloc[:, 0]
    y = df.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = CustomDataset(X_train, y_train, tokenizer, 512)
    test_dataset = CustomDataset(X_test, y_test,tokenizer, 512)
        
    
    return train_dataset, test_dataset
    
    
    
    
    
    
    
if __name__ == "__main__":
    print(tweets_preprosesing("data\Tweets.txt")[0][:10])
    