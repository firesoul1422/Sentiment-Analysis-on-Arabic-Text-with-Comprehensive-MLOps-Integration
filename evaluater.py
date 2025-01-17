import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def evaluation(eval):
    pred, true = eval
    pred = torch.argmax(pred, axis=-1)
    
    accurecy = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    recall = recall_score(true, pred)
    precision = precision_score(true, pred)
    
    
    return {"accuercy": accurecy,
            "f1_score": f1,
            "recall_score": recall,
            "precision_score": precision}
