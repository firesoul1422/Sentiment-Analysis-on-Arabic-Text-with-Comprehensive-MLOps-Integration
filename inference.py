from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data_prepration import punctuation_removal
from torch.nn.functional import softmax
from datetime import datetime


def inference(tweet):
    tokenizer = AutoTokenizer.from_pretrained("model/")
    model = AutoModelForSequenceClassification.from_pretrained("model/")

    preprocesed_text = punctuation_removal(tweet)
    tokonized_text = tokenizer.encode_plus(preprocesed_text, return_tensors="pt")
    
    input_ids = tokonized_text["input_ids"]
    attention_mask = tokonized_text["attention_mask"]
    
    result = model(input_ids, attention_mask).logits[0]
    
    labels_mapping = {0: "OBJ", 1: "POS", 2: "NEG", 3: "NEUTRAL"}
    
    label  = labels_mapping[result.argmax(-1).item()]
    
    confidents = softmax(result, -1)
    with open("Log_data/prediction_history.txt", "a") as f:
        f.write(f"{tweet}\t{result}\t{confidents[confidents.argmax(-1)]}\t{label}\t{datetime.now()}")
    return label
