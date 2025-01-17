import unittest
from data_prepration import punctuation_removal, tweets_preprosesing
from transformers import AutoTokenizer
from pathlib import Path
from torch.utils.data import Dataset

class TestdDataPrepration(unittest.TestCase):
    def test_punctuation_removal(self):
        
        # punctuation
        text = ".!هذا نص يوجد فيه, علامات تنصيص"
        result = punctuation_removal(text)
        self.assertEqual(text, "هذا نص يوجد فيه علامات تنصيص")
        
        # no punctuation
        text = "هذا نص يوجد فيه علامات تنصيص"
        result = punctuation_removal(text)
        self.assertEqual(text, "هذا نص يوجد فيه علامات تنصيص")
        
        
        # empty
        text = ""
        result = punctuation_removal(text)
        self.assertEqual(text, "")
    
    
    def test_tweets_preprosesing(self):
        data_dir = "data/Tweets.txt"
        my_file = Path(data_dir)

        # Skip the test if the file does not exist
        if not my_file.is_file():
            self.skipTest(f"Test data file not found: {data_dir}")

        # Test if the function returns a Dataset object
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        train_dataset, test_dataset = tweets_preprosesing(data_dir, tokenizer)
        self.assertIsInstance(train_dataset, Dataset)
        self.assertIsInstance(test_dataset, Dataset)

    def test_tweets_preprosesing_file_not_found(self):
        # Test if the function raises FileNotFoundError for a non-existent file
        data_dir = "data/non_existent_file.txt"
        tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        with self.assertRaises(FileNotFoundError):
            tweets_preprosesing(data_dir, tokenizer)
            
        
if __name__ == "__main__":
    unittest.main()