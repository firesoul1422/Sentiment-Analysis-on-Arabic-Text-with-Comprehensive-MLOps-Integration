import unittest
from training import training
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from pathlib import Path
from torch.utils.data import Dataset

class TestdTraining(unittest.TestCase):
    def test_training(self):
        data_dir = "data/Tweets.txt"
        my_file = Path(data_dir)

        if not my_file.is_file():
            self.skipTest(f"Test data file not found: {data_dir}")

        training(data_dir)
        
        model_expect_dest = "model/pytorch_model.bin"
        
        model_path = Path(model_expect_dest)
        
        self.assertTrue(model_path.is_file(), f"Model file not found: {model_expect_dest}")

    def test_training_loop(self):
        data_dir = "data/Tweets.txt"
        my_file = Path(data_dir)

        if not my_file.is_file():
            self.skipTest(f"Test data file not found: {data_dir}")

        try:
            training(data_dir)
        except Exception as e:
            self.fail(f"Training raised an exception: {e}")

    def test_model_initialization(self):
        model = AutoModelForSequenceClassification.from_pretrained("asafaya/bert-base-arabic")
        self.assertIsNotNone(model)

        
if __name__ == "__main__":
    unittest.main()