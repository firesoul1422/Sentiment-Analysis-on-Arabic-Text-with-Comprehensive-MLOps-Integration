import unittest
from inference import inference

class TestInference(unittest.TestCase):
    def test_inference(self):
        tweet = "This is a positive tweet!"
        prediction = inference(tweet)
        
        self.assertIn(prediction, [0, 1, 2, 3])

if __name__ == "__main__":
    unittest.main()