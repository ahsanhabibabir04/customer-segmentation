import unittest
import pandas as pd
import joblib

from pathlib import Path

class TestCustomerSegmentation(unittest.TestCase):

    def test_dataset_shape(self):
        df = pd.read_csv("data/segmented_customers.csv")
        self.assertEqual(len(df), 200)

    def test_no_missing_values(self):
        df = pd.read_csv("data/segmented_customers.csv")
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_segment_labels_exist(self):
        df = pd.read_csv("data/segmented_customers.csv")
        self.assertIn("Segment", df.columns)
        self.assertEqual(df["Segment"].nunique(), 5)

    def test_model_files_exist(self):
        self.assertTrue(Path("models/kmeans_model.pkl").exists())
        self.assertTrue(Path("models/scaler.pkl").exists())

    def test_model_predicts(self):
        model = joblib.load("models/kmeans_model.pkl")
        scaler = joblib.load("models/scaler.pkl")

        sample = pd.DataFrame({
            "Annual Income (k$)": [86],
            "Spending Score (1-100)": [82]
        })

        prediction = model.predict(scaler.transform(sample))
        self.assertEqual(len(prediction), 1)
        self.assertIn(int(prediction[0]), range(5))

    def test_visualization_exists(self):
        self.assertTrue(Path("models/customer_segments.png").exists())

if __name__ == "__main__":
    unittest.main()
