import kagglehub
import os

print("Downloading dataset...")
path = kagglehub.dataset_download("ismetsemedov/polymarket-prediction-markets")
print("Path to dataset files:", path)

print("Files in dataset:")
for root, dirs, files in os.walk(path):
    for f in files:
        print(os.path.join(root, f))
