import zipfile
import os

zip_path = "neu-surface-defect-database.zip"
extract_path = "data/NEU"

# Create extract folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully to 'data/NEU/'")

