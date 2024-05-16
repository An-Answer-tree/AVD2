import os
import random
import shutil
import pandas as pd
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define source and target directories
source_dir = 'total_mp4'
train_dir = 'train3'
test_dir = 'test3'
val_dir = 'val3'

# Create target directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Read the annotation file
excel_file = 'text_information_annotation.xls'
df = pd.read_excel(excel_file, sheet_name='annotation file')

# Set video filenames as the index and ensure video names are strings
df['video'] = df['video'].astype(str).str.zfill(6)
df.set_index('video', inplace=True)

# Get all filenames
all_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]

# Check existing files in 'total_mp4' directory
existing_files = [f for f in all_files if f in os.listdir(source_dir)]

# Convert list to a numpy array and shuffle
existing_files_arr = np.array(existing_files)
np.random.shuffle(existing_files_arr)
existing_files = existing_files_arr.tolist()

# Calculate the size of each set
total_files = len(existing_files)
train_size = int(total_files * 0.8)
test_size = int(total_files * 0.1)
val_size = total_files - train_size - test_size

# Split files
train_files = existing_files[:train_size]
test_files = existing_files[train_size:train_size + test_size]
val_files = existing_files[train_size + test_size:]

# Function to move files
def move_files(files, destination):
    for f in files:
        shutil.move(os.path.join(source_dir, f), os.path.join(destination, f))

# Move files to the respective directories
move_files(train_files, train_dir)
move_files(test_files, test_dir)
move_files(val_files, val_dir)

# Get filenames in each directory without extension and pad with zeros
train_files_no_ext = [os.path.splitext(f)[0].zfill(6) for f in train_files]
test_files_no_ext = [os.path.splitext(f)[0].zfill(6) for f in test_files]
val_files_no_ext = [os.path.splitext(f)[0].zfill(6) for f in val_files]

# Print the first few filenames
print("Train files:", train_files_no_ext[:5])
print("Test files:", test_files_no_ext[:5])
print("Validation files:", val_files_no_ext[:5])

# Use reindex to ensure correct matching and filter out non-existent indices
train_labels = df.reindex(train_files_no_ext).dropna(how='all')
test_labels = df.reindex(test_files_no_ext).dropna(how='all')
val_labels = df.reindex(val_files_no_ext).dropna(how='all')

# Print the first few rows of the generated DataFrames to confirm
print("Train labels DataFrame:")
print(train_labels.head())
print("Test labels DataFrame:")
print(test_labels.head())
print("Validation labels DataFrame:")
print(val_labels.head())

# Save the new Excel files
train_labels.to_excel('train_labels.xlsx', index=True)
test_labels.to_excel('test_labels.xlsx', index=True)
val_labels.to_excel('val_labels.xlsx', index=True)

print(f"Train set size: {len(train_files_no_ext)}")
print(f"Test set size: {len(test_files_no_ext)}")
print(f"Validation set size: {len(val_files_no_ext)}")
