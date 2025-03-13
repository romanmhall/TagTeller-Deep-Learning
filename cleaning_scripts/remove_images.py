import os
import glob

# Main directory path
main_directory = r'C:\Users\roman\PycharmProjects\pythonProject3\models_success'

# Common image file extensions
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

# Counter for tracking
deleted_count = 0

# Walk through all subdirectories
for root, dirs, files in os.walk(main_directory):
    # Process each image extension in each directory
    for ext in image_extensions:
        # Create pattern for glob
        pattern = os.path.join(root, ext)

        # Find all matching files
        image_files = glob.glob(pattern)

        # Delete each file
        for file in image_files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file}: {e}")

print(f"Total: {deleted_count} image files deleted from {main_directory} and its subdirectories")