import os
from pillow_heif import register_heif_opener
from PIL import Image

# Enable HEIC support
register_heif_opener()

LABELS_DIR = r"C:\Users\roman\PycharmProjects\pythonProject3\labels"

# Loop through all subdirectories (brand folders)
for brand_folder in os.listdir(LABELS_DIR):
    brand_path = os.path.join(LABELS_DIR, brand_folder)

    if os.path.isdir(brand_path):  # Ensure it's a folder
        for file in os.listdir(brand_path):
            # Case-insensitive file extension check
            if file.lower().endswith(('.heic', '.HEIC')):
                heic_path = os.path.join(brand_path, file)
                # Replace either .heic or .HEIC with .jpg
                jpg_filename = file.lower().replace('.heic', '.jpg')
                jpg_path = os.path.join(brand_path, jpg_filename)

                try:
                    img = Image.open(heic_path)
                    img.save(jpg_path, 'JPEG')
                    os.remove(heic_path)  # Delete original HEIC file
                    print(f"Converted: {heic_path} → {jpg_path}")
                except Exception as e:
                    print(f"Error converting {heic_path}: {e}")

print("🎉 All HEIC images converted to JPG!")
