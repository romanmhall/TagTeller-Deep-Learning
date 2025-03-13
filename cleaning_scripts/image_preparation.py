import os
import cv2
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

# Full path to labels directory based on your screenshot
LABELS_DIR = r"C:\Users\roman\PycharmProjects\pythonProject3\labels"


def enhance_image_for_classification(image_path, show_debug=False):
    """
    Enhance image for better classification by:
    1. Auto-rotating based on EXIF
    2. Normalizing brightness/contrast
    3. Cropping to focus on the shirt label

    Args:
        image_path: Path to the image file
        show_debug: Whether to show debug visualization
    """
    try:
        # Step 1: Open and auto-rotate image based on EXIF data
        img_pil = Image.open(image_path)

        # Auto-rotate based on EXIF
        if hasattr(img_pil, '_getexif') and img_pil._getexif():
            exif = dict((ExifTags.TAGS.get(k, k), v) for k, v in img_pil._getexif().items())

            if 'Orientation' in exif:
                orientation = exif['Orientation']

                if orientation == 2:
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    img_pil = img_pil.rotate(180)
                elif orientation == 4:
                    img_pil = img_pil.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    img_pil = img_pil.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    img_pil = img_pil.rotate(-90, expand=True)
                elif orientation == 7:
                    img_pil = img_pil.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    img_pil = img_pil.rotate(90, expand=True)

        # Convert to OpenCV format for advanced processing
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Step 2: Normalize brightness and contrast
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Step 3: Sharpen to make text more readable
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # Save the enhanced image
        cv2.imwrite(image_path, sharpened)

        if show_debug:
            # Show before/after for debugging
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
            plt.title('Enhanced')
            plt.show()

        print(f"Enhanced: {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Show a debug menu at startup
print("Image Enhancement for Brand Classification")
print("=" * 50)
print("This script will enhance images to improve brand classification accuracy.")
print("Enhancement includes: auto-rotation, contrast adjustment, and sharpening.")
print("=" * 50)

# Process a specific brand or all brands
while True:
    print("\nOptions:")
    print("1. Process a specific brand folder")
    print("2. Process all brand folders")
    print("3. Exit")

    choice = input("Select an option (1-3): ").strip()

    if choice == "3":
        print("Exiting...")
        break

    elif choice == "1":
        # List available brands
        brand_folders = [folder for folder in os.listdir(LABELS_DIR)
                         if os.path.isdir(os.path.join(LABELS_DIR, folder))]

        print("\nAvailable brand folders:")
        for i, folder in enumerate(brand_folders):
            print(f"{i + 1}. {folder}")

        try:
            brand_idx = int(input("\nEnter the number of the brand to process: ").strip()) - 1
            if 0 <= brand_idx < len(brand_folders):
                brand_to_process = [brand_folders[brand_idx]]
            else:
                print("Invalid selection. Please try again.")
                continue
        except ValueError:
            print("Please enter a valid number.")
            continue

    elif choice == "2":
        # Process all brands
        brand_to_process = [folder for folder in os.listdir(LABELS_DIR)
                            if os.path.isdir(os.path.join(LABELS_DIR, folder))]

    else:
        print("Invalid option. Please try again.")
        continue

    # Process selected brand(s)
    total_images = 0
    for brand in brand_to_process:
        brand_path = os.path.join(LABELS_DIR, brand)
        image_files = [f for f in os.listdir(brand_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"No images found in {brand} folder.")
            continue

        print(f"\nProcessing {brand} folder ({len(image_files)} images)")

        for file in image_files:
            image_path = os.path.join(brand_path, file)
            enhance_image_for_classification(image_path)
            total_images += 1

    print(f"\nProcessed {total_images} images across {len(brand_to_process)} brands!")

print("\nDone enhancing images. This should help improve classification accuracy.")
