import os
import json
import time
import signal
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from manager_models import ModelManager
from sklearn.utils.class_weight import compute_class_weight
import statistics


# Signal handler for graceful exit with Ctrl+C
def signal_handler(sig, frame):
    print('\n Training interrupted by user. Exiting gracefully...')
    sys.exit(0)


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Set accuracy threshold for stopping training
ACCURACY_THRESHOLD = 0.85  # 85% validation accuracy

# Set plateau detection parameters
PLATEAU_WINDOW = 10  # Number of iterations to check for plateau
PLATEAU_THRESHOLD = 0.01  # Maximum variance allowed to consider it a plateau

# Check for manual override flag
if len(sys.argv) > 1 and sys.argv[1].lower() == "--force":
    FORCE_TRAINING = True
    remaining_cycles = float('inf')  # Run indefinitely unless manually stopped
    print("FORCE mode enabled: Will run training regardless of current accuracy")
else:
    FORCE_TRAINING = False
    remaining_cycles = float('inf')  # Run until accuracy threshold by default

# Main loop
print(
    f"Starting continuous training. Will stop when validation accuracy reaches {ACCURACY_THRESHOLD * 100}%. Press Ctrl+C to stop.")

while True:
    try:
        # Set Up Dataset Directory
        LABEL_DIR = r"C:\Users\roman\PycharmProjects\pythonProject3\labels"

        # Select Brands to Fine-Tune - Use actual folder names from directory
        brand_folders = sorted([folder for folder in os.listdir(LABEL_DIR)
                                if os.path.isdir(os.path.join(LABEL_DIR, folder))])

        # Verify we have at least 2 brands to classify
        if len(brand_folders) < 2:
            print(f"ERROR: Found only {len(brand_folders)} brand folders. Need at least 2 for classification.")
            time.sleep(60)
            continue

        # Display the brands we'll train on
        print(f"Training on these brands: {brand_folders}")

        # Load the Best Available Model
        model_path = ModelManager.get_best_model()
        if not model_path:
            print("ERROR: No trained model found. Please train the model first.")
            time.sleep(60)  # Wait before retrying
            continue

        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model successfully loaded.")

        # Check if we already have a model that meets our accuracy threshold
        with open(ModelManager.accuracy_log, "r") as f:
            accuracy_data = json.load(f)

        # Check for plateau in model performance
        plateau_detected = False
        if len(accuracy_data) >= PLATEAU_WINDOW:
            # Get the most recent models (sort by timestamp in filename)
            recent_models = sorted(accuracy_data.items(), key=lambda x: x[0], reverse=True)[:PLATEAU_WINDOW]
            recent_accuracies = [acc for _, acc in recent_models]

            # Calculate variance and mean of recent accuracies
            acc_variance = statistics.variance(recent_accuracies) if len(recent_accuracies) > 1 else 1.0
            acc_mean = statistics.mean(recent_accuracies)

            # Check if variance is below threshold (indicating plateau)
            if acc_variance < PLATEAU_THRESHOLD:
                plateau_detected = True
                print(f"PLATEAU DETECTED: Last {PLATEAU_WINDOW} models show minimal variance ({acc_variance:.6f})")
                print(f"Recent accuracies: {[f'{acc:.4f}' for acc in recent_accuracies]}")
                print(f"Mean accuracy: {acc_mean:.4f}")

                # Ask user if they want to continue despite plateau - UPDATED SECTION
                user_choice = input("Plateau detected. Choose an option:\n"
                                    "1: Stop training now\n"
                                    "2: Run one more training cycle\n"
                                    "3: Run until accuracy threshold is met\n"
                                    "4: Run a custom number of cycles\n"
                                    "Enter your choice (1-4): ").strip()

                if user_choice == "1":
                    print("Training stopped due to plateau.")
                    sys.exit(0)  # Exit the script
                elif user_choice == "2":
                    print("Running one more training cycle despite plateau")
                    FORCE_TRAINING = True  # This will ensure we exit after one more cycle
                    remaining_cycles = 1  # Track one remaining cycle
                elif user_choice == "3":
                    print(f"Will continue training until {ACCURACY_THRESHOLD * 100}% accuracy is reached")
                    FORCE_TRAINING = False  # Disable force training to allow multiple cycles
                    remaining_cycles = float('inf')  # Infinite cycles until threshold is met
                elif user_choice == "4":
                    try:
                        num_cycles = int(input("Enter number of training cycles to run: ").strip())
                        if num_cycles < 1:
                            print("Invalid number. Running one cycle.")
                            num_cycles = 1
                        print(f"Will run {num_cycles} more training cycles")
                        remaining_cycles = num_cycles
                        FORCE_TRAINING = True  # Set to true so we can track cycles
                    except ValueError:
                        print("Invalid input. Running one cycle.")
                        remaining_cycles = 1
                        FORCE_TRAINING = True
                else:
                    print("Invalid choice. Stopping training.")
                    sys.exit(0)

        best_accuracy = max(accuracy_data.values()) if accuracy_data else 0
        if best_accuracy >= ACCURACY_THRESHOLD and not FORCE_TRAINING:
            print(
                f"Target accuracy of {ACCURACY_THRESHOLD * 100}% already achieved! Best model accuracy: {best_accuracy * 100:.2f}%")

            # Ask user if they want to train anyway - UPDATED SECTION
            user_choice = input("Accuracy threshold already met. Choose an option:\n"
                                "1: Stop training now\n"
                                "2: Run one more training cycle\n"
                                "3: Run a custom number of cycles\n"
                                "Enter your choice (1-3): ").strip()

            if user_choice == "1":
                print("Training stopped by user.")
                sys.exit(0)  # Exit the script
            elif user_choice == "2":
                print("Running one more training cycle")
                FORCE_TRAINING = True
                remaining_cycles = 1
            elif user_choice == "3":
                try:
                    num_cycles = int(input("Enter number of training cycles to run: ").strip())
                    if num_cycles < 1:
                        print("Invalid number. Running one cycle.")
                        num_cycles = 1
                    print(f"Will run {num_cycles} more training cycles")
                    remaining_cycles = num_cycles
                    FORCE_TRAINING = True  # Set to true so we can track cycles
                except ValueError:
                    print("Invalid input. Running one cycle.")
                    remaining_cycles = 1
                    FORCE_TRAINING = True
            else:
                print("Invalid choice. Stopping training.")
                sys.exit(0)

        print(f"Current best accuracy: {best_accuracy * 100:.2f}%, target: {ACCURACY_THRESHOLD * 100}%")

        # Unfreeze Some Layers for Fine-Tuning
        for layer in model.layers[-30:]:  # Unfreeze last 30 layers
            layer.trainable = True

        # Get existing layer names to avoid duplicates
        existing_layer_names = [layer.name for layer in model.layers]


        # Function to generate unique name
        def get_unique_name(base_name):
            counter = 1
            unique_name = f"{base_name}_{counter}"
            while unique_name in existing_layer_names:
                counter += 1
                unique_name = f"{base_name}_{counter}"
            existing_layer_names.append(unique_name)  # Add to list to avoid future conflicts
            return unique_name


        # Modify Model to Update Output Layer
        x = model.layers[-2].output  # Get the last hidden layer
        dropout_layer_name = get_unique_name("dropout_custom")
        dense_layer_name = get_unique_name("dense_output_custom")

        x = Dropout(0.3, name=dropout_layer_name)(x)  # Use unique name
        new_predictions = Dense(len(brand_folders), activation='softmax', name=dense_layer_name)(x)  # Use unique name

        print(f"Created new layers with unique names: {dropout_layer_name}, {dense_layer_name}")

        # Create a New Model with Updated Output
        model = Model(inputs=model.input, outputs=new_predictions)

        # Use Learning Rate Decay
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=100, decay_rate=0.96, staircase=True
        )

        # Recompile Model with a Better Optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Data Augmentation for More Training Variability
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            rotation_range=50,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.3,
            zoom_range=0.4,
            horizontal_flip=True,
            brightness_range=[0.4, 1.6],
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

        # Create Data Generators
        train_data = train_datagen.flow_from_directory(
            LABEL_DIR,
            classes=brand_folders,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            subset='training'
        )

        val_data = val_datagen.flow_from_directory(
            LABEL_DIR,
            classes=brand_folders,
            target_size=(224, 224),
            batch_size=16,
            class_mode='categorical',
            subset='validation'
        )

        # Compute Class Weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_data.classes),
            y=train_data.classes
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"Class Weights: {class_weight_dict}")

        # Save the class mapping for inference later
        class_indices = train_data.class_indices
        class_mapping = {v: k for k, v in class_indices.items()}  # Convert index->class to class->index

        model_metadata = {
            "class_mapping": class_mapping,
            "trained_on_folders": brand_folders,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Train the Model
        print("\nStarting fine-tuning...")
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=10,
            verbose=1,
            class_weight=class_weight_dict
        )

        # Get Best Validation Accuracy
        new_accuracy = max(history.history['val_accuracy'])
        print(f"New Model Accuracy: {new_accuracy:.4f}")

        # Save Fine-Tuned Model in `models/` with Timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")  # Unique identifier
        fine_tuned_model_name = f"brand_classifier_finetuned_{timestamp}.h5"
        fine_tuned_model_path = os.path.join(ModelManager.model_dir, fine_tuned_model_name)
        model.save(fine_tuned_model_path)
        print(f"Fine-tuned model saved as {fine_tuned_model_path}")

        # Save metadata alongside the model
        metadata_path = os.path.join(ModelManager.model_dir, f"metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        print(f"Model metadata saved to {metadata_path}")

        # Update ModelManager with the Best Model
        ModelManager.update_model(fine_tuned_model_name, new_accuracy)
        print("Accuracy log updated.")

        # Compare with previous best model
        with open(ModelManager.accuracy_log, "r") as f:
            accuracy_data = json.load(f)

        # Check if we've reached the accuracy threshold - UPDATED SECTION
        if new_accuracy >= ACCURACY_THRESHOLD and not FORCE_TRAINING:
            print(
                f"Target accuracy of {ACCURACY_THRESHOLD * 100}% achieved! Model accuracy: {new_accuracy * 100:.2f}%")
            print("Training completed successfully.")

            # Optional: Print best model info
            all_models = ModelManager.list_models()
            print("\nBest model details:")
            best_model_name = max(accuracy_data, key=accuracy_data.get)
            print(f"   Model name: {best_model_name}")
            print(f"   Accuracy: {accuracy_data[best_model_name] * 100:.2f}%")
            print(f"   Path: {os.path.join(ModelManager.model_dir, best_model_name)}")

            # Get corresponding metadata file
            model_timestamp = best_model_name.split('_')[-1].split('.')[0]
            metadata_file = os.path.join(ModelManager.model_dir, f"metadata_{model_timestamp}.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    best_metadata = json.load(f)
                print(f"   Trained on brands: {best_metadata['trained_on_folders']}")

            # Ask user if they want to run another training cycle
            user_choice = input("Would you like to run another training cycle? (y/n): ").strip().lower()
            if user_choice != 'y':
                sys.exit(0)  # Exit the script
            else:
                # Ask for number of cycles
                try:
                    num_cycles = int(
                        input("Enter number of additional training cycles to run (default: 1): ").strip() or "1")
                    if num_cycles < 1:
                        print("Invalid number. Running one cycle.")
                        num_cycles = 1
                    print(f"Will run {num_cycles} more training cycles")
                    remaining_cycles = num_cycles
                    FORCE_TRAINING = True  # Set to true so we can track cycles
                except ValueError:
                    print("Invalid input. Running one cycle.")
                    remaining_cycles = 1
                    FORCE_TRAINING = True
                continue

        # Filter out the current model to find the previous best
        previous_models = {k: v for k, v in accuracy_data.items() if k != fine_tuned_model_name}
        if previous_models:
            previous_best = max(previous_models.values())
            improvement = new_accuracy - previous_best

            if improvement > 0:
                print(f"IMPROVEMENT: New model is better by {improvement:.4f} accuracy points!")
            else:
                print(f"REGRESSION: New model is worse by {abs(improvement):.4f} accuracy points.")

        # List All Models
        ModelManager.list_models()

        # UPDATED EXIT CONDITION
        # Check if we need to exit based on remaining cycles
        if FORCE_TRAINING:
            # Decrement remaining cycles counter
            remaining_cycles -= 1
            if remaining_cycles <= 0:
                print(f"Requested training cycles complete. Exiting.")
                sys.exit(0)
            else:
                print(f"{remaining_cycles} training cycles remaining...")
                # Continue to next cycle without delay
                continue

        # Add delay between iterations if needed
        # print(f"\nWaiting 1 minute before next training cycle...")
        # for i in range(60, 0, -1):
        #    if i % 15 == 0:  # Show remaining time every 15 seconds
        #        print(f"{i} seconds remaining until next run...")
        #    time.sleep(1)

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Waiting 1 minute before retrying...")
        time.sleep(60)  # Wait a minute before retrying if an error occurs
