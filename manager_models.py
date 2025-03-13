import os
import json
import glob


class ModelManager:
    # Directory where models are stored
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    # File where accuracy logs are stored
    accuracy_log = os.path.join(model_dir, "accuracy_log.json")

    @classmethod
    def initialize(cls):
        """Initialize the model directory and accuracy log if they don't exist."""

        # Create models directory if it doesn't exist
        if not os.path.exists(cls.model_dir):
            os.makedirs(cls.model_dir)
            print(f"Created model directory at {cls.model_dir}")

        # Create accuracy log file if it doesn't exist
        if not os.path.exists(cls.accuracy_log):
            with open(cls.accuracy_log, "w") as f:
                json.dump({}, f)
            print(f"Created accuracy log at {cls.accuracy_log}")

    @classmethod
    def update_model(cls, model_name, accuracy):
        """Update the accuracy log with a new model."""

        # Initialize if needed
        cls.initialize()

        # Load existing accuracy data
        with open(cls.accuracy_log, "r") as f:
            accuracy_data = json.load(f)

        # Add new model accuracy
        accuracy_data[model_name] = float(accuracy)

        # Save updated accuracy data
        with open(cls.accuracy_log, "w") as f:
            json.dump(accuracy_data, f, indent=4)

        return model_name

    @classmethod
    def get_best_model(cls):
        """Get the path to the model with the highest accuracy."""

        # Initialize if needed
        cls.initialize()

        # Load accuracy data
        with open(cls.accuracy_log, "r") as f:
            accuracy_data = json.load(f)

        if not accuracy_data:
            return None

        # Find the model with the highest accuracy
        best_model = max(accuracy_data.items(), key=lambda x: x[1])[0]

        # Return the full path to the model
        model_path = os.path.join(cls.model_dir, best_model)

        # Verify the model file exists
        if os.path.exists(model_path):
            return model_path
        else:
            # Model file missing, clean up the accuracy log
            print(f"Best model file not found at {model_path}. Removing from log.")
            del accuracy_data[best_model]
            with open(cls.accuracy_log, "w") as f:
                json.dump(accuracy_data, f, indent=4)

            # Try to get the next best model
            return cls.get_best_model() if accuracy_data else None

    @classmethod
    def list_models(cls):
        """List all models and their accuracies."""

        # Initialize if needed
        cls.initialize()

        # Load accuracy data
        with open(cls.accuracy_log, "r") as f:
            accuracy_data = json.load(f)

        if not accuracy_data:
            print("No models found in the accuracy log.")
            return []

        # Sort models by accuracy (highest first)
        sorted_models = sorted(accuracy_data.items(), key=lambda x: x[1], reverse=True)

        print("\nAvailable Models (sorted by accuracy):")
        print("=" * 70)
        print(f"{'Model Name':<50} {'Accuracy':<10}")
        print("-" * 70)

        for model_name, accuracy in sorted_models:
            model_path = os.path.join(cls.model_dir, model_name)
            exists = "good path" if os.path.exists(model_path) else "bad path"
            print(f"{model_name:<50} {accuracy * 100:.2f}% {exists}")

        return [model[0] for model in sorted_models]

    @classmethod
    def get_model_metadata(cls, model_name):
        """Get metadata for a specific model if available."""

        # Extract timestamp from model name
        parts = model_name.split('_')
        if len(parts) > 1:
            timestamp = parts[-1].split('.')[0]
            metadata_path = os.path.join(cls.model_dir, f"metadata_{timestamp}.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)

        return None
