import joblib

# Load model (set your_model_path to your .pkl file)
model_path = "your_model_path"
svr_model = joblib.load(model_path)

# Get the number of support vectors
num_support_vectors = len(svr_model.support_vectors_)

# Print results
print(f"Number of support vectors: {num_support_vectors}")

# num_support_vectors = 2197  # [Unused] example value
feature_dimensions = 160  # Replace with the actual number of feature dimensions
total_parameters = num_support_vectors * feature_dimensions + 1
parameters_in_millions = total_parameters / 1e6

print(f"Number of parameters (in millions): {parameters_in_millions}M")