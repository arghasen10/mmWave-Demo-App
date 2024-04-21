import numpy as np
import pickle

# Create a demo model with dimensions (256, 64)
demo_model = np.random.rand(256, 64)

# File path to save the .pkl file
file_path = "demo_model.pkl"

# Save the demo model to a .pkl file
with open(file_path, 'wb') as file:
    pickle.dump(demo_model, file)

print(f"Demo model saved to '{file_path}'")
