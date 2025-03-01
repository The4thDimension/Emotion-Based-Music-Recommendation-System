# import torch
# import torch.nn as nn
# import tensorflow as tf
# import numpy as np

# # Load TensorFlow model
# keras_model = tf.keras.models.load_model("model.h5")

# # Extract model layers and parameters
# keras_weights = keras_model.get_weights()

# # Define an equivalent PyTorch model
# class EmotionModel(nn.Module):
#     def __init__(self):
#         super(EmotionModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(32 * 48 * 48, 7)  # Adjust based on actual architecture
    
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

# # Create PyTorch model
# torch_model = EmotionModel()

# # Convert and assign weights (manual mapping may be needed)
# with torch.no_grad():
#     torch_model.conv1.weight.copy_(torch.tensor(keras_weights[0]))
#     torch_model.conv1.bias.copy_(torch.tensor(keras_weights[1]))


# import h5py

# file_path = "model.h5"
# with h5py.File(file_path, "r") as f:
#     layer_names = list(f.keys())  # List all layers
#     print("Layers in saved model:", layer_names)



# import tensorflow as tf
# from keras import Sequential
# from keras import layers

# # Define the correct model architecture
# model = Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(62, 62, 1), name='conv2d_1'),
#     layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'),

#     layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
#     layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'),

#     layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
#     layers.MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3'),

#     layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_4'),

#     layers.Flatten(name='flatten_1'),

#     layers.Dense(1024, activation='relu', name='dense_1'),
#     layers.Dropout(0.5, name='dropout_1'),
    
#     layers.Dense(7, activation='relu', name='dense_2'),
#     layers.Dropout(0.5, name='dropout_2'),
    
#     layers.Dropout(0.5, name='dropout_3')
# ])

# # Load the weights
# model.load_weights("model.h5")

# # Save the complete model
# model.save("model_fixed.h5")

# print("Model fixed and saved successfully!")

# import tensorflow as tf
# import tf2onnx

# # Load the fixed Keras model
# keras_model = tf.keras.models.load_model("model_fixed.h5")
# keras_model.output_names=['output']

# # Define the input signature for the model
# input_signature = [tf.TensorSpec(shape=[None, 62, 62, 1], dtype=tf.float32, name="input")]

# # Convert to ONNX
# onnx_model_path = "model.onnx"
# onnx_model, _ = tf2onnx.convert.from_keras(
#     keras_model, opset=13, output_path=onnx_model_path, input_signature=input_signature
# )

# print(f"ONNX model saved at {onnx_model_path}")


import onnx
from onnx2pytorch import ConvertModel

# Load the ONNX model
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Convert to PyTorch
pytorch_model = ConvertModel(onnx_model)

print("ONNX model successfully converted to PyTorch!")




import torch

# Create a dummy input tensor (must match input shape)
dummy_input = torch.randn(1, 62, 62, 1)  # (batch_size, height, width, channels)

# Run inference
output = pytorch_model(dummy_input)

print("PyTorch Model Output:", output)


torch.save(pytorch_model.state_dict(), "model.pth")
print("PyTorch model saved successfully!")
