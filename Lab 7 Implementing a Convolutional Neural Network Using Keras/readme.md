# Lab 7: Handwritten Digit Recognition with Convolutional Neural Networks

## Overview
This lab implements a convolutional neural network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The project demonstrates an end-to-end deep learning workflow for computer vision tasks, from data preprocessing to model evaluation and prediction.

## Key Components

### Data Pipeline
- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Preprocessing**:
  - Pixel normalization (0-255 → 0-1)
  - Reshaping to (28, 28, 1) for CNN input
  - Visual inspection of sample digits

## Implementation Highlights

### Data Preparation
```python
# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
```

### Model Training
```python
model.compile(
    optimizer=SGD(learning_rate=0.1),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, epochs=10)
```

## Performance Results
| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 91.8% | 91.4% |
| Loss | 0.28 | 0.29 |
| Inference Time | - | 3ms/image |

## How to Use

1. Load the pretrained model:
```python
from tensorflow.keras.models import load_model
model = load_model('mnist_cnn.h5')
```

2. Make predictions:
```python
import numpy as np
from PIL import Image

# Preprocess custom image
img = Image.open('digit.png').convert('L')
img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

# Predict
prediction = model.predict(img_array)
digit = np.argmax(prediction)
```

## Dependencies
- Python 3.8+
- TensorFlow 2.8+
- matplotlib 3.5+
- numpy 1.21+
- pillow 9.0+ (for custom image prediction)

## Files Included
1. `mnist_cnn_training.ipynb`: Complete training notebook
2. `mnist_cnn.h5`: Saved model weights
3. `model_architecture.png`: Visual representation of CNN
4. `performance_curves.png`: Training metrics over epochs

## Key Findings
- Achieved 91.4% accuracy with relatively shallow architecture
- Global Average Pooling reduced parameters vs Flatten+Dense
- Batch normalization enabled faster convergence
- Model struggles most with ambiguous digit shapes

*Model predictions on test set samples with confidence scores*

## Best Practices
- Visualized layer activations to debug learning
- Used TensorBoard for training monitoring
- Implemented early stopping to prevent overfitting
- Analyzed confusion matrix for error patterns
