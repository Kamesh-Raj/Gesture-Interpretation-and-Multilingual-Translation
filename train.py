import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
X_train_features = np.load('X_train_features.npy')
y_train = np.load('y_train.npy')
X_test_features = np.load('X_test_features.npy')
y_test = np.load('y_test.npy')

# Normalize features
X_train_features = X_train_features / 255.0
X_test_features = X_test_features / 255.0

# Output dataset information
print(f"Total training sequences: {len(X_train_features)}")
print(f"Total testing sequences: {len(X_test_features)}")
print(f"Training features data shape: {X_train_features.shape}, Testing features data shape: {X_test_features.shape}")

# Define the CNN-LSTM model with bidirectional layers and regularization
def create_cnn_lstm_model(input_shape_features, num_classes):
    # Features input branch
    features_input = layers.Input(shape=input_shape_features)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(features_input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=regularizers.l2(0.01)))(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=features_input, outputs=output)
    return model

# Determine input shape and number of classes
sequence_length = X_train_features.shape[1]
num_features = X_train_features.shape[2]
num_classes = y_train.shape[1]
input_shape_features = (sequence_length, num_features)

# Create the model
cnn_lstm_model = create_cnn_lstm_model(input_shape_features, num_classes)

# Compile the model
cnn_lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
cnn_lstm_model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model and save history
cnn_lstm_history = cnn_lstm_model.fit(
    X_train_features, y_train,
    epochs=100,
    validation_data=(X_test_features, y_test),
    batch_size=2,
    callbacks=[early_stopping]
)

# Save the trained model and history
cnn_lstm_model.save('cnn_lstm_gesture_recognition_model1.h5')
np.save('cnn_lstm_history.npy', cnn_lstm_history.history)

# Evaluate the model on the test set
test_loss, test_accuracy = cnn_lstm_model.evaluate(X_test_features, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")