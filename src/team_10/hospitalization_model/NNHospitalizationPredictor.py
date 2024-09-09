import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix


class HospitalizationPredictionNN(nn.Module):
    def __init__(self, input_size, num_doctors, embedding_dim, hidden_size, output_size):
        super(HospitalizationPredictionNN, self).__init__()

        # Embedding layer for doctor ID
        self.embedding = nn.Embedding(num_doctors, embedding_dim)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, doctor_id):
        # Get doctor embedding
        doctor_embedding = self.embedding(doctor_id)

        # Concatenate doctor embedding with the input features
        x = torch.cat([x, doctor_embedding], dim=1)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class NNHospitalizationPredictor:
    def __init__(self, input_size, num_doctors, embedding_dim=10, hidden_size=64, output_size=3, learning_rate=0.001,
                 num_epochs=1000, encoding='onehot'):
        total_input_size = input_size + embedding_dim

        self.input_size = input_size
        self.num_doctors = num_doctors
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.encoding = encoding

        # Initialize the model with embedding for doctor IDs
        self.model = HospitalizationPredictionNN(total_input_size, num_doctors, embedding_dim, hidden_size, output_size)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize column transformers for scaling numeric and encoding categorical data
        self.numeric_transformer = StandardScaler()
        if encoding == 'onehot':
            self.categorical_transformer = OneHotEncoder(sparse_output=False)
        elif encoding == 'ordinal':
            self.categorical_transformer = OrdinalEncoder()

        # Initialize encoder for doctor ID
        self.doctor_encoder = LabelEncoder()

    # This method scales the numerical features and encodes the categorical features.
    def preprocess_data(self, x, y, doctor_column, numeric_columns, categorical_columns):
        # Apply scaling to numeric columns and encoding to categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_columns),
                ('cat', self.categorical_transformer, categorical_columns)
            ]
        )

        # Transform the input features (scale numerical and encode categorical)
        x_preprocessed = preprocessor.fit_transform(x)

        # Encode the doctor IDs
        doctor_encoded = self.doctor_encoder.fit_transform(doctor_column)

        y_array = y.to_numpy() if isinstance(y, pd.Series) else y

        # Encode the target variable based on selected encoding method
        if self.encoding == 'onehot':
            y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y_array.reshape(-1, 1))
        elif self.encoding == 'ordinal':
            y_encoded = OrdinalEncoder().fit_transform(y_array.reshape(-1, 1)).astype(int).flatten()

        return x_preprocessed, doctor_encoded, y_encoded

    # Split the data into train and test sets.
    def split_data(self, X, doctor, y, test_size=0.2):
        return train_test_split(X, doctor, y, test_size=test_size, random_state=42)

    # Train the model.
    def train(self, x_train, doctor_train, y_train):
        # Convert data to torch tensors
        x_train_tensor = torch.FloatTensor(x_train)
        doctor_train_tensor = torch.LongTensor(doctor_train)

        # Adjust label format based on encoding
        if self.encoding == 'onehot':
            y_train_tensor = torch.LongTensor(y_train.argmax(axis=1))
        elif self.encoding == 'ordinal':
            y_train_tensor = torch.LongTensor(y_train)

        for epoch in range(self.num_epochs):
            self.model.train()

            # Forward pass
            outputs = self.model(x_train_tensor, doctor_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model's accuracy on the test set.
    def evaluate(self, x_test, doctor_test, y_test):
        self.model.eval()

        # Convert data to torch tensors
        x_test_tensor = torch.FloatTensor(x_test)
        doctor_test_tensor = torch.LongTensor(doctor_test)

        # Adjust label format based on encoding
        if self.encoding == 'onehot':
            y_test_tensor = torch.LongTensor(y_test.argmax(axis=1))
        elif self.encoding == 'ordinal':
            y_test_tensor = torch.LongTensor(y_test)

        with torch.no_grad():
            # Get predictions
            test_outputs = self.model(x_test_tensor, doctor_test_tensor)
            _, predicted = torch.max(test_outputs, 1)

            # Calculate accuracy
            accuracy = accuracy_score(y_test_tensor, predicted)
            print(f'Accuracy: {accuracy * 100:.2f}%')

        # Return the predictions and ground truth labels
        return predicted, y_test_tensor
