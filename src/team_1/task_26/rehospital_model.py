import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RehospitalizationNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        A simple neural network model for classification.

        Parameters:
        -----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of neurons in the hidden layer.
        num_classes : int
            The number of output classes (for classification).
        """
        super(RehospitalizationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # Hidden layer to output

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Step 1: Prepare the data
def prepare_data(df):
    """
    Prepare the data by encoding categorical variables and normalizing features.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.

    Returns:
    --------
    X_train, X_test, y_train, y_test : Tensors
        The training and test sets for features (X) and labels (y).
    """
    # Encode gender as numeric: M -> 1, F -> 0
    df["gender_encoded"] = df["gender"].map({"M": 1, "F": 0})

    # Select features and target variable
    X = df[["age", "gender_encoded", "hospitalization_count"]]
    y = df["duration_classification"].map(
        {"short": 0, "medium": 1, "long": 2}
    )  # Encoding target variable

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(
        y_train.values, dtype=torch.long
    )  # Use long for classification
    y_test = torch.tensor(y_test.values, dtype=torch.long)

    return X_train, X_test, y_train, y_test


# Step 2: Define the model, loss function, and optimizer
def train_nn_model(
    X_train,
    y_train,
    input_size,
    hidden_size,
    num_classes,
    num_epochs=100,
    learning_rate=0.01,
):
    """
    Train the neural network model.

    Parameters:
    -----------
    X_train : Tensor
        The training feature data.
    y_train : Tensor
        The training labels.
    input_size : int
        The number of input features.
    hidden_size : int
        The number of neurons in the hidden layer.
    num_classes : int
        The number of output classes.
    num_epochs : int, optional
        The number of training epochs (default is 100).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 0.01).

    Returns:
    --------
    model : nn.Module
        The trained neural network model.
    """
    model = RehospitalizationNN(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model


# Step 3: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using test data and print the classification report.

    Parameters:
    -----------
    model : nn.Module
        The trained neural network model.
    X_test : Tensor
        The test feature data.
    y_test : Tensor
        The test labels.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

    print(classification_report(y_test, predicted))


if __name__ == "__main__":
    # Example Usage:
    # Assume `your_dataframe` already contains 'duration_classification', 'age', 'gender', etc.
    # Step 1: Prepare the data
    df = pd.read_csv("rehospital.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 2: Define the model and train it
    input_size = X_train.shape[1]
    hidden_size = 10  # You can adjust this based on your needs
    num_classes = 3  # We have 3 classes: short, medium, long
    model = train_nn_model(
        X_train,
        y_train,
        input_size,
        hidden_size,
        num_classes,
        num_epochs=100,
        learning_rate=0.01,
    )

    # Step 3: Evaluate the model
    evaluate_model(model, X_test, y_test)
