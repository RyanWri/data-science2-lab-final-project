import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RehospitalizationMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_prob=0.5):
        """
        A more advanced neural network model with dropout and batch normalization.

        Parameters:
        -----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of neurons in the hidden layers.
        num_classes : int
            The number of output classes (for classification).
        dropout_prob : float, optional
            Probability of an element to be zeroed (for dropout). Default is 0.5.
        """
        super(RehospitalizationMLP, self).__init__()
        
        # First hidden layer with batch normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        return out


def encode_target(df,target):
    """
    Encode the target variable into numeric values.

    Returns:
    --------
    encoded_target : pd.Series
        Encoded target variable.
    """
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(df[target])
    return encoded_target

# Step 1: Prepare the data
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_data(df, categorical_columns, numerical_columns, target_column):
    """
    Prepare the data by encoding categorical variables, normalizing numerical features,
    and splitting the dataset into training and test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data.
    categorical_columns : list
        List of column names that are categorical.
    numerical_columns : list
        List of column names that are numerical.
    target_column : str
        The name of the target column for classification.

    Returns:
    --------
    X_train, X_test, y_train, y_test : Tensors
        The training and test sets for features (X) and labels (y).
    """
    df_processed = df.copy()

    df_processed

    # Drop the target column from numerical and categorical columns if present
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)

    features = numerical_columns + categorical_columns

    X = df[features]
    y = encode_target(df_processed,target_column)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size = 0.3, random_state = 42
    )

    # Convert data to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

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
    dropout_prob=0.5,
    print_every=10,
    device=None
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
        The number of neurons in the hidden layers.
    num_classes : int
        The number of output classes.
    num_epochs : int, optional
        The number of training epochs (default is 100).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 0.01).
    dropout_prob : float, optional
        The probability of dropout (default is 0.5).
    print_every : int, optional
        How often to print training progress (default is every 10 epochs).
    device : str, optional
        Specify whether to run the model on 'cpu' or 'cuda' (default is None).

    Returns:
    --------
    model : nn.Module
        The trained neural network model.
    losses : list
        List of training losses for each epoch.
    """
    # Use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")

    # Initialize model, criterion, and optimizer
    model = RehospitalizationMLP(input_size, hidden_size, num_classes, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move data to the device
    X_train, y_train = X_train.to(device), y_train.to(device)

    losses = []  # To track the loss over epochs

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save the loss for tracking
        losses.append(loss.item())

        # Print progress every few epochs
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model, losses


def evaluate_model(model, X_test, y_test, device=None):
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
    device : str, optional
        Specify whether to run the model on 'cpu' or 'cuda' (default is None).
    """
    model.eval()  # Set the model to evaluation mode
    
    # Move data to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score

    # Move predictions and labels back to CPU for evaluation
    predicted = predicted.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # Print the classification report
    print("Classification Report:")
    print(classification_report(y_test, predicted))

    # Optionally, print accuracy
    accuracy = accuracy_score(y_test, predicted)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    
    # Step 1: Prepare the data
    df = pd.read_csv("rehospital.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 2: Define the model and train it
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 10  
    num_classes = 3  # 3 classes: short, medium, long
    
    # Train the model and retrieve losses
    model, losses = train_nn_model(
        X_train,
        y_train,
        input_size,
        hidden_size,
        num_classes,
        num_epochs=100,
        learning_rate=0.01
    )

    # Step 3: Plot the training loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 4: Evaluate the model
    evaluate_model(model, X_test, y_test)
