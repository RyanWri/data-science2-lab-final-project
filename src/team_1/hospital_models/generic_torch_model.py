import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class NeuralNetworkClassifier:
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the simple neural network model.

        Parameters:
        -----------
        input_size : int
            The number of input features.
        hidden_size : int
            The number of neurons in the hidden layer.
        num_classes : int
            The number of output classes (for classification).
        """
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ClassificationPipeline:
    def __init__(
        self, df, features, target, hidden_size=10, test_size=0.3, random_state=42
    ):
        """
        Initialize the classification pipeline.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the dataset.
        features : list
            List of column names that are the input features.
        target : str
            The column name of the target variable (must be categorical).
        hidden_size : int
            Number of neurons in the hidden layer of the neural network.
        test_size : float
            Proportion of the dataset to include in the test split.
        random_state : int
            Random seed for reproducibility of train-test split.
        """
        self.df = df
        self.features = features
        self.target = target
        self.hidden_size = hidden_size
        self.test_size = test_size
        self.random_state = random_state

    def encode_target(self):
        """
        Encode the target variable into numeric values.

        Returns:
        --------
        encoded_target : pd.Series
            Encoded target variable.
        """
        label_encoder = LabelEncoder()
        encoded_target = label_encoder.fit_transform(self.df[self.target])
        self.num_classes = len(np.unique(encoded_target))  # Get the number of classes
        return encoded_target

    def prepare_data(self):
        """
        Prepare the data by scaling features and encoding the target variable.

        Returns:
        --------
        X_train, X_test, y_train, y_test : torch.Tensors
            Training and testing data split and converted into torch tensors.
        """
        # Extract features and target
        X = self.df[self.features]
        y = self.encode_target()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state
        )

        # Convert data to torch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        return X_train, X_test, y_train, y_test

    def train_model(
        self,
        X_train,
        y_train,
        input_size,
        num_classes,
        num_epochs=100,
        learning_rate=0.01,
    ):
        """
        Train the neural network model.

        Parameters:
        -----------
        X_train : torch.Tensor
            The training data features.
        y_train : torch.Tensor
            The training data labels.
        input_size : int
            The number of input features.
        num_classes : int
            The number of output classes.
        num_epochs : int
            The number of epochs for training.
        learning_rate : float
            The learning rate for the optimizer.

        Returns:
        --------
        model : NeuralNetworkClassifier
            The trained model.
        """
        model = NeuralNetworkClassifier(
            input_size=input_size, hidden_size=self.hidden_size, num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            outputs = model.forward(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        return model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model using the test set.

        Parameters:
        -----------
        model : NeuralNetworkClassifier
            The trained model.
        X_test : torch.Tensor
            The test data features.
        y_test : torch.Tensor
            The test data labels.
        """
        model.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            outputs = model.forward(X_test)
            _, predicted = torch.max(outputs.data, 1)

        print(classification_report(y_test, predicted))


class ClassificationVisualization:
    """
    A class to handle visualization of classification results.
    This class does not need the DataFrame or features, just the actual
    and predicted values along with class names for plotting.
    """

    def plot_countplot(self, y_true, y_pred, class_names):
        """
        Plot a countplot comparing the actual and predicted class distributions.

        Parameters:
        -----------
        y_true : array-like
            Actual class labels.
        y_pred : array-like
            Predicted class labels.
        class_names : list
            List of class names for labeling the x-axis.
        """
        plt.figure(figsize=(10, 6))

        # Actual counts
        plt.subplot(1, 2, 1)
        sns.countplot(x=y_true, palette="viridis")
        plt.title("Actual Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(ticks=[0, 1, 2], labels=class_names)

        # Predicted counts
        plt.subplot(1, 2, 2)
        sns.countplot(x=y_pred, palette="viridis")
        plt.title("Predicted Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.xticks(ticks=[0, 1, 2], labels=class_names)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot a confusion matrix to evaluate the performance of the classification.

        Parameters:
        -----------
        y_true : array-like
            Actual class labels.
        y_pred : array-like
            Predicted class labels.
        class_names : list
            List of class names for labeling the matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap="viridis", xticks_rotation="vertical")
        plt.title("Confusion Matrix")
        plt.show()
