import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Define the neural network model
class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Define the PCA classification pipeline
class PCAClassificationPipeline:
    def __init__(
        self,
        df,
        features,
        target,
        n_components,
        hidden_size=10,
        test_size=0.3,
        random_state=42,
    ):
        self.df = df
        self.features = features
        self.target = target
        self.n_components = n_components
        self.hidden_size = hidden_size
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(self):
        """
        Prepares the data by applying PCA to the features and encoding the target variable.
        Returns train/test splits.
        """
        X = self.df[self.features]
        y = self.df[self.target]

        # Standardize features before applying PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Encode the target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y_encoded, test_size=self.test_size, random_state=self.random_state
        )

        # Convert to torch tensors
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
        """
        model = NeuralNetworkClassifier(
            input_size=input_size, hidden_size=self.hidden_size, num_classes=num_classes
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            outputs = model(X_train)
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
        """
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)

        print(classification_report(y_test, predicted))
