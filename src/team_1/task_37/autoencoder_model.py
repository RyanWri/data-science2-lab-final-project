import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



# Define the Autoencoder Model
class Autoencoder(nn.Module):
    def preprocessing_for_autoencoder(df):
        # Separate numeric and categorical columns
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        # Apply one-hot encoding to categorical columns
        df_cat_encoded = pd.get_dummies(df[cat_cols])

        # Concatenate numerical columns with the one-hot encoded categorical columns
        df_processed = pd.concat([df[num_cols], df_cat_encoded], axis=1)

        # Normalize the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df_processed.values)
        data_tensor = torch.FloatTensor(data)
        return data_tensor

    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def train_autoencoder(data_tensor, num_epochs = 1000, hidden_size = 2, lr=0.01 ):
        # Instantiate and train autoencoder
        input_size = data_tensor.shape[1]
        hidden_size = hidden_size  # Desired lower dimensionality
        model = Autoencoder(input_size, hidden_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        num_epochs = num_epochs

        # List to store loss over epochs
        loss_list = []
        # Train the model
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            encoded, decoded = model(data_tensor)
            loss = criterion(decoded, data_tensor)
            loss.backward()
            optimizer.step()
            # Store the loss for this epoch
            loss_list.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Plotting loss vs epochs
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), loss_list)
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        # Retrieve the encoded (compressed) data
        encoded_data, _ = model(data_tensor)
        encoded_numpy = encoded_data.detach().numpy()

        plt.figure(figsize=(8, 6))
        plt.scatter(encoded_numpy[:, 0], encoded_numpy[:, 1])
        plt.title("Encoded Data Space")
        plt.xlabel("Encoded Feature 1")
        plt.ylabel("Encoded Feature 2")
        plt.grid(True)
        plt.show()
        return model, encoded_numpy

    def cal_reconstruction_loss(model, data_tensor):
        # Forward pass to get the reconstructed output
        with torch.no_grad():
            encoded, decoded = model(data_tensor)

        # Calculate reconstruction error
        reconstruction_loss = nn.MSELoss()(decoded, data_tensor)
        print(f"Reconstruction Loss (MSE): {reconstruction_loss.item():.4f}")

