import numpy as np
import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, X_shape, Y_shape, hidden_layer_size=300, p_dropout=0.0):
        super().__init__()
        self.num_features = X_shape[-1]
        self.num_labels = Y_shape[-1]
        self.p_dropout = p_dropout
        self.hidden_layer = torch.nn.Linear(self.num_features, hidden_layer_size)
        self.output_layer = torch.nn.Linear(hidden_layer_size, self.num_labels)

    def forward(self, src):
        X = F.relu(self.hidden_layer(src))
        X = F.dropout(X, self.p_dropout)
        X = self.output_layer(X)
        return X

    @torch.no_grad()
    def predict(self, src):
        src = torch.tensor(src.values.astype(float), dtype=torch.float32)
        return self.forward(src).numpy()

    def fit(self, X, Y, learning_rate=1e-3, batch_size=16, epochs=1000):
        X = torch.tensor(X.values.astype(float), dtype=torch.float32)
        Y = torch.tensor(Y.values.astype(float), dtype=torch.float32)
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X, Y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_Y in dataloader:
                # Zero the gradients
                opt.zero_grad()

                # Forward pass
                outputs = self(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_Y)

                # Backward pass
                loss.backward()

                # Update weights
                opt.step()

                # Accumulate the batch loss
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            # print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
