import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import ClusterData, ClusterLoader
from sklearn.utils.class_weight import compute_class_weight


# ========== Class Weight Function ==========
def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute class weights based on the frequency of classes."""
    weights = compute_class_weight('balanced', classes=list(range(num_classes)), y=labels.cpu().numpy())
    return torch.tensor(weights, dtype=torch.float)


# ========== Model Definition ==========
class ClusterGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClusterGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2, concat=True, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * 2, output_dim, heads=1, concat=False, dropout=0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ========== Simplified Main Function ==========
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Load data and maps
    print("[INFO] Loading data...")
    feature_map, label_map = load_maps(MAPS_DIR)
    train_data, feat_dim, label_num = build_graph(TRAIN_FILE, feature_map, label_map)
    val_data, _, _ = build_graph(VAL_FILE, feature_map, label_map)

    # Step 2: Compute class weights for the loss function
    print("[INFO] Computing class weights...")
    class_weights = compute_class_weights(train_data.y, label_num).to(device)

    # Step 3: Initialize model and optimizer
    print("[INFO] Initializing model...")
    model = ClusterGAT(feat_dim, 32, label_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Step 4: Training loop (simplified)
    print("[INFO] Starting training loop...")
    max_epochs = 2000
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x.to(device), train_data.edge_index.to(device))
        loss = F.nll_loss(out, train_data.y.to(device), weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item():.6f}")

        # Early stopping condition (based on loss or accuracy can be added here)
        if epoch > 100 and loss.item() < 1e-4:
            print("[INFO] Early stopping triggered.")
            break

    # Step 5: Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        val_out = model(val_data.x.to(device), val_data.edge_index.to(device))
        val_loss = F.nll_loss(val_out, val_data.y.to(device))
        print(f"[INFO] Validation Loss: {val_loss.item():.6f}")

    # Step 6: Save the best model (after training is complete)
    torch.save(model.state_dict(), BEST_CKPT)
    print("[INFO] Model training complete, saved to:", BEST_CKPT)


if __name__ == "__main__":
    main()
