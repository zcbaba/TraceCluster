
# 1. Data loading function
def load_test_data(file_path):
    """Simplified data loading process, returning graph data."""
    # Core logic for loading feature and label maps
    feature_map, label_map = load_feature_and_label_maps('e3/models')
    ground_truth = load_ground_truth('groundtruth.txt')

    # Construct graph structure and node features
    data = construct_graph(file_path, feature_map, label_map, ground_truth)

    return data, feature_map, label_map


# 2. Model definition: Simplified Cluster-GAT model
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


# 3. Evaluation and inference
def evaluate_model(model, data, device):
    """Evaluate the model."""
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        prob = torch.softmax(out, dim=1).cpu()
        pred = torch.argmax(prob, dim=1).cpu()

    # Return only predictions
    return pred, prob


# 4. Rejecting and predicting (based on threshold)
def reject_and_predict(prob, threshold=0.6):
    """Handle rejected nodes."""
    p1 = prob.max(dim=1).values
    reject_mask = p1 < threshold
    rejected_nodes = reject_mask.nonzero(as_tuple=True)[0]

    return reject_mask, rejected_nodes


# 5. Save alarm (rejected nodes' neighborhood information)
def save_alarm(rejected_nodes, adj, adj2):
    """Save rejected nodes' neighborhood information to a file."""
    with open('alarm.txt', 'w', encoding='utf-8') as fw:
        for i in rejected_nodes:
            fw.write(f'{i}:')
            neighbor = set()
            # Inward: i <- j <- k
            if i in adj:
                for j in adj[i]:
                    neighbor.add(j)
                    if j in adj:
                        for k in adj[j]:
                            neighbor.add(k)
            # Outward: i -> j -> k
            if i in adj2:
                for j in adj2[i]:
                    neighbor.add(j)
                    if j in adj2:
                        for k in adj2[j]:
                            neighbor.add(k)
            for j in neighbor:
                fw.write(f' {j}\n')


# 6. Main function
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data, feature_map, label_map = load_test_data('e3/cadets_test.txt')
    model = ClusterGAT(len(feature_map) * 2, 32, len(label_map)).to(device)

    # Load the model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    pred, prob = evaluate_model(model, data, device)

    # Reject nodes and save alarm information
    reject_mask, rejected_nodes = reject_and_predict(prob)
    save_alarm(rejected_nodes, adj, adj2)

    # Print predictions
    print(f"Predictions: {pred}")


# Execute main function
if __name__ == '__main__':
    main()
