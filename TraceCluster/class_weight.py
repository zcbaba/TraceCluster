def calculate_class_weights(data):
    class_counts = torch.bincount(data.y)
    class_weights = 1.0 / class_counts.float()
    class_weights /= class_weights.sum()
    return class_weights.to(device)