def calculate_metrics(groundtruth_file: str, alarm_file: str, id_to_uuid_file: str):
    """
    Calculate precision, recall, and F-score based on ground truth and alarm data.
    
    :param groundtruth_file: Path to the ground truth file.
    :param alarm_file: Path to the alarm file.
    :param id_to_uuid_file: Path to the id to uuid mapping file.
    :return: Precision, Recall, F-score
    """
    # Load node mapping (id -> uuid)
    node_map = {}
    with open(id_to_uuid_file, 'r') as f:
        for line in f:
            node_id, uuid = line.strip().split(' ')
            node_map[int(node_id)] = uuid

    # Load ground truth (node id -> truth)
    gt = {}
    with open(groundtruth_file, 'r') as f_gt:
        for line in f_gt:
            node_id = int(line.strip().split(' ')[0])
            gt[node_id] = 1  # Mark as ground truth node

    # Initialize answer list
    ans = {}

    # Read alarm file and classify nodes
    with open(alarm_file, 'r') as f_alarm:
        for line in f_alarm:
            if line == '\n': continue
            if ':' not in line:
                tot_node = int(line.strip())
                # Initially mark all as true negatives
                for i in range(tot_node):
                    ans[i] = 'tn'
                # Mark ground truth nodes as false negatives
                for i in gt:
                    ans[i] = 'fn'
                continue
            
            # Process the alarm node and its neighbors
            a, b = line.strip().split(':')
            a = int(a)
            b = b.strip().split(' ')
            flag = 0
            
            # Mark true positives
            for i in b:
                if i and int(i) in gt:
                    ans[int(i)] = 'tp'
                    flag = 1

            # If the current node is ground truth, mark as true positive
            if a in gt:
                ans[a] = 'tp'
            elif flag == 0:
                ans[a] = 'fp'

    # Calculate metrics
    tp = sum(1 for x in ans.values() if x == 'tp')
    tn = sum(1 for x in ans.values() if x == 'tn')
    fp = sum(1 for x in ans.values() if x == 'fp')
    fn = sum(1 for x in ans.values() if x == 'fn')

    # Calculate precision, recall, and F-score
    eps = 1e-10  # To avoid division by zero
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    fscore = 2 * precision * recall / (precision + recall + eps)

    # Print results
    print(f"True Positives: {tp}, False Positives: {fp}, True Negatives: {tn}, False Negatives: {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Score: {fscore}")
    
    return precision, recall, fscore


# Example usage:
groundtruth_file = 'groundtruth_nodeId.txt'
alarm_file = 'alarm.txt'
id_to_uuid_file = 'id_to_uuid.txt'

calculate_metrics(groundtruth_file, alarm_file, id_to_uuid_file)
