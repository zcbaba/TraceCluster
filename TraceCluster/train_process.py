def build_graph(file_path: str,
                feature_map: Dict[str, int],
                label_map: Dict[str, int]) -> Tuple[Data, int, int]:

    feature_num = len(feature_map)
    label_num = len(label_map)
    node_cnt = 0
    nodeId_map: Dict[str, int] = {}
    edge_s: List[int] = []
    edge_e: List[int] = []
    provenance: List[Tuple[int, int, int, int, int]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            src_raw, src_type_raw, dst_raw, dst_type_raw, edge_raw = parts[:5]

            if (src_type_raw not in label_map) or (dst_type_raw not in label_map) or (edge_raw not in feature_map):
                continue
            if src_raw not in nodeId_map:
                nodeId_map[src_raw] = node_cnt
                node_cnt += 1
            if dst_raw not in nodeId_map:
                nodeId_map[dst_raw] = node_cnt
                node_cnt += 1

            src_id = nodeId_map[src_raw]
            dst_id = nodeId_map[dst_raw]
            src_type = label_map[src_type_raw]
            dst_type = label_map[dst_type_raw]
            edge_idx = feature_map[edge_raw]

            edge_s.append(src_id)
            edge_e.append(dst_id)
            provenance.append((src_id, src_type, dst_id, dst_type, edge_idx))

    fin_dim = feature_num * 2
    x_list = [[0] * fin_dim for _ in range(node_cnt)]
    y_list = [0] * node_cnt

    for src_id, src_type, dst_id, dst_type, edge_idx in provenance:
        x_list[src_id][edge_idx] += 1
        y_list[src_id] = src_type
        x_list[dst_id][edge_idx + feature_num] += 1
        y_list[dst_id] = dst_type

    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)
    return data, fin_dim, label_num