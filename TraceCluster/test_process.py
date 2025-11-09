def TestDataset(file_path):
    feature_map = {}
    feature_num = 0
    with open('feature.txt', 'r', encoding='utf-8') as f_feature:
        for line in f_feature:
            temp = line.strip('\n').split('\t')
            if len(temp) != 2:
                continue
            feature_map[temp[0]] = int(temp[1])
            feature_num += 1

    label_map = {}
    label_num = 0
    with open('label.txt', 'r', encoding='utf-8') as f_label:
        for line in f_label:
            temp = line.strip('\n').split('\t')
            if len(temp) != 2:
                continue
            label_map[temp[0]] = int(temp[1])
            label_num += 1

    ground_truth = {}
    gt_path = 'groundtruth.txt'

    if os.path.exists(gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f_gt:
            for line in f_gt:
                uuid = line.strip('\n')
                if uuid:
                    ground_truth[uuid] = 1
    else:
        print(f"[warn] ：{gt_path}")

    node_cnt = 0
    provenance = []   # [srcId, srcTypeIdx, dstId, dstTypeIdx, edgeTypeIdx]
    edge_s, edge_e = [], []
    nodeId_map = {}
    nodeA = []
    adj = {}
    adj2 = {}
    invalid_lines = 0

    fw_gt = open('groundtruth_nodeId.txt', 'w', encoding='utf-8')
    fw_id = open('id_to_uuid.txt', 'w', encoding='utf-8')

    with open(file_path, 'r', encoding='utf-8') as f:
        for raw in f:
            temp = raw.strip('\n').split('\t')

            if len(temp) < 5:
                invalid_lines += 1
                print(f"Invalid line detected (len < 5): {raw.strip()}")
                continue

            src_uuid, src_type, dst_uuid, dst_type, edge_type = temp[0], temp[1], temp[2], temp[3], temp[4]

            if (src_type not in label_map) or (dst_type not in label_map) or (edge_type not in feature_map):
                continue

            if src_uuid not in nodeId_map:
                nodeId_map[src_uuid] = node_cnt
                fw_id.write(f"{node_cnt} {src_uuid}\n")
                if src_uuid in ground_truth:
                    fw_gt.write(f"{nodeId_map[src_uuid]} {src_type} {src_uuid}\n")
                    nodeA.append(node_cnt)
                node_cnt += 1
            if dst_uuid not in nodeId_map:
                nodeId_map[dst_uuid] = node_cnt
                fw_id.write(f"{node_cnt} {dst_uuid}\n")
                if dst_uuid in ground_truth:
                    fw_gt.write(f"{nodeId_map[dst_uuid]} {dst_type} {dst_uuid}\n")
                    nodeA.append(node_cnt)
                node_cnt += 1

            src_id = nodeId_map[src_uuid]
            dst_id = nodeId_map[dst_uuid]
            src_label_idx = label_map[src_type]
            dst_label_idx = label_map[dst_type]
            edge_idx = feature_map[edge_type]
            edge_s.append(src_id)
            edge_e.append(dst_id)
            if dst_id in adj:
                adj[dst_id].append(src_id)
            else:
                adj[dst_id] = [src_id]

            if src_id in adj2:
                adj2[src_id].append(dst_id)
            else:
                adj2[src_id] = [dst_id]

            provenance.append([src_id, src_label_idx, dst_id, dst_label_idx, edge_idx])

    fw_gt.close()
    fw_id.close()

    final_feature_dim = feature_num * 2  #
    x_list = [[0] * final_feature_dim for _ in range(node_cnt)]
    y_list = [0 for _ in range(node_cnt)]

    for src_id, src_label_idx, dst_id, dst_label_idx, edge_idx in provenance:
        x_list[src_id][edge_idx] += 1
        y_list[src_id] = src_label_idx
        x_list[dst_id][edge_idx + feature_num] += 1
        y_list[dst_id] = dst_label_idx

    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor([edge_s, edge_e], dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index)

    test_mask = torch.ones(len(y), dtype=torch.bool)
    data.test_mask = test_mask

    feature_num_out = final_feature_dim

    neibor = set()
    _neibor = {}

    for i in nodeA:
        neibor.add(i)
        if i not in _neibor:
            _neibor[i] = []
        if i not in _neibor[i]:
            _neibor[i].append(i)

        # 入向扩展：i <- j <- k
        if i in adj:
            for j in adj[i]:
                neibor.add(j)
                if j not in _neibor:
                    _neibor[j] = []
                if i not in _neibor[j]:
                    _neibor[j].append(i)

                if j in adj:
                    for k in adj[j]:
                        neibor.add(k)
                        if k not in _neibor:
                            _neibor[k] = []
                        if i not in _neibor[k]:
                            _neibor[k].append(i)

        # 出向扩展：i -> j -> k
        if i in adj2:
            for j in adj2[i]:
                neibor.add(j)
                if j in adj2:
                    for k in adj2[j]:
                        neibor.add(k)

    _nodeA = list(neibor)

    if invalid_lines > 0:
        print(f"[INFO] Skipped {invalid_lines} invalid lines (len < 5).")
    print(f"[INFO] Nodes: {node_cnt}, Edges: {len(edge_s)}, "
          f"FeatureDim(final): {feature_num_out}, LabelNum: {label_num}")

    return [data], feature_num_out, label_num, adj, adj2, nodeA, _nodeA, _neibor