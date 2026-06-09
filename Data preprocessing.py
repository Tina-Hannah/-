import pandas as pd
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

# ========================== 1. 配置参数（请根据你的路径修改）==========================
NODE_CSV_PATH = r"D:\IMF-Pytorch-main\IMF-Pytorch-main\融合后节点1.csv"  # 你的节点数据路径
REL_CSV_PATH = r"D:\IMF-Pytorch-main\IMF-Pytorch-main\融合后关系1.csv"  # 你的关系数据路径
OUTPUT_DIR = r"D:\IMF-Pytorch-main\IMF-Pytorch-main\imf_data"  # 输出文件夹

# 特征配置：根据你实体属性的实际字段调整
FEATURE_CONFIG = {
    "bbox_dim": 4,  # bbox包含4个坐标值（x1,y1,x2,y2）
    "density_map": {"low": 0, "medium": 1, "high": 2, "very_high": 3},  # density映射
    "orientation_map": {"horizontal": 0, "vertical": 1, "other": 2}  # orientation映射
}


# ========================== 2. 工具函数（复用IMF-Pytorch核心逻辑）==========================
def write_json(data, path):
    """复用IMF的json写入逻辑，保存ID映射字典"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_pkl(data, path):
    """复用IMF的pkl写入逻辑，保存实体特征"""
    with open(path, "wb") as f:
        pickle.dump(data, f)


def parse_entity_attr(attr_str, config):
    """解析实体属性字段，生成特征向量（核心：适配你的属性格式）"""
    # 1. 清理属性字符串（处理Neo4j导出的格式）
    attr_str = attr_str.replace("{", "").replace("}", "").replace("'", "").replace('"', '')
    attr_dict = {}
    for item in attr_str.split(", "):
        if ": " in item:
            key, val = item.split(": ", 1)
            attr_dict[key.strip()] = val.strip()

    # 2. 提取特征（按配置生成向量）
    feature = []
    # 2.1 解析bbox（坐标值转浮点数）
    if "bbox" in attr_dict:
        bbox = attr_dict["bbox"].replace("[", "").replace("]", "").split(", ")
        bbox = [float(x) for x in bbox[:config["bbox_dim"]]]  # 取前4个坐标
        feature.extend(bbox)
    else:
        feature.extend([0.0] * config["bbox_dim"])  # 缺失值用0填充

    # 2.2 解析density（映射为整数）
    if "density" in attr_dict:
        density_val = attr_dict["density"].lower()
        density_code = config["density_map"].get(density_val, 0)  # 未知值用0
        feature.append(density_code)
    else:
        feature.append(0)

    # 2.3 解析orientation（映射为整数）
    if "orientation" in attr_dict:
        orient_val = attr_dict["orientation"].lower()
        orient_code = config["orientation_map"].get(orient_val, 2)  # 未知值用other
        feature.append(orient_code)
    else:
        feature.append(2)

    # 3. 特征归一化（复用IMF的F.normalize）
    feature_tensor = torch.tensor(feature, dtype=torch.float32)
    feature_norm = F.normalize(feature_tensor, p=2, dim=0).tolist()  # L2归一化
    return feature_norm


# ========================== 3. 实体处理：ID映射+特征编码 ==========================
def process_entities(node_df, config, output_dir):
    """
    输入：节点DataFrame
    输出：entity2id.json、entity_features.pkl
    """
    # 3.1 实体ID映射（将原始节点ID转为连续整数）
    original_entity_ids = node_df["节点ID"].unique().tolist()
    entity2id = {str(orig_id): idx for idx, orig_id in enumerate(original_entity_ids)}  # 原始ID转字符串避免类型问题
    write_json(entity2id, f"{output_dir}/entity2id.json")
    print(f"✅ 实体ID映射完成：共{len(entity2id)}个实体")

    # 3.2 实体多模态特征编码
    entity_features = {}
    for _, row in node_df.iterrows():
        orig_entity_id = str(row["节点ID"])  # 匹配entity2id的key类型
        entity_idx = entity2id[orig_entity_id]
        attr_str = row["所有属性"]

        # 生成特征向量
        feature_vec = parse_entity_attr(attr_str, config)
        entity_features[entity_idx] = feature_vec  # 用映射后的ID作为key

    # 保存特征
    write_pkl(entity_features, f"{output_dir}/entity_features.pkl")
    print(f"✅ 实体特征编码完成：特征维度{len(next(iter(entity_features.values())))}")
    return entity2id


# ========================== 4. 关系处理：ID映射+三元组生成 ==========================
def process_relations(rel_df, entity2id, output_dir):
    """
    输入：关系DataFrame、实体ID映射
    输出：relation2id.json、train.txt/valid.txt/test.txt
    """
    # 4.1 关系类型ID映射
    relation_types = rel_df["关系类型"].unique().tolist()
    relation2id = {rel_type: idx for idx, rel_type in enumerate(relation_types)}
    write_json(relation2id, f"{output_dir}/relation2id.json")
    print(f"✅ 关系ID映射完成：共{len(relation2id)}种关系")

    # 4.2 生成三元组（h, r, t）：原始ID→映射后ID
    triples = []
    for _, row in rel_df.iterrows():
        # 转换头实体、尾实体、关系的ID
        h_orig = str(row["起始节点ID"])
        t_orig = str(row["结束节点ID"])
        r_type = row["关系类型"]

        # 跳过实体ID不在映射中的异常数据
        if h_orig not in entity2id or t_orig not in entity2id or r_type not in relation2id:
            continue

        h = entity2id[h_orig]
        r = relation2id[r_type]
        t = entity2id[t_orig]
        triples.append((h, r, t))

    print(f"✅ 三元组生成完成：共{len(triples)}个有效三元组")

    # 4.3 划分训练集/验证集/测试集（复用IMF的8:1:1划分逻辑）
    train_triples, temp_triples = train_test_split(triples, test_size=0.2, random_state=42)
    valid_triples, test_triples = train_test_split(temp_triples, test_size=0.5, random_state=42)

    # 保存三元组（IMF要求的txt格式：每行h r t）
    def save_triples(triple_list, path):
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in triple_list:
                f.write(f"{h} {r} {t}\n")

    save_triples(train_triples, f"{output_dir}/train.txt")
    save_triples(valid_triples, f"{output_dir}/valid.txt")
    save_triples(test_triples, f"{output_dir}/test.txt")
    print(f"✅ 数据集划分完成：训练集{len(train_triples)} | 验证集{len(valid_triples)} | 测试集{len(test_triples)}")
    return relation2id


# ========================== 5. 主函数：执行完整预处理流程 ==========================
if __name__ == "__main__":
    # 1. 读取原始数据
    print("=" * 50)
    print("1. 读取Neo4j导出数据")
    print("=" * 50)
    node_df = pd.read_csv(NODE_CSV_PATH)
    rel_df = pd.read_csv(REL_CSV_PATH)
    print(f"节点数据：{node_df.shape}行 | 关系数据：{rel_df.shape}行")

    # 2. 处理实体
    print("\n" + "=" * 50)
    print("2. 处理实体（ID映射+特征编码）")
    print("=" * 50)
    entity2id = process_entities(node_df, FEATURE_CONFIG, OUTPUT_DIR)

    # 3. 处理关系
    print("\n" + "=" * 50)
    print("3. 处理关系（ID映射+三元组生成）")
    print("=" * 50)
    relation2id = process_relations(rel_df, entity2id, OUTPUT_DIR)

    # 4. 输出总结
    print("\n" + "=" * 50)
    print("预处理完成！输出文件列表：")
    print("=" * 50)
    output_files = [
        "entity2id.json（实体ID映射）",
        "relation2id.json（关系ID映射）",
        "entity_features.pkl（实体多模态特征）",
        "train.txt（训练集三元组）",
        "valid.txt（验证集三元组）",
        "test.txt（测试集三元组）"
    ]
    for file in output_files:
        print(f"✅ {OUTPUT_DIR}/{file.split('（')[0]}")
    print(f"\n提示：将{OUTPUT_DIR}文件夹复制到IMF-Pytorch的data目录下，即可开始训练！")