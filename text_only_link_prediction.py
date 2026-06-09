# text_only_link_prediction_with_ranking.py
"""
纯文本链路预测 - 添加排序评估指标（修复版）
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import defaultdict
import warnings
import json
import os
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')


class TextOnlyLinkPredictorWithRanking:
    """支持排序评估的纯文本链路预测器"""

    def __init__(self, nodes_path, relations_path):
        print("📂 加载纯文本数据...")
        self.nodes_df = pd.read_csv(nodes_path, encoding='utf-8')
        self.relations_df = pd.read_csv(relations_path, encoding='utf-8')

        print(f"  ✅ 节点数: {len(self.nodes_df)}")
        print(f"  ✅ 关系数: {len(self.relations_df)}")

        self.graph = nx.Graph()
        self.vectorizer = TfidfVectorizer(max_features=100, min_df=1)
        self.scaler = StandardScaler()
        self.model = None

        self._build_graph()
        self._prepare_text_features()

    def _build_graph(self):
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(row['节点ID'])
        for _, row in self.relations_df.iterrows():
            self.graph.add_edge(row['起始节点ID'], row['结束节点ID'])
        print(f"✅ 图构建完成: {self.graph.number_of_nodes()} 节点, {self.graph.number_of_edges()} 边")

    def _prepare_text_features(self):
        texts = []
        for _, row in self.nodes_df.iterrows():
            node_name = str(row.get('节点名称', ''))
            node_type = str(row.get('节点类型', ''))
            texts.append(f"{node_name} {node_type}")

        self.text_features = self.vectorizer.fit_transform(texts).toarray()
        self.node_ids = list(self.nodes_df['节点ID'])
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        self.node_names = {}
        self.node_types = {}
        for _, row in self.nodes_df.iterrows():
            nid = row['节点ID']
            self.node_names[nid] = str(row.get('节点名称', ''))
            self.node_types[nid] = str(row.get('节点类型', ''))

        print(f"✅ 文本特征维度: {self.text_features.shape}")

    def extract_pair_features(self, node1, node2):
        features = []
        idx1 = self.node_to_idx.get(node1)
        idx2 = self.node_to_idx.get(node2)

        if idx1 is None or idx2 is None:
            return [0] * 25

        vec1 = self.text_features[idx1]
        vec2 = self.text_features[idx2]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        cos_sim = np.dot(vec1, vec2) / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        features.extend([cos_sim, np.linalg.norm(vec1 - vec2), np.dot(vec1, vec2)])

        deg1 = self.graph.degree(node1)
        deg2 = self.graph.degree(node2)
        features.extend([deg1, deg2, deg1 + deg2, abs(deg1 - deg2), deg1 * deg2])

        try:
            neighbors1 = set(self.graph.neighbors(node1))
            neighbors2 = set(self.graph.neighbors(node2))
            common = len(neighbors1 & neighbors2)
            features.extend([common, common / max(deg1, 1), common / max(deg2, 1)])
        except:
            features.extend([0, 0, 0])

        name1 = self.node_names.get(node1, '').lower()
        name2 = self.node_names.get(node2, '').lower()
        features.extend([len(name1), len(name2)])
        features.append(1 if name1 and name2 and (name1 in name2 or name2 in name1) else 0)

        return features

    def prepare_training_data(self, negative_ratio=0.3):
        print("\n📊 准备训练数据...")
        positive_pairs = []
        for _, row in self.relations_df.iterrows():
            rel_type = row.get('关系类型', 'unknown')
            if rel_type != 'SAME_AS':
                positive_pairs.append((row['起始节点ID'], row['结束节点ID']))

        positive_pairs = list(set(positive_pairs))
        print(f"  正样本数: {len(positive_pairs)}")

        all_nodes = list(self.graph.nodes())
        existing_edges = set(positive_pairs)
        negative_pairs = []
        target_negative = int(len(positive_pairs) * negative_ratio)
        degrees = [(node, self.graph.degree(node)) for node in all_nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        high_degree_nodes = [node for node, deg in degrees[:min(100, len(degrees))]]

        while len(negative_pairs) < target_negative:
            node1 = random.choice(high_degree_nodes)
            node2 = random.choice(all_nodes)
            if node1 != node2 and (node1, node2) not in existing_edges:
                negative_pairs.append((node1, node2))

        print(f"  负样本数: {len(negative_pairs)}")

        X, y = [], []
        for n1, n2 in tqdm(positive_pairs, desc="正样本"):
            X.append(self.extract_pair_features(n1, n2))
            y.append(1)
        for n1, n2 in tqdm(negative_pairs, desc="负样本"):
            X.append(self.extract_pair_features(n1, n2))
            y.append(0)

        return np.array(X), np.array(y)

    def train(self, X, y):
        if len(X) == 0:
            return None

        print("\n🚀 训练模型...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }

        print(f"\n📊 二分类评估:")
        print(f"  准确率: {results['accuracy']:.4f}")
        print(f"  精确率: {results['precision']:.4f}")
        print(f"  召回率: {results['recall']:.4f}")
        print(f"  F1分数: {results['f1']:.4f}")
        print(f"  AUC: {results['auc']:.4f}")

        return results

    def predict_score(self, node1, node2):
        """返回预测分数"""
        if self.model is None:
            return 0.0
        features = self.extract_pair_features(node1, node2)
        X = self.scaler.transform(np.array([features]))
        return self.model.predict_proba(X)[0, 1]

    def evaluate_ranking(self, test_triples, max_samples=100):
        """
        排序评估：给定 (头实体, 关系)，预测尾实体的排名
        这让你可以和知识图谱嵌入模型进行公平对比

        Args:
            test_triples: 测试三元组列表 [(h, r, t), ...]
            max_samples: 最大评估样本数（避免太慢）
        """
        print("\n📊 排序评估 (与KG嵌入模型对比)...")

        all_nodes = list(self.graph.nodes())

        # 限制评估样本数
        if len(test_triples) > max_samples:
            test_triples = test_triples[:max_samples]
            print(f"  限制评估样本数为: {max_samples}")

        ranks = []

        for h, r, t in tqdm(test_triples, desc="排序评估"):
            # 对所有可能的尾实体计算分数
            scores = []
            for candidate in all_nodes:
                if candidate == h:  # 避免预测自身
                    continue
                score = self.predict_score(h, candidate)
                scores.append((candidate, score))

            # 按分数降序排序
            scores.sort(key=lambda x: x[1], reverse=True)

            # 找到真实尾实体的排名
            rank = 1
            found = False
            for cand, _ in scores:
                if cand == t:
                    found = True
                    break
                rank += 1

            if found:
                ranks.append(rank)
            else:
                ranks.append(len(all_nodes))  # 未找到时设为最大排名

        # 计算排序指标
        mrr = np.mean(1.0 / np.array(ranks))
        mean_rank = np.mean(ranks)

        hits = {}
        for k in [1, 3, 10, 100]:
            hits[f'Hits@{k}'] = np.mean(np.array(ranks) <= k)

        ranking_results = {
            'MRR': float(mrr),
            'Mean Rank': float(mean_rank),
            **hits
        }

        print(f"\n📊 排序评估结果 (与KG嵌入模型对比):")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Mean Rank: {mean_rank:.2f}")
        for k in [1, 3, 10, 100]:
            print(f"  Hits@{k}: {hits[f'Hits@{k}']:.4f}")

        return ranking_results


def main():
    print("=" * 70)
    print("📝 纯文本链路预测实验 (含排序评估)")
    print("=" * 70)

    DATA_DIR = r"D:\phototextCode"
    NODES_PATH = os.path.join(DATA_DIR, "text_node.csv")
    RELATIONS_PATH = os.path.join(DATA_DIR, "text_rel.csv")

    # 检查文件
    if not os.path.exists(NODES_PATH):
        print(f"❌ 节点文件不存在: {NODES_PATH}")
        return
    if not os.path.exists(RELATIONS_PATH):
        print(f"❌ 关系文件不存在: {RELATIONS_PATH}")
        return

    # 加载和训练
    predictor = TextOnlyLinkPredictorWithRanking(NODES_PATH, RELATIONS_PATH)
    X, y = predictor.prepare_training_data(negative_ratio=0.3)
    results = predictor.train(X, y)

    # 保存二分类结果
    with open('text_only_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 准备排序评估数据（将关系转换为三元组格式）
    print("\n📂 准备排序评估数据...")
    test_triples = []
    for _, row in predictor.relations_df.iterrows():
        rel_type = row.get('关系类型', 'unknown')
        if rel_type != 'SAME_AS':
            test_triples.append((row['起始节点ID'], rel_type, row['结束节点ID']))

    print(f"  总三元组数: {len(test_triples)}")

    # 排序评估（模拟KG补全任务）
    # 注意：纯文本模型不区分关系类型，所以关系类型在这里只是占位符
    ranking_results = predictor.evaluate_ranking(test_triples, max_samples=50)

    # 保存排序结果
    with open('text_only_ranking_results.json', 'w', encoding='utf-8') as f:
        json.dump(ranking_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("✅ 实验完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()