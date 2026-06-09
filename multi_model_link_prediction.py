# multi_model_link_prediction_fixed.py
"""
多种知识图谱嵌入模型对比实验 - 修复版
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import os
import random
from tqdm import tqdm


# ==================== 配置 ====================
class Config:
    DIM = 64
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    MARGIN = 1.0  # 统一 margin
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_FREQ = 20
    PATIENCE = 50


config = Config()


# ==================== 数据加载 ====================
def load_data_from_fusion(nodes_path, relations_path):
    """从融合后的CSV文件加载数据"""
    print("\n📂 从融合后文件加载数据...")

    nodes_df = pd.read_csv(nodes_path, encoding='utf-8')
    relations_df = pd.read_csv(relations_path, encoding='utf-8')

    print(f"  ✅ 节点数: {len(nodes_df)}")
    print(f"  ✅ 关系数: {len(relations_df)}")

    # 构建实体和关系映射
    entities = set()
    relations = set()
    triples = []

    for _, row in relations_df.iterrows():
        source = str(row['起始节点ID'])
        target = str(row['结束节点ID'])
        rel_type = str(row['关系类型'])

        if rel_type == 'SAME_AS':
            continue

        entities.add(source)
        entities.add(target)
        relations.add(rel_type)
        triples.append((source, rel_type, target))

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    print(f"  ✅ 实体数: {len(entity2id)}")
    print(f"  ✅ 关系数: {len(relation2id)}")
    print(f"  ✅ 三元组数: {len(triples)}")

    # 划分数据集
    random.seed(42)
    random.shuffle(triples)

    train_size = int(0.8 * len(triples))
    valid_size = int(0.1 * len(triples))

    train_triples = triples[:train_size]
    valid_triples = triples[train_size:train_size + valid_size]
    test_triples = triples[train_size + valid_size:]

    print(f"  📊 划分结果: 训练集={len(train_triples)}, 验证集={len(valid_triples)}, 测试集={len(test_triples)}")

    # 显示关系分布
    rel_counts = defaultdict(int)
    for _, r, _ in triples:
        rel_counts[r] += 1
    print(f"\n  📈 关系类型分布:")
    for r, c in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"     {r}: {c}")

    return train_triples, valid_triples, test_triples, entity2id, relation2id


class KGDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


# ==================== TransE 模型 ====================
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)
        return score

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


# ==================== DistMult 模型 ====================
class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        score = torch.sum(h_emb * r_emb * t_emb, dim=-1)
        return -score

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin - pos_score + neg_score))


# ==================== ComplEx 模型 ====================
class ComplEx(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.entity_emb_real = nn.Embedding(num_entities, dim)
        self.entity_emb_imag = nn.Embedding(num_entities, dim)
        self.relation_emb_real = nn.Embedding(num_relations, dim)
        self.relation_emb_imag = nn.Embedding(num_relations, dim)
        self.margin = margin

        nn.init.xavier_uniform_(self.entity_emb_real.weight)
        nn.init.xavier_uniform_(self.entity_emb_imag.weight)
        nn.init.xavier_uniform_(self.relation_emb_real.weight)
        nn.init.xavier_uniform_(self.relation_emb_imag.weight)

    def forward(self, h, r, t):
        h_re = self.entity_emb_real(h)
        h_im = self.entity_emb_imag(h)
        r_re = self.relation_emb_real(r)
        r_im = self.relation_emb_imag(r)
        t_re = self.entity_emb_real(t)
        t_im = self.entity_emb_imag(t)

        # 复数得分计算
        score_re = h_re * r_re * t_re + h_im * r_im * t_re + h_re * r_im * t_im - h_im * r_re * t_im
        score = torch.sum(score_re, dim=-1)
        return -score

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin - pos_score + neg_score))


# ==================== RotatE 模型（修复版） ====================
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        # RotatE 使用复数嵌入，所以实体维度是 dim*2
        self.entity_emb = nn.Embedding(num_entities, dim * 2)
        self.relation_emb = nn.Embedding(num_relations, dim)  # 关系只存储相位
        self.margin = margin
        self.dim = dim

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)  # (batch, 2*dim)
        r_emb = self.relation_emb(r)  # (batch, dim)
        t_emb = self.entity_emb(t)  # (batch, 2*dim)

        # 分离实部和虚部
        h_re, h_im = h_emb.chunk(2, dim=-1)
        t_re, t_im = t_emb.chunk(2, dim=-1)

        # 关系的相位
        r_phase = r_emb
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        # 旋转: h * r
        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re

        # 计算距离
        score = torch.sqrt((h_rot_re - t_re) ** 2 + (h_rot_im - t_im) ** 2).sum(dim=-1)
        return score

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


# ==================== 训练器 ====================
class KGTrainer:
    def __init__(self, model, device='cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.best_mrr = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader, num_entities):
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc="训练", leave=False):
            h, r, t = [x.to(self.device) for x in batch]

            # 生成负样本（替换尾实体）
            neg_t = torch.randint(0, num_entities, t.shape).to(self.device)

            pos_score = self.model(h, r, t)
            neg_score = self.model(h, r, neg_t)

            loss = self.model.loss(pos_score, neg_score)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        return total_loss / max(batch_count, 1)

    def evaluate(self, eval_triples, entity2id, relation2id, hits_k=[1, 3, 10]):
        """评估模型"""
        self.model.eval()

        # 创建反向映射
        id2entity = {v: k for k, v in entity2id.items()}
        all_entities = list(range(len(entity2id)))

        ranks = []

        with torch.no_grad():
            for h, r, t in tqdm(eval_triples, desc="评估", leave=False):
                h_idx = entity2id[h]
                r_idx = relation2id[r]
                t_idx = entity2id[t]

                # 批量计算所有尾实体的得分
                h_tensor = torch.tensor([h_idx] * len(all_entities)).to(self.device)
                r_tensor = torch.tensor([r_idx] * len(all_entities)).to(self.device)
                t_tensor = torch.tensor(all_entities).to(self.device)

                scores = self.model(h_tensor, r_tensor, t_tensor)

                # 计算排名
                _, indices = torch.sort(scores)
                rank = (indices == t_idx).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

        # 计算指标
        mrr = np.mean(1.0 / np.array(ranks))
        hits = {k: np.mean(np.array(ranks) <= k) for k in hits_k}

        return {'MRR': mrr, **{f'Hits@{k}': v for k, v in hits.items()}}


# ==================== 主实验 ====================
def run_experiment():
    print("=" * 70)
    print("🔬 多模型链路预测对比实验 (使用融合后数据)")
    print("=" * 70)
    print(f"\n⚙️ 配置:")
    print(f"  嵌入维度: {config.DIM}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  训练轮数: {config.EPOCHS}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  设备: {config.DEVICE}")

    # 数据路径
    DATA_DIR = r"D:\phototextCode"
    NODES_PATH = os.path.join(DATA_DIR, "融合后节点_带注意力.csv")
    RELATIONS_PATH = os.path.join(DATA_DIR, "融合后关系_带注意力.csv")

    if not os.path.exists(NODES_PATH):
        print(f"\n❌ 节点文件不存在: {NODES_PATH}")
        return

    # 加载数据
    train_triples, valid_triples, test_triples, entity2id, relation2id = load_data_from_fusion(
        NODES_PATH, RELATIONS_PATH
    )

    train_dataset = KGDataset(train_triples, entity2id, relation2id)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 定义模型
    models = {
        'TransE': TransE(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'DistMult': DistMult(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'ComplEx': ComplEx(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'RotatE': RotatE(len(entity2id), len(relation2id), config.DIM // 2, margin=config.MARGIN)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"训练模型: {model_name}")
        print(f"{'=' * 50}")

        trainer = KGTrainer(model, config.DEVICE, lr=config.LEARNING_RATE)

        best_mrr = 0
        patience_counter = 0

        for epoch in range(1, config.EPOCHS + 1):
            loss = trainer.train_epoch(train_loader, len(entity2id))

            if epoch % config.EVAL_FREQ == 0:
                metrics = trainer.evaluate(valid_triples, entity2id, relation2id)
                print(
                    f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, Hits@10={metrics['Hits@10']:.4f}")

                if metrics['MRR'] > best_mrr:
                    best_mrr = metrics['MRR']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.PATIENCE:
                    print(f"  ⏳ 早停于 epoch {epoch}")
                    break

        # 测试评估
        print(f"\n  📊 测试集评估...")
        test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
        results[model_name] = test_metrics
        print(f"\n✅ {model_name} 测试结果:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results).T
    results_df.to_csv('fusion_model_comparison_results.csv', encoding='utf-8')

    print("\n" + "=" * 70)
    print("📊 模型对比结果汇总:")
    print(results_df.round(4))
    print("\n✅ 结果已保存: fusion_model_comparison_results.csv")

    return results_df


if __name__ == "__main__":
    run_experiment()