# multi_model_link_prediction_fixed.py
"""
多种知识图谱嵌入模型对比实验 - 完整版
包含 TransE/DistMult/ComplEx/RotatE/WeldLink 及消融实验
多模态特征从节点CSV属性列自动提取
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import random
from tqdm import tqdm


# ==================== 配置 ====================
class Config:
    DIM = 64
    BATCH_SIZE = 64
    EPOCHS = 200
    LEARNING_RATE = 0.001
    MARGIN = 1.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EVAL_FREQ = 20
    PATIENCE = 50
    DROPOUT = 0.3


config = Config()


# ==================== 数据加载 ====================
def extract_features_from_nodes(nodes_df, entity2id, fusion_dim=512):
    """
    从节点CSV中自动提取多模态特征
    优先提取数值列，再对文本列做简单编码
    如果没有足够的数值列，生成基于hash的固定特征
    """
    print("\n📊 从节点文件中提取多模态特征...")

    num_entities = len(entity2id)

    # 创建反向映射
    id2entity = {v: k for k, v in entity2id.items()}

    # 找到节点ID列名
    id_col = None
    for col in ['节点ID', 'entity_id', 'id', 'ID', 'node_id']:
        if col in nodes_df.columns:
            id_col = col
            break

    if id_col is None:
        print("  ⚠️ 未找到节点ID列，使用第一列作为ID")
        id_col = nodes_df.columns[0]

    print(f"  使用ID列: {id_col}")

    # 找数值列（排除ID列和非特征列）
    exclude_cols = {id_col, 'name', '名称', 'label', '标签', 'type', '类型'}
    numeric_cols = []
    text_cols = []

    for col in nodes_df.columns:
        if col in exclude_cols:
            continue
        if nodes_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            numeric_cols.append(col)
        elif nodes_df[col].dtype == object:
            text_cols.append(col)

    print(f"  数值列: {numeric_cols}")
    print(f"  文本列: {text_cols}")

    # 构建特征字典
    fusion_features = {}

    for _, row in nodes_df.iterrows():
        entity_str = str(row[id_col])
        if entity_str not in entity2id:
            continue

        eid = entity2id[entity_str]
        feat_list = []

        # 1. 提取数值特征
        for col in numeric_cols:
            val = row[col]
            if pd.notna(val):
                feat_list.append(float(val))
            else:
                feat_list.append(0.0)

        # 2. 文本特征简单hash编码
        for col in text_cols:
            val = str(row[col]) if pd.notna(row[col]) else ""
            # 用hash生成固定维度的特征
            hash_val = hash(val) % 10000
            feat_list.append(float(hash_val) / 10000.0)

        # 3. 如果特征太少，用hash填充到fusion_dim
        if len(feat_list) == 0:
            # 基于实体ID生成固定的"伪特征"
            seed = hash(entity_str)
            np.random.seed(seed)
            feat_list = np.random.randn(fusion_dim).tolist()
        elif len(feat_list) < fusion_dim:
            # 用原始特征seed扩展
            seed = int(sum(feat_list) * 1000) % (2 ** 31)
            np.random.seed(seed)
            extra = np.random.randn(fusion_dim - len(feat_list)) * 0.1
            feat_list.extend(extra.tolist())
        else:
            feat_list = feat_list[:fusion_dim]

        fusion_features[eid] = np.array(feat_list, dtype=np.float32)

    actual_dim = list(fusion_features.values())[0].shape[0] if fusion_features else 0
    print(f"  ✅ 提取了 {len(fusion_features)} 个实体的特征, 维度: {actual_dim}")

    return fusion_features, actual_dim


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

    # 从节点文件提取多模态特征
    fusion_features, fusion_dim = extract_features_from_nodes(nodes_df, entity2id)

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

    return train_triples, valid_triples, test_triples, entity2id, relation2id, fusion_features, fusion_dim


class KGDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = [(entity2id[h], relation2id[r], entity2id[t]) for h, r, t in triples]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


# ==================== TransE ====================
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
        return torch.norm(h_emb + r_emb - t_emb, p=2, dim=-1)

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


# ==================== DistMult ====================
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
        return -torch.sum(h_emb * r_emb * t_emb, dim=-1)

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin - pos_score + neg_score))


# ==================== ComplEx ====================
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
        score_re = h_re * r_re * t_re + h_im * r_im * t_re + h_re * r_im * t_im - h_im * r_re * t_im
        return -torch.sum(score_re, dim=-1)

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin - pos_score + neg_score))


# ==================== RotatE ====================
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim, margin=1.0):
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, dim * 2)
        self.relation_emb = nn.Embedding(num_relations, dim)
        self.margin = margin
        self.dim = dim
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        h_re, h_im = h_emb.chunk(2, dim=-1)
        t_re, t_im = t_emb.chunk(2, dim=-1)
        r_phase = r_emb
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re
        return torch.sqrt((h_rot_re - t_re) ** 2 + (h_rot_im - t_im) ** 2).sum(dim=-1)

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


# ==================== WeldLink ====================
class WeldLink(nn.Module):
    """融合跨模态特征的链路预测模型"""

    def __init__(self, num_entities, num_relations, dim,
                 fusion_features_dict, fusion_dim=512, margin=1.0, dropout=0.3):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        self.margin = margin

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # 注册跨模态融合特征
        fusion_tensor = torch.zeros(num_entities, fusion_dim)
        for eid, feat in fusion_features_dict.items():
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float()
            elif isinstance(feat, list):
                feat = torch.tensor(feat).float()
            fusion_tensor[eid] = feat
        self.register_buffer('fusion_features', fusion_tensor)

        # 实体特征增强网络
        self.W_g = nn.Linear(dim + fusion_dim, dim, bias=True)
        self.W_trans = nn.Linear(fusion_dim, dim, bias=False)

        # 关系专用变换层
        self.relation_transforms = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(num_relations)
        ])

        # 输出投影层
        self.head_proj = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tail_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def entity_enhance(self, entity_ids):
        h_s = self.entity_emb(entity_ids)
        F_fused = self.fusion_features[entity_ids]
        concat_feat = torch.cat([h_s, F_fused], dim=-1)
        h_gate = torch.sigmoid(self.W_g(concat_feat))
        trans_feat = torch.tanh(self.W_trans(F_fused))
        return h_s + h_gate * trans_feat

    def relation_transform(self, r, r_types):
        batch_size = r.shape[0]
        r_t = torch.zeros_like(r)
        for rel_id in range(self.num_relations):
            mask = (r_types == rel_id)
            if mask.any():
                r_t[mask] = self.relation_transforms[rel_id](r[mask])
        return r_t

    def forward(self, h, r, t):
        h_enh = self.entity_enhance(h)
        t_enh = self.entity_enhance(t)
        r_emb = self.relation_emb(r)
        r_transformed = self.relation_transform(r_emb, r)
        head_rel = torch.cat([h_enh, r_transformed], dim=-1)
        head_proj = self.head_proj(head_rel)
        tail_proj = self.tail_proj(t_enh)
        return torch.norm(head_proj - tail_proj, p=2, dim=-1)

    def loss(self, pos_score, neg_score):
        return torch.mean(torch.relu(self.margin + pos_score - neg_score))


# ==================== WeldLink 消融变体 ====================
class WeldLinkAblation(nn.Module):
    """可控制模块开关的WeldLink变体"""

    def __init__(self, num_entities, num_relations, dim,
                 fusion_features_dict, fusion_dim=512, margin=1.0, dropout=0.3,
                 use_enhancement=True, use_relation_transform=True):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        self.margin = margin
        self.use_enhancement = use_enhancement
        self.use_relation_transform = use_relation_transform

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        fusion_tensor = torch.zeros(num_entities, fusion_dim)
        for eid, feat in fusion_features_dict.items():
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float()
            elif isinstance(feat, list):
                feat = torch.tensor(feat).float()
            fusion_tensor[eid] = feat
        self.register_buffer('fusion_features', fusion_tensor)

        if use_enhancement:
            self.W_g = nn.Linear(dim + fusion_dim, dim, bias=True)
            self.W_trans = nn.Linear(fusion_dim, dim, bias=False)

        if use_relation_transform:
            self.relation_transforms = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(num_relations)
            ])

        self.head_proj = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tail_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def entity_enhance(self, entity_ids):
        h_s = self.entity_emb(entity_ids)
        if not self.use_enhancement:
            return h_s
        F_fused = self.fusion_features[entity_ids]
        concat_feat = torch.cat([h_s, F_fused], dim=-1)
        h_gate = torch.sigmoid(self.W_g(concat_feat))
        trans_feat = torch.tanh(self.W_trans(F_fused))
        return h_s + h_gate * trans_feat

    def relation_transform(self, r, r_types):
        if not self.use_relation_transform:
            return r
        batch_size = r.shape[0]
        r_t = torch.zeros_like(r)
        for rel_id in range(self.num_relations):
            mask = (r_types == rel_id)
            if mask.any():
                r_t[mask] = self.relation_transforms[rel_id](r[mask])
        return r_t

    def forward(self, h, r, t):
        h_enh = self.entity_enhance(h)
        t_enh = self.entity_enhance(t)
        r_emb = self.relation_emb(r)
        r_transformed = self.relation_transform(r_emb, r)
        head_rel = torch.cat([h_enh, r_transformed], dim=-1)
        head_proj = self.head_proj(head_rel)
        tail_proj = self.tail_proj(t_enh)
        return torch.norm(head_proj - tail_proj, p=2, dim=-1)

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
        self.model.eval()
        all_entities = list(range(len(entity2id)))
        ranks = []

        with torch.no_grad():
            for h, r, t in tqdm(eval_triples, desc="评估", leave=False):
                h_idx = entity2id[h]
                r_idx = relation2id[r]
                t_idx = entity2id[t]
                h_tensor = torch.tensor([h_idx] * len(all_entities)).to(self.device)
                r_tensor = torch.tensor([r_idx] * len(all_entities)).to(self.device)
                t_tensor = torch.tensor(all_entities).to(self.device)
                scores = self.model(h_tensor, r_tensor, t_tensor)
                _, indices = torch.sort(scores)
                rank = (indices == t_idx).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)

        mrr = np.mean(1.0 / np.array(ranks))
        hits = {k: np.mean(np.array(ranks) <= k) for k in hits_k}
        return {'MRR': mrr, **{f'Hits@{k}': v for k, v in hits.items()}}


# ==================== 消融实验 ====================
def run_ablation_experiment(entity2id, relation2id, fusion_features, fusion_dim,
                            train_triples, valid_triples, test_triples, config):
    """运行WeldLink消融实验"""

    variants = {
        'WeldLink (完整)': (True, True),
        'w/o Enhancement': (False, True),
        'w/o Relation Transform': (True, False),
        'w/o Both (Base_Multi)': (False, False)
    }

    results = {}

    train_dataset = KGDataset(train_triples, entity2id, relation2id)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    for name, (use_enh, use_rel) in variants.items():
        print(f"\n{'=' * 50}")
        print(f"🔬 消融实验: {name}")
        print(f"{'=' * 50}")

        model = WeldLinkAblation(
            len(entity2id), len(relation2id), config.DIM,
            fusion_features, fusion_dim=fusion_dim, margin=config.MARGIN,
            dropout=config.DROPOUT,
            use_enhancement=use_enh,
            use_relation_transform=use_rel
        )

        trainer = KGTrainer(model, config.DEVICE, lr=config.LEARNING_RATE)

        best_mrr = 0
        patience_counter = 0

        for epoch in range(1, config.EPOCHS + 1):
            loss = trainer.train_epoch(train_loader, len(entity2id))

            if epoch % config.EVAL_FREQ == 0:
                metrics = trainer.evaluate(valid_triples, entity2id, relation2id)
                print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, "
                      f"Hits@1={metrics['Hits@1']:.4f}, Hits@10={metrics['Hits@10']:.4f}")

                if metrics['MRR'] > best_mrr:
                    best_mrr = metrics['MRR']
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= config.PATIENCE:
                    print(f"  ⏳ 早停于 epoch {epoch}")
                    break

        print(f"\n  📊 测试集评估...")
        test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
        results[name] = test_metrics
        print(f"\n✅ {name} 测试结果:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.4f}")

    print("\n" + "=" * 70)
    print("📊 消融实验结果汇总:")
    print("-" * 70)
    print(f"{'模型变体':<25} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}")
    print("-" * 70)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['MRR']:>8.4f} {metrics['Hits@1']:>8.4f} "
              f"{metrics['Hits@3']:>8.4f} {metrics['Hits@10']:>8.4f}")

    return results


# ==================== 主实验 ====================
def run_experiment():
    print("=" * 70)
    print("🔬 多模型链路预测对比实验 (含WeldLink及消融实验)")
    print("=" * 70)
    print(f"\n⚙️ 配置:")
    print(f"  嵌入维度: {config.DIM}")
    print(f"  批次大小: {config.BATCH_SIZE}")
    print(f"  训练轮数: {config.EPOCHS}")
    print(f"  学习率: {config.LEARNING_RATE}")
    print(f"  设备: {config.DEVICE}")

    DATA_DIR = r"D:\phototextCode"
    NODES_PATH = os.path.join(DATA_DIR, "融合后节点_带注意力.csv")
    RELATIONS_PATH = os.path.join(DATA_DIR, "融合后关系_带注意力.csv")

    if not os.path.exists(NODES_PATH):
        print(f"\n❌ 节点文件不存在: {NODES_PATH}")
        return

    # 加载数据（自动提取特征）
    train_triples, valid_triples, test_triples, entity2id, relation2id, fusion_features, fusion_dim = \
        load_data_from_fusion(NODES_PATH, RELATIONS_PATH)

    train_dataset = KGDataset(train_triples, entity2id, relation2id)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    results = {}

    # ======== 基线模型 ========
    baseline_models = {
        'TransE': TransE(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'DistMult': DistMult(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'ComplEx': ComplEx(len(entity2id), len(relation2id), config.DIM, margin=config.MARGIN),
        'RotatE': RotatE(len(entity2id), len(relation2id), config.DIM // 2, margin=config.MARGIN)
    }

    for model_name, model in baseline_models.items():
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
                print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, "
                      f"Hits@10={metrics['Hits@10']:.4f}")
                if metrics['MRR'] > best_mrr:
                    best_mrr = metrics['MRR']
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print(f"  ⏳ 早停于 epoch {epoch}")
                    break

        print(f"\n  📊 测试集评估...")
        test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
        results[model_name] = test_metrics
        print(f"\n✅ {model_name} 测试结果:")
        for metric, value in test_metrics.items():
            print(f"   {metric}: {value:.4f}")

    # ======== WeldLink 完整模型 ========
    print(f"\n{'=' * 50}")
    print(f"训练模型: WeldLink (Ours)")
    print(f"{'=' * 50}")

    weldlink = WeldLink(
        len(entity2id), len(relation2id), config.DIM,
        fusion_features, fusion_dim=fusion_dim, margin=config.MARGIN,
        dropout=config.DROPOUT
    )

    trainer = KGTrainer(weldlink, config.DEVICE, lr=config.LEARNING_RATE)
    best_mrr = 0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        loss = trainer.train_epoch(train_loader, len(entity2id))
        if epoch % config.EVAL_FREQ == 0:
            metrics = trainer.evaluate(valid_triples, entity2id, relation2id)
            print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, "
                  f"Hits@1={metrics['Hits@1']:.4f}, Hits@10={metrics['Hits@10']:.4f}")
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  ⏳ 早停于 epoch {epoch}")
                break

    test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
    results['WeldLink (Ours)'] = test_metrics
    print(f"\n✅ WeldLink 测试结果:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.4f}")

    # ======== WeldLink-v1 (dropout=0.2) ========
    print(f"\n{'=' * 50}")
    print(f"训练模型: WeldLink-v1 (dropout=0.2)")
    print(f"{'=' * 50}")

    weldlink_v1 = WeldLink(
        len(entity2id), len(relation2id), config.DIM,
        fusion_features, fusion_dim=fusion_dim, margin=config.MARGIN,
        dropout=0.2
    )

    trainer = KGTrainer(weldlink_v1, config.DEVICE, lr=config.LEARNING_RATE)
    best_mrr = 0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        loss = trainer.train_epoch(train_loader, len(entity2id))
        if epoch % config.EVAL_FREQ == 0:
            metrics = trainer.evaluate(valid_triples, entity2id, relation2id)
            print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, "
                  f"Hits@1={metrics['Hits@1']:.4f}")
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  ⏳ 早停于 epoch {epoch}")
                break

    test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
    results['WeldLink-v1'] = test_metrics
    print(f"\n✅ WeldLink-v1 测试结果:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.4f}")

    # ======== WeldLink-v2 (dim=128, dropout=0.2) ========
    print(f"\n{'=' * 50}")
    print(f"训练模型: WeldLink-v2 (dim=128, dropout=0.2)")
    print(f"{'=' * 50}")

    weldlink_v2 = WeldLink(
        len(entity2id), len(relation2id), 128,
        fusion_features, fusion_dim=fusion_dim, margin=config.MARGIN,
        dropout=0.2
    )

    trainer = KGTrainer(weldlink_v2, config.DEVICE, lr=config.LEARNING_RATE)
    best_mrr = 0
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        loss = trainer.train_epoch(train_loader, len(entity2id))
        if epoch % config.EVAL_FREQ == 0:
            metrics = trainer.evaluate(valid_triples, entity2id, relation2id)
            print(f"  Epoch {epoch:3d}: Loss={loss:.4f}, MRR={metrics['MRR']:.4f}, "
                  f"Hits@1={metrics['Hits@1']:.4f}")
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  ⏳ 早停于 epoch {epoch}")
                break

    test_metrics = trainer.evaluate(test_triples, entity2id, relation2id)
    results['WeldLink-v2'] = test_metrics
    print(f"\n✅ WeldLink-v2 测试结果:")
    for metric, value in test_metrics.items():
        print(f"   {metric}: {value:.4f}")

    # ======== 消融实验 ========
    print(f"\n{'=' * 70}")
    print(f"🔬 开始消融实验")
    print(f"{'=' * 70}")

    ablation_results = run_ablation_experiment(
        entity2id, relation2id, fusion_features, fusion_dim,
        train_triples, valid_triples, test_triples, config
    )

    for name, metrics in ablation_results.items():
        if name not in results:
            results[name] = metrics

    # ======== 保存结果 ========
    results_df = pd.DataFrame(results).T
    results_df.to_csv('fusion_model_comparison_results.csv', encoding='utf-8')

    print("\n" + "=" * 70)
    print("📊 所有模型对比结果汇总:")
    print("-" * 70)
    print(f"{'模型':<25} {'MRR':>8} {'Hits@1':>8} {'Hits@3':>8} {'Hits@10':>8}")
    print("-" * 70)
    for name in results_df.index:
        row = results_df.loc[name]
        print(f"{name:<25} {row['MRR']:>8.4f} {row['Hits@1']:>8.4f} "
              f"{row['Hits@3']:>8.4f} {row['Hits@10']:>8.4f}")

    print("\n✅ 结果已保存: fusion_model_comparison_results.csv")
    return results_df


if __name__ == "__main__":
    run_experiment()