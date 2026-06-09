import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import pickle
import numpy as np
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# ========================== 1. 配置参数 ==========================
def parse_args():
    parser = argparse.ArgumentParser(description='简化版知识图谱链路预测（使用实体特征）')

    # 数据路径
    parser.add_argument('--data_path', type=str, default=r'D:\IMF-Pytorch-main\IMF-Pytorch-main\imf_data',
                        help='预处理后的数据目录')
    parser.add_argument('--save_path', type=str, default=r'D:\IMF-Pytorch-main\IMF-Pytorch-main\enhanced_model',
                        help='模型权重保存目录')

    # 模型配置
    parser.add_argument('--dim', type=int, default=256, help='关系嵌入维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout率')

    # 训练配置
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')
    parser.add_argument('--epochs', type=int, default=800, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')

    # 评估配置
    parser.add_argument('--eval_freq', type=int, default=20, help='每多少轮评估一次')
    parser.add_argument('--patience', type=int, default=80, help='早停耐心值')
    parser.add_argument('--save_best', type=bool, default=True, help='保存最优模型')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='评估时的批次大小')

    args = parser.parse_args()
    return args


# ========================== 2. 数据加载 ==========================
def load_data(args):
    """加载预处理后的数据集"""
    print("\n" + "=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)

    # 1. 加载ID映射
    with open(os.path.join(args.data_path, 'entity2id.json'), 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    with open(os.path.join(args.data_path, 'relation2id.json'), 'r', encoding='utf-8') as f:
        relation2id = json.load(f)

    n_entity = len(entity2id)
    n_relation = len(relation2id)
    print(f"✅ 实体数量: {n_entity}, 关系数量: {n_relation}")

    # 2. 加载实体特征
    entity_features_path = os.path.join(args.data_path, 'entity_features.pkl')
    if os.path.exists(entity_features_path):
        with open(entity_features_path, 'rb') as f:
            entity_features_raw = pickle.load(f)
        print(f"✅ 加载实体特征: {type(entity_features_raw)}")

        if isinstance(entity_features_raw, dict):
            # 找出最大特征维度
            max_dim = max([len(feat) for feat in entity_features_raw.values()])
            entity_feature_matrix = np.zeros((n_entity, max_dim), dtype=np.float32)
            for eid, feat in entity_features_raw.items():
                if int(eid) < n_entity:
                    entity_feature_matrix[int(eid), :len(feat)] = feat[:max_dim]
            entity_features = torch.tensor(entity_feature_matrix, dtype=torch.float32)
        else:
            entity_features = torch.tensor(entity_features_raw, dtype=torch.float32)

        print(f"   - 特征维度: {entity_features.shape}")
    else:
        print(f"⚠️ 未找到特征文件，使用随机初始化")
        entity_features = torch.randn(n_entity, args.dim, dtype=torch.float32)

    # 3. 加载三元组
    def load_triples(path):
        clean_path = path.replace('.txt', '_clean.txt')
        if os.path.exists(clean_path):
            use_path = clean_path
        else:
            use_path = path

        triples = []
        with open(use_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 3:
                    h, r, t = map(int, parts)
                    triples.append((h, r, t))
        return np.array(triples)

    train_triples = load_triples(os.path.join(args.data_path, 'train.txt'))
    valid_triples = load_triples(os.path.join(args.data_path, 'valid.txt'))
    test_triples = load_triples(os.path.join(args.data_path, 'test.txt'))

    print(f"✅ 训练集: {len(train_triples)}, 验证集: {len(valid_triples)}, 测试集: {len(test_triples)}")

    return {
        'entity2id': entity2id,
        'relation2id': relation2id,
        'entity_features': entity_features,
        'train_triples': train_triples,
        'valid_triples': valid_triples,
        'test_triples': test_triples,
        'n_entity': n_entity,
        'n_relation': n_relation,
    }


# ========================== 3. 改进的模型 ==========================
class ImprovedKGEModel(nn.Module):
    """改进的知识图谱嵌入模型（更好的特征融合）"""

    def __init__(self, entity_features, n_relation, dim=256, dropout=0.3):
        super(ImprovedKGEModel, self).__init__()

        feature_dim = entity_features.shape[1]
        self.n_entity = entity_features.shape[0]
        self.n_relation = n_relation
        self.dim = dim

        # 实体嵌入：使用预训练特征
        self.entity_embedding = nn.Embedding.from_pretrained(entity_features, freeze=False)

        # 特征增强层：提升特征表达能力
        self.feature_enhance = nn.Sequential(
            nn.Linear(feature_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 关系嵌入
        self.relation_embedding = nn.Embedding(n_relation, dim)

        # 关系变换（提升模型表达能力）
        self.relation_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 正则化和dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)

        # 初始化
        nn.init.xavier_uniform_(self.relation_embedding.weight)

        print(f"✅ 改进模型初始化完成")
        print(f"   - 原始特征维度: {feature_dim} -> 嵌入维度: {dim}")
        print(f"   - 可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, batch_input):
        """
        Args:
            batch_input: (batch_size, 2) 包含头实体和关系ID
        Returns:
            scores: (batch_size, n_entity) 预测分数
        """
        h = batch_input[:, 0]  # 头实体
        r = batch_input[:, 1]  # 关系

        # 获取并增强实体嵌入
        h_raw = self.entity_embedding(h)
        h_emb = self.feature_enhance(h_raw)

        # 关系嵌入和变换
        r_raw = self.relation_embedding(r)
        r_emb = self.relation_transform(r_raw)

        # 平移评分: h + r
        h_plus_r = h_emb + r_emb
        h_plus_r = self.layer_norm(h_plus_r)
        h_plus_r = self.dropout(h_plus_r)

        # 获取所有实体的增强表示
        all_entities_raw = self.entity_embedding.weight
        all_entities = self.feature_enhance(all_entities_raw)

        # 计算相似度
        scores = torch.mm(h_plus_r, all_entities.t())

        return scores

    def loss_func(self, scores, target):
        """计算损失函数（支持标签平滑）"""
        return F.binary_cross_entropy_with_logits(scores, target)


# ========================== 4. 评估函数 ==========================
def evaluate(model, triples, n_entity, args):
    """评估模型性能"""
    model.eval()

    hits1, hits3, hits10 = 0, 0, 0
    mr_sum, mrr_sum = 0, 0
    total = len(triples)

    if total == 0:
        print("   ⚠️ 评估集为空")
        return {'MR': 0, 'MRR': 0, 'Hits@1': 0, 'Hits@3': 0, 'Hits@10': 0}

    # 转换为tensor
    all_h = torch.tensor(triples[:, 0], dtype=torch.long).to(args.device)
    all_r = torch.tensor(triples[:, 1], dtype=torch.long).to(args.device)
    all_t = triples[:, 2]

    print(f"   评估中... 共 {total} 个三元组", end="", flush=True)

    with torch.no_grad():
        for i in range(0, total, args.eval_batch_size):
            end_idx = min(i + args.eval_batch_size, total)

            batch_h = all_h[i:end_idx]
            batch_r = all_r[i:end_idx]
            batch_t = all_t[i:end_idx]

            batch_input = torch.stack([batch_h, batch_r], dim=1)

            try:
                scores = model(batch_input)

                for j in range(len(batch_t)):
                    sample_scores = scores[j]
                    true_t = batch_t[j]

                    if torch.isnan(sample_scores).any() or torch.isinf(sample_scores).any():
                        continue
                    if true_t >= n_entity:
                        continue

                    sorted_idx = torch.argsort(sample_scores, descending=True).cpu().numpy()
                    rank_list = np.where(sorted_idx == true_t)[0]

                    if len(rank_list) == 0:
                        continue

                    rank = rank_list[0] + 1
                    mr_sum += rank
                    mrr_sum += 1.0 / rank

                    if rank <= 1:
                        hits1 += 1
                    if rank <= 3:
                        hits3 += 1
                    if rank <= 10:
                        hits10 += 1

            except Exception as e:
                print(f"\n评估错误: {e}")
                continue

    print(" 完成")

    return {
        'MR': round(mr_sum / total, 2),
        'MRR': round(mrr_sum / total, 4),
        'Hits@1': round(hits1 / total * 100, 2),
        'Hits@3': round(hits3 / total * 100, 2),
        'Hits@10': round(hits10 / total * 100, 2)
    }


def predict_links(model, head_entity_id, relation_id, top_k=10):
    """
    预测给定头实体和关系下的尾实体

    Args:
        head_entity_id: 头实体ID
        relation_id: 关系ID
        top_k: 返回top-k个预测

    Returns:
        List of (entity_id, score) 按分数降序排列
    """
    model.eval()
    with torch.no_grad():
        # 准备输入
        batch_input = torch.tensor([[head_entity_id, relation_id]], dtype=torch.long).to(args.device)

        # 预测所有尾实体的分数
        scores = model(batch_input)  # (1, n_entity)

        # 获取top-k
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)

        # 转换为列表
        predictions = [(idx.item(), score.item())
                       for idx, score in zip(top_indices[0], top_scores[0])]

    return predictions


def batch_predict_and_save(model, data, args, output_file='predictions.txt'):
    """
    批量预测并保存结果
    """
    model.eval()

    # 获取ID到名称的映射（如果有）
    id2entity = {v: k for k, v in data['entity2id'].items()}
    id2relation = {v: k for k, v in data['relation2id'].items()}

    predictions = []

    # 示例1：对所有测试三元组进行预测验证
    print("\n" + "=" * 60)
    print("链路预测结果示例")
    print("=" * 60)

    test_triples = data['test_triples'][:20]  # 取前20个示例

    with torch.no_grad():
        for h, r, t in test_triples:
            # 预测
            batch_input = torch.tensor([[h, r]], dtype=torch.long).to(args.device)
            scores = model(batch_input)[0]

            # 获取top-10预测
            top_scores, top_indices = torch.topk(scores, k=10)

            # 真实尾实体的排名
            true_score = scores[t].item()
            true_rank = (scores > scores[t]).sum().item() + 1

            # 保存结果
            result = {
                'head': id2entity.get(h, str(h)),
                'relation': id2relation.get(r, str(r)),
                'true_tail': id2entity.get(t, str(t)),
                'true_rank': true_rank,
                'predictions': [
                    {
                        'entity': id2entity.get(idx.item(), str(idx.item())),
                        'score': score.item()
                    }
                    for idx, score in zip(top_indices, top_scores)
                ]
            }
            predictions.append(result)

            # 打印示例
            print(f"\n头实体: {result['head']}")
            print(f"关系: {result['relation']}")
            print(f"真实尾实体: {result['true_tail']} (排名: {true_rank})")
            print(f"Top-10预测:")
            for i, pred in enumerate(result['predictions'][:10], 1):
                print(f"  {i}. {pred['entity']} (分数: {pred['score']:.4f})")

    # 保存到文件
    with open(os.path.join(args.save_path, output_file), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 预测结果已保存到: {os.path.join(args.save_path, output_file)}")

    return predictions


def predict_for_all_relations(model, data, args, top_k=10):
    """
    为所有关系生成链路预测示例
    """
    model.eval()

    id2entity = {v: k for k, v in data['entity2id'].items()}
    id2relation = {v: k for k, v in data['relation2id'].items()}

    # 选择一些头实体作为示例
    sample_heads = list(data['entity2id'].values())[:5]  # 前5个实体

    all_predictions = {}

    with torch.no_grad():
        for head_id in sample_heads:
            head_name = id2entity[head_id]
            all_predictions[head_name] = {}

            for rel_id, rel_name in id2relation.items():
                # 预测
                batch_input = torch.tensor([[head_id, rel_id]], dtype=torch.long).to(args.device)
                scores = model(batch_input)[0]

                # 获取top-k
                top_scores, top_indices = torch.topk(scores, k=top_k)

                all_predictions[head_name][rel_name] = [
                    {
                        'tail_entity': id2entity.get(idx.item(), str(idx.item())),
                        'score': score.item()
                    }
                    for idx, score in zip(top_indices, top_scores)
                ]

    # 保存结果
    output_file = os.path.join(args.save_path, 'full_predictions.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完整预测结果已保存到: {output_file}")

    return all_predictions

# ========================== 5. 训练主循环 ==========================
def train(args):
    # 1. 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 2. 加载数据
    data = load_data(args)

    # 3. 构建模型
    print("\n" + "=" * 60)
    print("步骤2: 构建改进模型")
    print("=" * 60)

    model = ImprovedKGEModel(
        entity_features=data['entity_features'],
        n_relation=data['n_relation'],
        dim=args.dim,
        dropout=args.dropout
    ).to(args.device)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    # 4. 训练初始化
    best_mrr = 0.0
    best_hits10 = 0.0
    early_stop_count = 0
    train_triples = data['train_triples']
    valid_triples = data['valid_triples']
    test_triples = data['test_triples']
    n_entity = data['n_entity']

    print("\n" + "=" * 60)
    print("步骤3: 开始训练")
    print("=" * 60)
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 60)

    train_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        # 打乱训练集
        np.random.shuffle(train_triples)

        total_loss = 0.0
        n_batches = (len(train_triples) + args.batch_size - 1) // args.batch_size
        valid_batches = 0

        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min((batch_idx + 1) * args.batch_size, len(train_triples))
            pos_batch = train_triples[start:end]

            if len(pos_batch) < 2:
                continue

            valid_batches += 1

            # 准备输入
            batch_input = torch.tensor(pos_batch[:, :2], dtype=torch.long).to(args.device)

            # 创建目标矩阵（使用标签平滑）
            target = torch.zeros(len(pos_batch), n_entity).to(args.device)
            for i, (_, _, t) in enumerate(pos_batch):
                if t < n_entity:
                    # 标签平滑
                    smooth_value = 0.9
                    target[i, t] = smooth_value
                    target[i, :] += (1 - smooth_value) / n_entity

            try:
                # 前向传播
                scores = model(batch_input)

                # 计算损失
                loss = model.loss_func(scores, target)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            except Exception as e:
                print(f"\n批次 {batch_idx} 错误: {e}")
                continue

        scheduler.step()

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            train_losses.append(avg_loss)
        else:
            avg_loss = 0

        # 评估
        if epoch % args.eval_freq == 0 or epoch == 1:
            print(f"\n{'=' * 60}")
            print(f"Epoch [{epoch}/{args.epochs}]")
            print(f"训练损失: {avg_loss:.4f}")

            # 验证集评估
            valid_metrics = evaluate(model, valid_triples, n_entity, args)
            print(f"验证集: MR={valid_metrics['MR']}, MRR={valid_metrics['MRR']}, "
                  f"Hits@1={valid_metrics['Hits@1']}%, Hits@3={valid_metrics['Hits@3']}%, "
                  f"Hits@10={valid_metrics['Hits@10']}%")

            # 保存最优模型（基于MRR）
            if valid_metrics['MRR'] > best_mrr:
                improvement = valid_metrics['MRR'] - best_mrr
                best_mrr = valid_metrics['MRR']
                best_hits10 = valid_metrics['Hits@10']
                early_stop_count = 0

                if args.save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_mrr': best_mrr,
                        'best_hits10': best_hits10,
                        'train_loss': avg_loss,
                        'valid_metrics': valid_metrics
                    }, os.path.join(args.save_path, 'model_best.pth'))
                    print(f"✅ 保存最优模型 (MRR: {best_mrr - improvement:.4f} → {best_mrr:.4f})")
            else:
                early_stop_count += 1
                print(f"⏳ 早停计数: {early_stop_count}/{args.patience} (最佳MRR: {best_mrr:.4f})")

                if early_stop_count >= args.patience:
                    print(f"⚠️ 早停触发")
                    break
        else:
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] - 训练损失: {avg_loss:.4f}")

    # 测试集评估
    print("\n" + "=" * 60)
    print("步骤4: 测试集最终评估")
    print("=" * 60)

    # 加载最优模型
    checkpoint_path = os.path.join(args.save_path, 'model_best.pth')
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载最优模型 (epoch {checkpoint['epoch']}, MRR={checkpoint['best_mrr']:.4f})")
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")

    # 测试评估
    test_metrics = evaluate(model, test_triples, n_entity, args)
    print(f"\n测试集最终结果:")
    print(f"  - MR (Mean Rank): {test_metrics['MR']}")
    print(f"  - MRR: {test_metrics['MRR']}")
    print(f"  - Hits@1: {test_metrics['Hits@1']}%")
    print(f"  - Hits@3: {test_metrics['Hits@3']}%")
    print(f"  - Hits@10: {test_metrics['Hits@10']}%")

    # 测试评估后添加预测
    print("\n" + "=" * 60)
    print("步骤5: 生成链路预测结果")
    print("=" * 60)

    # 加载最优模型
    checkpoint_path = os.path.join(args.save_path, 'model_best.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 1. 生成示例预测
        predictions = batch_predict_and_save(model, data, args, 'example_predictions.json')

        # 2. 可选：生成完整预测
        full_predictions = predict_for_all_relations(model, data, args, top_k=10)

        # 3. 打印几个预测示例
        print("\n" + "=" * 60)
        print("链路预测示例（新三元组）")
        print("=" * 60)

        # 示例：预测某个实体和关系下的尾实体
        sample_entity = list(data['entity2id'].keys())[0]
        sample_relation = list(data['relation2id'].keys())[0]

        head_id = data['entity2id'][sample_entity]
        rel_id = data['relation2id'][sample_relation]

        predictions = predict_links(model, head_id, rel_id, top_k=5)

        print(f"\n预测: ({sample_entity}, {sample_relation}, ?)")
        print("最可能的尾实体:")
        for i, (entity_id, score) in enumerate(predictions, 1):
            entity_name = [k for k, v in data['entity2id'].items() if v == entity_id][0]
            print(f"  {i}. {entity_name} (分数: {score:.4f})")

    # 保存结果
    results = {
        'best_valid_mrr': best_mrr,
        'best_valid_hits10': best_hits10,
        'test_metrics': test_metrics,
        'train_losses': train_losses,
        'config': vars(args)
    }

    with open(os.path.join(args.save_path, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 训练完成！结果保存至: {args.save_path}")
    print(f"   - 模型: model_best.pth")
    print(f"   - 结果: test_results.json")


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("改进版知识图谱链路预测训练")
    print("=" * 60)
    print("训练配置:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 60)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = 'cpu'

    print(f"使用设备: {args.device}")

    torch.manual_seed(42)
    np.random.seed(42)

    train(args)