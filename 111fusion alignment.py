"""
多模态知识图谱融合代码 - 优化召回版（基于原始正确代码最小化修改）
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import ast
import re
from collections import defaultdict
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 配置 ====================
class FusionConfig:
    """融合配置 - 优化召回版"""
    # 相似度权重（保持不变）
    WEIGHTS = {
        'name': 0.05,
        'type': 0.15,
        'semantic': 0.30,
        'relation': 0.10,
        'attention': 0.40
    }

    # ========== 关键修改1：降低粗筛阈值，增加候选对 ==========
    COARSE_THRESHOLD = 0.05      # 从0.10降到0.05
    SUCCESS_THRESHOLD = 0.55     # 保持不变
    FUZZY_THRESHOLD = 0.40       # 保持不变

    # 缺陷类型关键词映射（保持不变）
    DEFECT_KEYWORDS = {
        'Porosity': ['气孔', 'porosity', '气孔区域', '密集气孔', '气泡', '空洞'],
        'Crack': ['裂纹', 'crack', '开裂', '裂缝', '裂痕'],
        'Spatter': ['飞溅', 'spatter', '飞溅聚集', '焊渣'],
        'Undercut': ['咬边', 'undercut', '咬口', '边缘凹陷'],
        'Overlap': ['重叠', 'overlap', '重叠凸起', '搭接'],
        'Pit': ['凹坑', 'pit', '凹陷', '坑', '坑洞']
    }

    # 构件类型映射
    COMPONENT_KEYWORDS = {
        'weld_seam': ['焊缝', '焊缝主体', '焊道', '焊接处'],
        'weld_surface': ['焊缝表面', '焊道表面'],
        'weld_edge': ['焊缝边缘', '焊道边缘'],
        'parent_metal': ['母材', '基材', 'base metal'],
        'weld_junction': ['焊缝交界', '焊缝结合处']
    }

    # ========== 新增：同义词映射表 ==========
    SYNONYM_MAP = {
        '气孔': ['气孔', 'porosity', '气泡', '空洞', 'porous'],
        '裂纹': ['裂纹', 'crack', '裂缝', '开裂', '裂痕', 'cracking'],
        '咬边': ['咬边', 'undercut', '咬口', '边缘凹陷'],
        '飞溅': ['飞溅', 'spatter', '焊渣', '飞溅物', 'spattering'],
        '凹坑': ['凹坑', 'pit', '凹陷', '坑', '坑洞', 'pitting'],
        '未熔合': ['未熔合', 'lack of fusion', '未融合', '熔合不良'],
        '焊缝': ['焊缝', 'weld', '焊道', '焊接处'],
        '母材': ['母材', 'base metal', 'parent metal', '基材']
    }

    # 多头注意力配置
    ATTENTION_CONFIG = {
        'd_model': 128,
        'n_heads': 8,
        'max_seq_len': 32,
        'dropout': 0.1
    }


config = FusionConfig()

# ==================== 多头注意力模块（保持不变） ====================
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(q.device)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


class AttentionFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.max_seq_len = config['max_seq_len']
        self.feature_embedding = nn.Linear(1, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.attention = MultiHeadAttentionLayer(
            d_model=self.d_model, n_heads=config['n_heads'], dropout=config['dropout']
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def encode_feature(self, feature_vec):
        if len(feature_vec) < self.max_seq_len:
            feature_vec = np.pad(feature_vec, (0, self.max_seq_len - len(feature_vec)), 'constant')
        else:
            feature_vec = feature_vec[:self.max_seq_len]
        feature_tensor = torch.FloatTensor(feature_vec).unsqueeze(-1)
        seq_len = feature_tensor.shape[0]
        positions = torch.arange(0, seq_len).unsqueeze(0)
        feature_emb = self.feature_embedding(feature_tensor)
        pos_emb = self.position_embedding(positions)
        return (feature_emb + pos_emb).unsqueeze(0)

    def calculate_attention_similarity(self, text_feature, img_feature):
        text_enc = self.encode_feature(text_feature)
        img_enc = self.encode_feature(img_feature)
        text_attn_out, _ = self.attention(text_enc, img_enc, img_enc)
        img_attn_out, _ = self.attention(img_enc, text_enc, text_enc)
        text_pooled = torch.mean(text_attn_out, dim=1)
        img_pooled = torch.mean(img_attn_out, dim=1)
        similarity = F.cosine_similarity(text_pooled, img_pooled, dim=-1).item()
        return max(0.0, min(1.0, similarity))


attention_model = AttentionFeatureEncoder(config.ATTENTION_CONFIG)
attention_model.eval()


# ==================== 1. 数据预处理 ====================
def clean_node_label(label_str):
    if pd.isna(label_str) or label_str is None or str(label_str).strip() == "":
        return ""
    clean_str = re.sub(r'\[|\]|__\w+__', '', str(label_str))
    parts = [p.strip() for p in clean_str.split(',') if p.strip()]
    chinese_part = [p for p in parts if re.search(r'[\u4e00-\u9fa5]', p)]
    english_part = [p for p in parts if not re.search(r'[\u4e00-\u9fa5]', p)]
    if chinese_part and english_part:
        return f"{chinese_part[0]}|{english_part[0]}"
    elif chinese_part:
        return chinese_part[0]
    elif english_part:
        return english_part[0]
    else:
        return clean_str


def extract_defect_type_from_text(text_name):
    text_lower = text_name.lower()
    for defect_type, keywords in config.DEFECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return defect_type
    return None


def calc_name_similarity(name1, name2):
    clean1 = clean_node_label(name1)
    clean2 = clean_node_label(name2)
    return float(cross_lang_similarity(clean1, clean2))


def cross_lang_similarity(label1, label2):
    if label1 == "" or label2 == "":
        return 0.0
    label1_parts = label1.split('|') if '|' in label1 else [label1]
    label2_parts = label2.split('|') if '|' in label2 else [label2]
    l1_cn = [p for p in label1_parts if re.search(r'[\u4e00-\u9fa5]', p)]
    l1_en = [p for p in label1_parts if not re.search(r'[\u4e00-\u9fa5]', p)]
    l2_cn = [p for p in label2_parts if re.search(r'[\u4e00-\u9fa5]', p)]
    l2_en = [p for p in label2_parts if not re.search(r'[\u4e00-\u9fa5]', p)]
    scores = []
    if l1_cn and l2_cn:
        scores.append(fuzz.ratio(l1_cn[0], l2_cn[0]) / 100)
    if l1_en and l2_en:
        scores.append(fuzz.ratio(l1_en[0].lower(), l2_en[0].lower()) / 100)
    if (l1_cn and l2_en) or (l1_en and l2_cn):
        mix_label1 = ' '.join(label1_parts).lower()
        mix_label2 = ' '.join(label2_parts).lower()
        scores.append(fuzz.partial_ratio(mix_label1, mix_label2) / 100)
    return max(scores) if scores else 0.0


# ========== 关键修改2：增强名称相似度（加入同义词匹配） ==========
def enhanced_name_similarity(text_name, img_name, text_type, img_defect_type):
    """增强的名称相似度计算 - 加入同义词匹配"""
    base_sim = calc_name_similarity(text_name, img_name)

    # 同义词匹配
    text_lower = text_name.lower()
    img_lower = img_name.lower()
    for term, synonyms in config.SYNONYM_MAP.items():
        term_in_text = any(s.lower() in text_lower for s in synonyms)
        term_in_img = any(s.lower() in img_lower for s in synonyms)
        if term_in_text and term_in_img:
            return 0.75

    # 缺陷类型匹配
    if img_defect_type and pd.notna(img_defect_type):
        defect_keywords = config.DEFECT_KEYWORDS.get(img_defect_type, [])
        for keyword in defect_keywords:
            if keyword.lower() in text_name.lower():
                return 0.8

    text_type_clean = clean_node_label(text_type)
    if 'Defect' in text_type_clean or '质量缺陷' in text_type_clean:
        text_defect = extract_defect_type_from_text(text_name)
        if text_defect and img_defect_type:
            if text_defect == img_defect_type:
                return 0.7

    if '施工阶段' in text_type_clean:
        return base_sim * 0.5
    return base_sim


def calc_type_similarity(type1, type2):
    clean1 = clean_node_label(type1)
    clean2 = clean_node_label(type2)
    if not clean1 or not clean2:
        return 0.0
    def extract_core_type(type_str):
        if type_str.startswith('['):
            try:
                type_list = ast.literal_eval(type_str)
                if len(type_list) >= 2:
                    return type_list[-1]
            except:
                pass
        return type_str
    core1 = extract_core_type(clean1)
    core2 = extract_core_type(clean2)
    if core1 == core2:
        return 1.0
    if ('Defect' in core1 or '质量缺陷' in core1) and ('Defect' in core2 or '质量缺陷' in core2):
        return 0.5
    return float(fuzz.ratio(core1.lower(), core2.lower()) / 100)


def get_feature_vector(attr_dict):
    embedding = attr_dict.get('embedding', [])
    if embedding and isinstance(embedding, list) and len(embedding) > 0:
        return np.array(embedding, dtype=np.float32)
    text = str(attr_dict.get('name', attr_dict.get('text', ''))).lower()
    char_features = [ord(c) / 1000 for c in text[:config.ATTENTION_CONFIG['max_seq_len']]]
    defect_features = []
    for defect_type in config.DEFECT_KEYWORDS.keys():
        keywords = config.DEFECT_KEYWORDS[defect_type]
        has_defect = any(kw.lower() in text for kw in keywords)
        defect_features.append(1.0 if has_defect else 0.0)
    comp_features = []
    for comp_type in config.COMPONENT_KEYWORDS.keys():
        keywords = config.COMPONENT_KEYWORDS[comp_type]
        has_comp = any(kw.lower() in text for kw in keywords)
        comp_features.append(1.0 if has_comp else 0.0)
    combined = np.concatenate([char_features, defect_features, comp_features])
    if len(combined) == 0:
        combined = np.zeros(config.ATTENTION_CONFIG['max_seq_len'], dtype=np.float32)
    return combined.astype(np.float32)


def calc_attention_similarity(attr1, attr2):
    try:
        if pd.isna(attr1) or pd.isna(attr2):
            return 0.0
        if isinstance(attr1, str):
            try:
                attr1_dict = json.loads(attr1.replace("'", '"'))
            except:
                attr1_dict = {}
        else:
            attr1_dict = {}
        if isinstance(attr2, str):
            try:
                attr2_dict = json.loads(attr2.replace("'", '"'))
            except:
                attr2_dict = {}
        else:
            attr2_dict = {}
        feat1 = get_feature_vector(attr1_dict)
        feat2 = get_feature_vector(attr2_dict)
        with torch.no_grad():
            similarity = attention_model.calculate_attention_similarity(feat1, feat2)
        return float(similarity)
    except Exception:
        return 0.0


def calc_semantic_similarity(attr1, attr2):
    try:
        if pd.isna(attr1) or pd.isna(attr2):
            return 0.0
        if isinstance(attr1, str):
            try:
                attr1_dict = json.loads(attr1.replace("'", '"'))
            except:
                attr1_dict = {}
        else:
            attr1_dict = {}
        if isinstance(attr2, str):
            try:
                attr2_dict = json.loads(attr2.replace("'", '"'))
            except:
                attr2_dict = {}
        else:
            attr2_dict = {}
        emb1 = attr1_dict.get('embedding', [])
        emb2 = attr2_dict.get('embedding', [])
        if emb1 and emb2 and isinstance(emb1, list) and isinstance(emb2, list):
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)
            if len(emb1) > 0 and len(emb2) > 0:
                return max(0.0, np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
        text1 = str(attr1_dict.get('name', attr1_dict.get('text', '')))[:100]
        text2 = str(attr2_dict.get('name', attr2_dict.get('text', '')))[:100]
        return float(fuzz.ratio(text1.lower(), text2.lower()) / 100)
    except Exception:
        return 0.0


def calc_relation_similarity(rel_count1, rel_count2):
    if rel_count1 == 0 and rel_count2 == 0:
        return 0.5
    if rel_count1 == 0 or rel_count2 == 0:
        return 0.3
    return float(min(rel_count1, rel_count2) / max(rel_count1, rel_count2))


# ==================== 2. 加载数据 ====================
def load_image_data(img_node_path: str, img_rel_path: str):
    print(f"📂 加载图像数据...")
    img_df = pd.read_csv(img_node_path, encoding='utf-8')
    print(f"  ✅ 图像节点: {len(img_df)} 个")
    if not img_df.empty and '节点ID' in img_df.columns:
        img_df['节点ID'] = pd.to_numeric(img_df['节点ID'], errors='coerce').fillna(0).astype(int)

    defect_types = {}
    for _, row in img_df.iterrows():
        node_type = row.get('节点类型', '')
        if 'Defect' in str(node_type):
            match = re.search(r'Defect,\s*([^\]\s]+)', str(node_type))
            if match:
                defect_type = match.group(1)
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        if '缺陷类型' in img_df.columns:
            dt = row.get('缺陷类型', '')
            if dt and pd.notna(dt):
                defect_types[dt] = defect_types.get(dt, 0) + 1
    if defect_types:
        print(f"  📊 图像缺陷类型分布: {defect_types}")

    img_rel_df = None
    if Path(img_rel_path).exists():
        img_rel_df = pd.read_csv(img_rel_path, encoding='utf-8')
        print(f"  ✅ 图像关系: {len(img_rel_df)} 条")
        if '起始节点ID' in img_rel_df.columns:
            img_rel_df['起始节点ID'] = pd.to_numeric(img_rel_df['起始节点ID'], errors='coerce').fillna(0).astype(int)
        if '结束节点ID' in img_rel_df.columns:
            img_rel_df['结束节点ID'] = pd.to_numeric(img_rel_df['结束节点ID'], errors='coerce').fillna(0).astype(int)
    else:
        print(f"  ⚠️ 图像关系文件不存在: {img_rel_path}")
    return img_df, img_rel_df


def load_text_data(text_node_path: str, text_rel_path: str):
    print(f"📂 加载文本数据...")
    text_df = pd.read_csv(text_node_path, encoding='utf-8')
    text_rel_df = pd.read_csv(text_rel_path, encoding='utf-8')
    if not text_df.empty and '节点ID' in text_df.columns:
        text_df['节点ID'] = pd.to_numeric(text_df['节点ID'], errors='coerce').fillna(0).astype(int)
    print(f"  ✅ 文本节点: {len(text_df)} 个")
    print(f"  ✅ 文本关系: {len(text_rel_df)} 条")
    return text_df, text_rel_df


# ==================== 3. 粗筛（保持不变，只是阈值降低了） ====================
def coarse_filter(text_df, img_df):
    candidates = []
    id_col = '节点ID'
    type_col = '节点类型'
    name_col = '节点名称'

    total_pairs = len(text_df) * len(img_df)
    processed = 0

    print(f"  开始粗筛，共 {total_pairs} 个候选对...")

    for _, text_row in text_df.iterrows():
        text_id = text_row[id_col] if pd.notna(text_row[id_col]) else 0
        text_name = text_row[name_col] if pd.notna(text_row[name_col]) else ""
        text_type = text_row[type_col] if pd.notna(text_row[type_col]) else ""

        if not text_name:
            continue

        for _, img_row in img_df.iterrows():
            processed += 1
            if processed % 5000 == 0:
                print(f"    已处理 {processed}/{total_pairs} 个节点对...")

            img_id = img_row[id_col] if pd.notna(img_row[id_col]) else 0
            img_name = img_row[name_col] if pd.notna(img_row[name_col]) else ""
            img_type = img_row[type_col] if pd.notna(img_row[type_col]) else ""
            img_defect_type = img_row.get('缺陷类型', '')

            name_sim = enhanced_name_similarity(text_name, img_name, text_type, img_defect_type)
            type_sim = calc_type_similarity(text_type, img_type)

            if name_sim >= config.COARSE_THRESHOLD or type_sim >= config.COARSE_THRESHOLD:
                candidates.append({
                    'text_entity_id': text_id,
                    'img_entity_id': img_id,
                    'text_clean_name': clean_node_label(text_name),
                    'img_clean_name': clean_node_label(img_name),
                    'text_type': clean_node_label(text_type),
                    'img_type': clean_node_label(img_type),
                    'img_defect_type': img_defect_type,
                    'name_similarity': round(name_sim, 3),
                    'type_similarity': round(type_sim, 3)
                })

    unique_candidates = {}
    for cand in candidates:
        key = (cand['text_entity_id'], cand['img_entity_id'])
        if key not in unique_candidates:
            unique_candidates[key] = cand
        else:
            existing = unique_candidates[key]
            if cand['name_similarity'] + cand['type_similarity'] > existing['name_similarity'] + existing['type_similarity']:
                unique_candidates[key] = cand

    candidate_df = pd.DataFrame(list(unique_candidates.values()))
    print(f"  ✅ 粗筛完成: {len(candidate_df)} 个候选对")
    return candidate_df


# ==================== 4. 精筛（保持不变） ====================
def fuse_similarity(text_df, img_df, candidate_df, text_rel_count, img_rel_count):
    refined_results = []
    id_col = '节点ID'
    attr_col = '所有属性'

    text_dict = {row[id_col]: row for _, row in text_df.iterrows()}
    img_dict = {row[id_col]: row for _, row in img_df.iterrows()}

    print(f"  开始精筛，共 {len(candidate_df)} 个候选对...")

    for idx, cand in candidate_df.iterrows():
        if idx % 100 == 0 and idx > 0:
            print(f"    已处理 {idx}/{len(candidate_df)} 个候选对...")

        try:
            text_id = cand['text_entity_id']
            img_id = cand['img_entity_id']

            if text_id not in text_dict or img_id not in img_dict:
                continue

            text_row = text_dict[text_id]
            img_row = img_dict[img_id]

            name_sim = cand['name_similarity']
            type_sim = cand['type_similarity']
            semantic_sim = calc_semantic_similarity(text_row[attr_col], img_row[attr_col])
            rel_sim = calc_relation_similarity(text_rel_count.get(text_id, 0), img_rel_count.get(img_id, 0))
            attention_sim = calc_attention_similarity(text_row[attr_col], img_row[attr_col])

            total_score = float(
                config.WEIGHTS['name'] * name_sim +
                config.WEIGHTS['type'] * type_sim +
                config.WEIGHTS['semantic'] * semantic_sim +
                config.WEIGHTS['relation'] * rel_sim +
                config.WEIGHTS['attention'] * attention_sim
            )

            refined_results.append({
                '文本实体ID': text_id,
                '图像实体ID': img_id,
                '文本清理后名称': cand['text_clean_name'],
                '图像清理后名称': cand['img_clean_name'],
                '图像缺陷类型': cand.get('img_defect_type', ''),
                '文本类型': cand['text_type'],
                '图像类型': cand['img_type'],
                '名称相似度': round(name_sim, 3),
                '类型相似度': round(type_sim, 3),
                '语义相似度': round(semantic_sim, 3),
                '关系相似度': round(rel_sim, 3),
                '注意力相似度': round(attention_sim, 3),
                '融合总分': round(total_score, 3)
            })
        except Exception as e:
            continue

    refined_df = pd.DataFrame(refined_results)
    if not refined_df.empty:
        refined_df = refined_df.sort_values('融合总分', ascending=False)

    print(f"  ✅ 精筛完成: {len(refined_df)} 个对齐对")
    return refined_df


def get_relation_counts(rel_df, subject_col='起始节点ID', object_col='结束节点ID'):
    if rel_df is None or rel_df.empty:
        return {}
    counts = {}
    if subject_col in rel_df.columns:
        for node, count in rel_df[subject_col].value_counts().items():
            counts[node] = counts.get(node, 0) + count
    if object_col in rel_df.columns:
        for node, count in rel_df[object_col].value_counts().items():
            counts[node] = counts.get(node, 0) + count
    return counts


def align_decision(refined_df):
    if refined_df.empty:
        return {'对齐成功': pd.DataFrame(), '模糊案例': pd.DataFrame(), '对齐失败': pd.DataFrame()}

    aligned = refined_df[refined_df['融合总分'] >= config.SUCCESS_THRESHOLD].copy()
    fuzzy = refined_df[(refined_df['融合总分'] >= config.FUZZY_THRESHOLD) & (refined_df['融合总分'] < config.SUCCESS_THRESHOLD)].copy()
    unaligned = refined_df[refined_df['融合总分'] < config.FUZZY_THRESHOLD].copy()

    aligned = aligned.sort_values('融合总分', ascending=False).drop_duplicates('文本实体ID', keep='first')
    fuzzy = fuzzy.sort_values('融合总分', ascending=False).drop_duplicates('文本实体ID', keep='first')

    return {'对齐成功': aligned, '模糊案例': fuzzy, '对齐失败': unaligned}


def generate_fused_nodes(text_df, img_df, aligned_df, output_path):
    fused_nodes = []
    max_id = 0
    if not text_df.empty and '节点ID' in text_df.columns:
        text_max = text_df['节点ID'].max()
        if pd.notna(text_max):
            max_id = max(max_id, int(text_max))
    if not img_df.empty and '节点ID' in img_df.columns:
        img_max = img_df['节点ID'].max()
        if pd.notna(img_max):
            max_id = max(max_id, int(img_max))
    next_id = max_id + 1 if max_id > 0 else 1

    for _, row in aligned_df.iterrows():
        text_id = row['文本实体ID']
        img_id = row['图像实体ID']
        text_row = text_df[text_df['节点ID'] == text_id]
        if text_row.empty:
            continue
        text_row = text_row.iloc[0]
        fused_node = {
            '节点ID': next_id,
            '节点类型': f"[__Node__, Fused, {clean_node_label(text_row['节点类型'])}]",
            '节点名称': text_row['节点名称'],
            '所有属性': json.dumps({
                'original_text_id': int(text_id),
                'fused_image_id': int(img_id),
                'fusion_confidence': row['融合总分'],
                'image_defect_type': row.get('图像缺陷类型', ''),
                'original_name': text_row['节点名称'],
                'fused_image_name': row['图像清理后名称'],
                'attention_similarity': row.get('注意力相似度', 0.0)
            }, ensure_ascii=False)
        }
        fused_nodes.append(fused_node)
        next_id += 1

    fused_df = pd.DataFrame(fused_nodes)
    all_nodes = pd.concat([text_df, img_df, fused_df], ignore_index=True)
    all_nodes['节点ID'] = pd.to_numeric(all_nodes['节点ID'], errors='coerce').fillna(0).astype(int)
    all_nodes.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  ✅ 融合节点已保存: {output_path} ({len(all_nodes)} 个节点)")
    return fused_df


def generate_fused_relations(text_rel_df, img_rel_df, aligned_df, fused_nodes_df, output_path):
    print("\n🔗 生成融合关系文件...")
    all_rels = []
    next_rel_id = 1

    if text_rel_df is not None and not text_rel_df.empty:
        text_rels = text_rel_df.copy()
        text_rels['关系ID'] = range(next_rel_id, next_rel_id + len(text_rels))
        all_rels.append(text_rels)
        next_rel_id += len(text_rels)
        print(f"  📄 添加文本关系: {len(text_rels)} 条")

    if img_rel_df is not None and not img_rel_df.empty:
        img_rels = img_rel_df.copy()
        img_rels['关系ID'] = range(next_rel_id, next_rel_id + len(img_rels))
        all_rels.append(img_rels)
        next_rel_id += len(img_rels)
        print(f"  🖼️ 添加图像关系: {len(img_rels)} 条")

    if not aligned_df.empty and not fused_nodes_df.empty:
        fused_node_map = {}
        for _, row in fused_nodes_df.iterrows():
            try:
                attrs = json.loads(row['所有属性'].replace("'", '"'))
                if 'original_text_id' in attrs:
                    fused_node_map[attrs['original_text_id']] = row['节点ID']
            except:
                pass

        same_as_rels = []
        for _, row in aligned_df.iterrows():
            text_id = row['文本实体ID']
            img_id = row['图像实体ID']
            if text_id in fused_node_map:
                same_as_rels.append({
                    '关系ID': next_rel_id,
                    '关系类型': 'SAME_AS',
                    '起始节点ID': text_id,
                    '结束节点ID': fused_node_map[text_id],
                    '关系属性': json.dumps({'confidence': row['融合总分'], 'method': 'entity_alignment_with_attention', 'fused_image_id': img_id, 'attention_similarity': row.get('注意力相似度', 0.0)}, ensure_ascii=False)
                })
                next_rel_id += 1
                same_as_rels.append({
                    '关系ID': next_rel_id,
                    '关系类型': 'SAME_AS',
                    '起始节点ID': img_id,
                    '结束节点ID': fused_node_map[text_id],
                    '关系属性': json.dumps({'confidence': row['融合总分'], 'method': 'entity_alignment_with_attention', 'fused_text_id': text_id, 'attention_similarity': row.get('注意力相似度', 0.0)}, ensure_ascii=False)
                })
                next_rel_id += 1

        if same_as_rels:
            same_as_df = pd.DataFrame(same_as_rels)
            all_rels.append(same_as_df)
            print(f"  🔗 添加SAME_AS对齐关系: {len(same_as_rels)} 条")

    if all_rels:
        final_rel_df = pd.concat(all_rels, ignore_index=True)
        expected_cols = ['关系ID', '关系类型', '起始节点ID', '结束节点ID', '关系属性']
        for col in expected_cols:
            if col not in final_rel_df.columns:
                final_rel_df[col] = ''
        final_rel_df = final_rel_df[expected_cols]
        final_rel_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  ✅ 融合关系已保存: {output_path} ({len(final_rel_df)} 条关系)")
        return final_rel_df
    return pd.DataFrame()


# ==================== 5. 主函数 ====================
def main():
    print("=" * 70)
    print("🔗 多模态知识图谱融合（优化召回版）")
    print("=" * 70)

    print("\n📂 步骤1: 加载数据")
    text_df, text_rel_df = load_text_data('text_node.csv', 'text_rel.csv')
    img_df, img_rel_df = load_image_data('image_node.csv', 'image_rel.csv')

    if img_df.empty:
        print("\n❌ 图像数据为空")
        return

    print("\n📊 步骤2: 计算关系数量")
    text_rel_count = get_relation_counts(text_rel_df)
    img_rel_count = get_relation_counts(img_rel_df) if img_rel_df is not None else {}
    print(f"  ✅ 文本节点关系数: {len(text_rel_count)}")
    print(f"  ✅ 图像节点关系数: {len(img_rel_count)}")

    print("\n🔍 步骤3: 粗筛候选对")
    candidate_df = coarse_filter(text_df, img_df)

    if candidate_df.empty:
        print("\n⚠️ 没有候选对")
        return

    print("\n🎯 步骤4: 精筛融合")
    refined_df = fuse_similarity(text_df, img_df, candidate_df, text_rel_count, img_rel_count)

    if refined_df.empty:
        print("\n❌ 精筛后没有对齐对")
        return

    print("\n前10个高匹配度结果:")
    display_cols = ['文本实体ID', '图像实体ID', '文本清理后名称', '图像缺陷类型', '名称相似度', '注意力相似度', '融合总分']
    print(refined_df[display_cols].head(10).to_string())

    print("\n📊 步骤5: 决策对齐")
    result = align_decision(refined_df)

    aligned = result['对齐成功']
    fuzzy = result['模糊案例']
    unaligned = result['对齐失败']

    print(f"\n  ✅ 对齐成功（总分≥{config.SUCCESS_THRESHOLD}）: {len(aligned)} 个")
    print(f"  ⚠️ 模糊案例: {len(fuzzy)} 个")
    print(f"  ❌ 对齐失败: {len(unaligned)} 个")

    print("\n💾 步骤6: 保存结果")
    if len(aligned) > 0:
        aligned.to_excel('融合成功结果_带注意力.xlsx', index=False)
        print(f"  ✅ 成功结果: 融合成功结果_带注意力.xlsx ({len(aligned)} 条)")

    if len(fuzzy) > 0:
        fuzzy.to_excel('模糊案例_需审核_带注意力.xlsx', index=False)
        print(f"  ✅ 模糊案例: 模糊案例_需审核_带注意力.xlsx ({len(fuzzy)} 条)")

    fused_nodes_df = generate_fused_nodes(text_df, img_df, aligned, '融合后节点_带注意力.csv')
    fused_rels_df = generate_fused_relations(text_rel_df, img_rel_df, aligned, fused_nodes_df, '融合后关系_带注意力.csv')

    print("\n" + "=" * 70)
    print("📈 融合统计")
    print("=" * 70)
    print(f"  原始文本节点: {len(text_df)}")
    print(f"  原始图像节点: {len(img_df)}")
    print(f"  成功对齐: {len(aligned)}")
    print(f"  对齐覆盖率: {len(aligned)/len(text_df)*100:.1f}%")
    print(f"  融合后总节点: {len(text_df) + len(img_df) + len(aligned)}")
    print(f"  融合后总关系: {len(fused_rels_df) if not fused_rels_df.empty else 0}")

    print("\n" + "=" * 70)
    print("🎉 融合完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()