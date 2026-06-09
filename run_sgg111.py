"""
焊接缺陷SGG完整系统 - 处理所有文件并提取关系
"""

import json
import os
import sys
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import datetime

print("🚀 焊接缺陷SGG完整系统")
print("=" * 60)

# ==================== 配置 ====================
class WeldConfig:
    """焊接配置类"""
    # 构件区域（焊缝相关）
    COMPONENT_CLASSES = {
        'weld_seam': 0,          # 焊缝主体
        'weld_surface': 1,       # 焊缝表面
        'weld_edge': 2,          # 焊缝边缘
        'parent_metal': 3,       # 母材
        'weld_junction': 4       # 焊缝交界
    }

    # 缺陷区域
    DEFECT_CLASSES = {
        'porosity_region': 5,    # 气孔区域
        'crack_line': 6,         # 裂纹线段
        'spatter_cluster': 7,    # 飞溅聚集
        'undercut_groove': 8,    # 咬边沟槽
        'overlap_bump': 9,       # 重叠凸起
        'pit_depression': 10     # 凹坑区域
    }

    # 所有区域类别
    REGION_CLASSES = {**COMPONENT_CLASSES, **DEFECT_CLASSES}

    # 关系类别
    RELATION_CLASSES = {
        'contains': 0,          # 包含
        'adjacent_to': 1,       # 相邻
        'part_of': 2,           # 属于
        'located_on': 3,        # 位于...之上
        'located_in': 4,        # 位于...之内
        'along_with': 5,        # 沿着
        'crosses': 6,           # 穿过
        'surrounds': 7,         # 环绕
        'connected_to': 8,      # 连接到
        'parallel_to': 9        # 平行于
    }

    # 标签映射（您的标签 -> 系统标签）
    LABEL_MAPPING = {
        'porosity': 'porosity_region',
        'crack': 'crack_line',
        'spatter': 'spatter_cluster',
        'undercut': 'undercut_groove',
        'overlap': 'overlap_bump',
        'pit': 'pit_depression',
        'weld': 'weld_seam',
        'seam': 'weld_seam'
    }

    # 焊缝特征（用于识别焊缝位置）
    WELD_SEAM_PROPERTIES = {
        'min_aspect_ratio': 2.0,     # 最小长宽比（焊缝通常较长）
        'center_threshold': 0.3,     # 焊缝通常在图像中心区域
        'size_ratio': 0.5           # 焊缝通常占据图像较大面积
    }

config = WeldConfig()

# ==================== 数据类 ====================
class WeldRegion:
    """焊接区域"""
    def __init__(self, region_id: int, class_name: str, bbox: List[float], confidence: float = 0.95):
        self.id = region_id
        self.class_name = class_name
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.attributes = {}

        # 根据类别设置属性
        if class_name == 'porosity_region':
            area = self.area
            if area < 100:
                self.attributes['num_pores'] = 1
                self.attributes['density'] = 'low'
            elif area < 500:
                self.attributes['num_pores'] = 2
                self.attributes['density'] = 'medium'
            elif area < 1000:
                self.attributes['num_pores'] = 3
                self.attributes['density'] = 'high'
            else:
                self.attributes['num_pores'] = max(4, int(area / 250))
                self.attributes['density'] = 'very_high'

        elif class_name == 'spatter_cluster':
            self.attributes['num_spatters'] = max(3, int(self.area / 80))
            self.attributes['density'] = 'high' if self.area > 200 else 'medium'

        elif class_name == 'crack_line':
            self.attributes['length'] = max(self.width, self.height)
            self.attributes['orientation'] = 'horizontal' if self.width > self.height else 'vertical'
            self.attributes['type'] = 'longitudinal' if self.width > self.height else 'transverse'

        elif class_name == 'undercut_groove':
            self.attributes['depth_estimate'] = min(self.width, self.height) * 0.5
            self.attributes['location'] = 'edge' if self.width < self.height else 'surface'

        elif class_name == 'overlap_bump':
            self.attributes['height_estimate'] = min(self.width, self.height) * 0.8
            self.attributes['shape'] = 'protrusion'

        elif class_name == 'pit_depression':
            self.attributes['depth_estimate'] = min(self.width, self.height) * 0.3
            self.attributes['shape_regularity'] = 'irregular' if abs(self.width - self.height) > 5 else 'regular'

        elif class_name in ['weld_seam', 'weld_surface']:
            self.attributes['aspect_ratio'] = self.width / max(self.height, 1)
            self.attributes['position'] = 'center' if self.center_x_ratio > 0.3 and self.center_x_ratio < 0.7 else 'edge'

        elif class_name == 'weld_edge':
            self.attributes['sharpness'] = 'sharp' if min(self.width, self.height) < 10 else 'blunt'

        elif class_name == 'parent_metal':
            self.attributes['texture'] = 'smooth'
            self.attributes['position'] = 'left' if self.center_x_ratio < 0.3 else 'right' if self.center_x_ratio > 0.7 else 'center'

    @property
    def center(self) -> Tuple[float, float]:
        """区域中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def center_x_ratio(self) -> float:
        """中心点x坐标相对于图像宽度的比例"""
        x_center, _ = self.center
        return x_center / (self.bbox[2] - self.bbox[0]) if (self.bbox[2] - self.bbox[0]) > 0 else 0.5

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """长宽比"""
        return self.width / max(self.height, 1)

    @property
    def is_component(self) -> bool:
        """是否为构件区域"""
        return self.class_name in config.COMPONENT_CLASSES

    @property
    def is_defect(self) -> bool:
        """是否为缺陷区域"""
        return self.class_name in config.DEFECT_CLASSES

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            "id": self.id,
            "class": self.class_name,
            "bbox": [float(x) for x in self.bbox],
            "confidence": float(self.confidence),
            "is_component": self.is_component,
            "is_defect": self.is_defect
        }
        result.update(self.attributes)
        return result

class WeldRelation:
    """焊接关系"""
    def __init__(self, subject_id: int, object_id: int, relation: str, confidence: float = 0.8):
        self.subject_id = subject_id
        self.object_id = object_id
        self.relation = relation
        self.confidence = confidence

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "subject_id": self.subject_id,
            "object_id": self.object_id,
            "relation": self.relation,
            "confidence": float(self.confidence)
        }

class SceneGraph:
    """场景图"""
    def __init__(self, image_id: str, image_path: str, regions: List[WeldRegion],
                 relations: List[WeldRelation], image_size: Tuple[int, int]):
        self.image_id = image_id
        self.image_path = image_path
        self.regions = regions
        self.relations = relations
        self.image_size = image_size

        # 创建ID到区域的映射
        self.region_dict = {r.id: r for r in regions}

    def get_region_by_id(self, region_id: int) -> Optional[WeldRegion]:
        """根据ID获取区域"""
        return self.region_dict.get(region_id)

    def get_component_regions(self) -> List[WeldRegion]:
        """获取所有构件区域"""
        return [r for r in self.regions if r.is_component]

    def get_defect_regions(self) -> List[WeldRegion]:
        """获取所有缺陷区域"""
        return [r for r in self.regions if r.is_defect]

    def get_weld_seam(self) -> Optional[WeldRegion]:
        """获取焊缝主体区域"""
        for region in self.regions:
            if region.class_name == 'weld_seam':
                return region
        return None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "image": self.image_id,
            "image_path": self.image_path,
            "image_size": list(self.image_size),
            "regions": [r.to_dict() for r in self.regions],
            "relations": [r.to_dict() for r in self.relations]
        }

# ==================== 关系检测器 ====================
class RelationDetector:
    """关系检测器"""

    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """计算IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        return intersection / (area1 + area2 - intersection + 1e-8)

    @staticmethod
    def calculate_containment(subject_bbox: List[float], object_bbox: List[float]) -> float:
        """计算包含程度"""
        iou = RelationDetector.calculate_iou(subject_bbox, object_bbox)

        # 计算object在subject中的面积比例
        x1 = max(subject_bbox[0], object_bbox[0])
        y1 = max(subject_bbox[1], object_bbox[1])
        x2 = min(subject_bbox[2], object_bbox[2])
        y2 = min(subject_bbox[3], object_bbox[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        object_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])

        return intersection / (object_area + 1e-8)

    @staticmethod
    def calculate_adjacency(bbox1: List[float], bbox2: List[float], image_size: Tuple[int, int]) -> float:
        """计算相邻程度"""
        # 计算水平距离
        if bbox1[0] > bbox2[2]:  # bbox1在bbox2右边
            h_distance = bbox1[0] - bbox2[2]
        elif bbox2[0] > bbox1[2]:  # bbox2在bbox1右边
            h_distance = bbox2[0] - bbox1[2]
        else:
            h_distance = 0

        # 计算垂直距离
        if bbox1[1] > bbox2[3]:  # bbox1在bbox2下边
            v_distance = bbox1[1] - bbox2[3]
        elif bbox2[1] > bbox1[3]:  # bbox2在bbox1下边
            v_distance = bbox2[1] - bbox1[3]
        else:
            v_distance = 0

        min_distance = min(h_distance, v_distance) if h_distance > 0 and v_distance > 0 else max(h_distance, v_distance)

        # 归一化距离（距离越小，相邻程度越高）
        max_distance = max(image_size) * 0.1
        if min_distance > max_distance:
            return 0.0

        return 1.0 - (min_distance / max_distance)

    @staticmethod
    def detect_relations(regions: List[WeldRegion], image_size: Tuple[int, int]) -> List[WeldRelation]:
        """检测区域间的关系"""
        relations = []

        if len(regions) < 2:
            return relations

        # 分离构件和缺陷
        components = [r for r in regions if r.is_component]
        defects = [r for r in regions if r.is_defect]

        # 自动创建缺失的构件区域
        if not components and defects:
            components = RelationDetector._create_missing_components(defects, image_size)
            regions.extend(components)

        # 基于焊接知识的缺陷-构件关系规则
        defect_component_rules = {
            # 重叠overlap: 焊缝表面凸起明显的部分，多出现在焊道成型差的时候
            'overlap_bump': {
                'located_on': ['weld_surface', 'weld_seam'],
                'part_of': ['weld_surface'],
                'adjacent_to': ['weld_edge']
            },

            # 咬边undercut: 母材与焊缝交界处，属于母材损伤
            'undercut_groove': {
                'located_in': ['weld_junction'],
                'adjacent_to': ['parent_metal', 'weld_edge'],
                'part_of': ['weld_junction']
            },

            # 气孔porosity: 集中在焊道区域，形状较规则
            'porosity_region': {
                'located_in': ['weld_seam', 'weld_surface'],
                'part_of': ['weld_seam'],
                'adjacent_to': ['porosity_region']  # 气孔之间可能相邻
            },

            # 裂纹crack: 多为直线段，少数曲折
            'crack_line': {
                'along_with': ['weld_seam', 'weld_edge'],
                'crosses': ['weld_seam'],
                'located_in': ['weld_surface', 'weld_seam']
            },

            # 飞溅spatter: 主要集中在焊道两侧母材表面区域
            'spatter_cluster': {
                'located_on': ['parent_metal'],
                'adjacent_to': ['weld_edge'],
                'surrounds': ['weld_seam']  # 飞溅环绕焊缝
            },

            # 凹坑pit: 凹坑与气孔特征类似，但不如气孔形状规则
            'pit_depression': {
                'located_in': ['weld_surface', 'weld_seam'],
                'part_of': ['weld_surface'],
                'adjacent_to': ['overlap_bump', 'undercut_groove']  # 与其他缺陷相邻
            }
        }

        # 构件之间的关系
        component_component_rules = {
            'weld_seam': {
                'contains': ['weld_surface', 'weld_edge'],
                'adjacent_to': ['parent_metal'],
                'part_of': ['weld_surface']  # 焊缝主体属于焊缝表面的一部分
            },
            'weld_surface': {
                'located_on': ['weld_seam'],
                'adjacent_to': ['weld_edge', 'parent_metal']
            },
            'weld_edge': {
                'located_on': ['weld_seam'],
                'adjacent_to': ['parent_metal', 'weld_junction']
            },
            'parent_metal': {
                'adjacent_to': ['weld_edge', 'weld_junction'],
                'surrounds': ['weld_seam']
            },
            'weld_junction': {
                'located_in': ['parent_metal'],
                'adjacent_to': ['weld_edge']
            }
        }

        # 生成所有可能的区域对
        for i, sub_region in enumerate(regions):
            for j, obj_region in enumerate(regions):
                if i == j:  # 跳过自身
                    continue

                sub_class = sub_region.class_name
                obj_class = obj_region.class_name

                # 检查空间关系
                iou = RelationDetector.calculate_iou(sub_region.bbox, obj_region.bbox)
                containment = RelationDetector.calculate_containment(sub_region.bbox, obj_region.bbox)
                adjacency = RelationDetector.calculate_adjacency(sub_region.bbox, obj_region.bbox, image_size)

                # 基于空间关系推断
                relation_candidates = []

                if containment > 0.7:
                    if sub_region.area > obj_region.area:
                        relation_candidates.append(('contains', containment * 0.9))
                    else:
                        relation_candidates.append(('part_of', containment * 0.9))
                        relation_candidates.append(('located_in', containment * 0.8))

                if iou > 0.3 and iou < 0.7:
                    relation_candidates.append(('connected_to', iou * 0.7))

                if adjacency > 0.5:
                    relation_candidates.append(('adjacent_to', adjacency * 0.8))

                # 检查缺陷-构件关系规则
                if sub_class in defect_component_rules and obj_class in components:
                    for rel_type, allowed_classes in defect_component_rules[sub_class].items():
                        if obj_class in allowed_classes:
                            # 根据关系类型调整置信度
                            if rel_type in ['located_in', 'part_of'] and containment > 0.3:
                                relation_candidates.append((rel_type, containment * 0.9))
                            elif rel_type == 'adjacent_to' and adjacency > 0.3:
                                relation_candidates.append((rel_type, adjacency * 0.8))
                            elif rel_type in ['along_with', 'parallel_to']:
                                # 检查方向一致性
                                sub_orientation = 'horizontal' if sub_region.width > sub_region.height else 'vertical'
                                obj_orientation = 'horizontal' if obj_region.width > obj_region.height else 'vertical'
                                if sub_orientation == obj_orientation:
                                    orientation_score = 0.8
                                else:
                                    orientation_score = 0.3
                                relation_candidates.append((rel_type, orientation_score))
                            elif rel_type == 'crosses':
                                # 检查交叉关系
                                if iou > 0.1 and iou < 0.5:
                                    relation_candidates.append((rel_type, iou * 0.6))
                            elif rel_type == 'surrounds':
                                # 检查环绕关系
                                if containment < 0.3 and adjacency > 0.4:
                                    relation_candidates.append((rel_type, adjacency * 0.7))

                # 检查构件-构件关系规则
                if sub_class in component_component_rules and obj_class in components:
                    for rel_type, allowed_classes in component_component_rules[sub_class].items():
                        if obj_class in allowed_classes:
                            # 根据关系类型调整置信度
                            if rel_type in ['contains', 'part_of'] and containment > 0.5:
                                relation_candidates.append((rel_type, containment * 0.9))
                            elif rel_type == 'adjacent_to' and adjacency > 0.3:
                                relation_candidates.append((rel_type, adjacency * 0.8))
                            elif rel_type in ['located_on', 'located_in']:
                                relation_candidates.append((rel_type, 0.7))

                # 同类型区域之间的关系
                if sub_class == obj_class:
                    if sub_class == 'porosity_region' and adjacency > 0.4:
                        relation_candidates.append(('adjacent_to', adjacency * 0.7))
                    elif sub_class == 'spatter_cluster' and adjacency > 0.3:
                        relation_candidates.append(('connected_to', adjacency * 0.6))
                    elif sub_class == 'crack_line':
                        # 检查裂纹是否连接
                        if adjacency > 0.6:
                            relation_candidates.append(('connected_to', adjacency * 0.8))

                # 选择最佳关系
                if relation_candidates:
                    # 按置信度排序
                    relation_candidates.sort(key=lambda x: x[1], reverse=True)
                    best_relation, best_confidence = relation_candidates[0]

                    # 计算综合置信度
                    final_confidence = (best_confidence *
                                      sub_region.confidence *
                                      obj_region.confidence)

                    if final_confidence > 0.3:  # 阈值
                        relation = WeldRelation(
                            subject_id=sub_region.id,
                            object_id=obj_region.id,
                            relation=best_relation,
                            confidence=final_confidence
                        )
                        relations.append(relation)

        # 限制关系数量（避免太多）
        if len(relations) > 30:
            relations.sort(key=lambda x: x.confidence, reverse=True)
            relations = relations[:30]

        return relations

    @staticmethod
    def _create_missing_components(defects: List[WeldRegion], image_size: Tuple[int, int]) -> List[WeldRegion]:
        """创建缺失的构件区域"""
        components = []

        # 分析缺陷分布以推断焊缝位置
        weld_center_x = image_size[0] / 2
        weld_center_y = image_size[1] / 2

        # 如果有缺陷，根据缺陷位置调整焊缝位置
        if defects:
            defect_centers_x = [d.center[0] for d in defects]
            defect_centers_y = [d.center[1] for d in defects]
            weld_center_x = np.mean(defect_centers_x)
            weld_center_y = np.mean(defect_centers_y)

        # 创建焊缝主体（占据图像中部，呈长条状）
        weld_seam_width = image_size[0] * 0.6
        weld_seam_height = image_size[1] * 0.3
        weld_seam_x1 = weld_center_x - weld_seam_width / 2
        weld_seam_y1 = weld_center_y - weld_seam_height / 2
        weld_seam_x2 = weld_center_x + weld_seam_width / 2
        weld_seam_y2 = weld_center_y + weld_seam_height / 2

        # 边界检查
        weld_seam_x1 = max(0, weld_seam_x1)
        weld_seam_y1 = max(0, weld_seam_y1)
        weld_seam_x2 = min(image_size[0], weld_seam_x2)
        weld_seam_y2 = min(image_size[1], weld_seam_y2)

        weld_seam = WeldRegion(
            region_id=1001,
            class_name='weld_seam',
            bbox=[weld_seam_x1, weld_seam_y1, weld_seam_x2, weld_seam_y2],
            confidence=0.85
        )
        components.append(weld_seam)

        # 创建焊缝表面（比焊缝主体稍小）
        weld_surface_margin = min(weld_seam_width, weld_seam_height) * 0.1
        weld_surface = WeldRegion(
            region_id=1002,
            class_name='weld_surface',
            bbox=[
                weld_seam_x1 + weld_surface_margin,
                weld_seam_y1 + weld_surface_margin,
                weld_seam_x2 - weld_surface_margin,
                weld_seam_y2 - weld_surface_margin
            ],
            confidence=0.8
        )
        components.append(weld_surface)

        # 创建焊缝边缘（位于焊缝两侧）
        weld_edge_width = image_size[0] * 0.05
        weld_edge_left = WeldRegion(
            region_id=1003,
            class_name='weld_edge',
            bbox=[
                max(0, weld_seam_x1 - weld_edge_width),
                weld_seam_y1,
                weld_seam_x1,
                weld_seam_y2
            ],
            confidence=0.75
        )
        weld_edge_right = WeldRegion(
            region_id=1004,
            class_name='weld_edge',
            bbox=[
                weld_seam_x2,
                weld_seam_y1,
                min(image_size[0], weld_seam_x2 + weld_edge_width),
                weld_seam_y2
            ],
            confidence=0.75
        )
        components.append(weld_edge_left)
        components.append(weld_edge_right)

        # 创建母材（焊缝两侧的区域）
        parent_metal_left = WeldRegion(
            region_id=1005,
            class_name='parent_metal',
            bbox=[0, 0, weld_seam_x1, image_size[1]],
            confidence=0.9
        )
        parent_metal_right = WeldRegion(
            region_id=1006,
            class_name='parent_metal',
            bbox=[weld_seam_x2, 0, image_size[0], image_size[1]],
            confidence=0.9
        )
        components.append(parent_metal_left)
        components.append(parent_metal_right)

        # 创建焊缝交界（焊缝与母材交界处）
        weld_junction_left = WeldRegion(
            region_id=1007,
            class_name='weld_junction',
            bbox=[
                max(0, weld_seam_x1 - weld_edge_width),
                weld_seam_y1,
                weld_seam_x1,
                weld_seam_y2
            ],
            confidence=0.7
        )
        weld_junction_right = WeldRegion(
            region_id=1008,
            class_name='weld_junction',
            bbox=[
                weld_seam_x2,
                weld_seam_y1,
                min(image_size[0], weld_seam_x2 + weld_edge_width),
                weld_seam_y2
            ],
            confidence=0.7
        )
        components.append(weld_junction_left)
        components.append(weld_junction_right)

        return components

# ==================== 主处理函数 ====================
def process_all_files():
    """处理所有文件"""
    print("正在处理所有图像文件...")
    print("-" * 40)

    # 检查数据目录
    data_dir = Path("dataset/welding_images")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return []

    # 获取所有文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(list(data_dir.glob(f"*{ext}")))
        image_files.extend(list(data_dir.glob(f"*{ext.upper()}")))

    if not image_files:
        print("❌ 未找到图像文件")
        return []

    print(f"📸 找到 {len(image_files)} 个图像文件")

    # 创建输出目录
    output_dir = Path("dataset/processed")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_scene_graphs = []
    all_regions = []
    all_relations = []

    # 处理每个文件
    processed_count = 0
    error_count = 0

    for img_file in image_files:
        try:
            # 查找对应的JSON文件
            json_file = data_dir / f"{img_file.stem}.json"

            if not json_file.exists():
                print(f"⚠️  跳过 {img_file.name}: 未找到JSON文件")
                error_count += 1
                continue

            # 处理单个文件
            scene_graph = process_single_file(str(img_file), str(json_file))
            if scene_graph:
                all_scene_graphs.append(scene_graph)
                all_regions.extend(scene_graph.regions)
                all_relations.extend(scene_graph.relations)
                processed_count += 1

                # 每处理10个文件打印一次进度
                if processed_count % 10 == 0:
                    print(f"✅ 已处理 {processed_count}/{len(image_files)} 个文件")

                # 保存场景图
                output_file = output_dir / f"{scene_graph.image_id}_sgg.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(scene_graph.to_dict(), f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"❌ 处理 {img_file.name} 出错: {e}")
            error_count += 1
            continue

    print(f"\n📊 处理完成统计:")
    print(f"  ✅ 成功处理: {processed_count} 个文件")
    print(f"  ❌ 处理失败: {error_count} 个文件")
    print(f"  📍 总区域数: {len(all_regions)}")
    print(f"  🔗 总关系数: {len(all_relations)}")

    return all_scene_graphs

def process_single_file(image_path: str, json_path: str) -> Optional[SceneGraph]:
    """处理单个文件"""
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # 读取图像获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️  无法读取图像: {image_path}")
            return None

        h, w = img.shape[:2]

        # 解析区域
        regions = []
        region_id = 1

        for shape in annotation.get('shapes', []):
            label = shape.get('label', '').strip().lower()
            points = shape.get('points', [])

            # 标签映射
            if label in config.LABEL_MAPPING:
                class_name = config.LABEL_MAPPING[label]
            else:
                # 尝试模糊匹配
                for key, value in config.LABEL_MAPPING.items():
                    if key in label or label in key:
                        class_name = value
                        break
                else:
                    class_name = label  # 保持原标签

            if len(points) >= 2:
                # 转换为边界框
                x1, y1 = points[0]
                x2, y2 = points[1] if len(points) > 1 else points[0]

                # 确保坐标正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    # 创建区域
                    region = WeldRegion(
                        region_id=region_id,
                        class_name=class_name,
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=0.95
                    )
                    regions.append(region)
                    region_id += 1

        # 检测关系
        relations = RelationDetector.detect_relations(regions, (w, h))

        # 创建场景图
        image_id = Path(image_path).stem
        scene_graph = SceneGraph(
            image_id=image_id,
            image_path=image_path,
            regions=regions,
            relations=relations,
            image_size=(w, h)
        )

        return scene_graph

    except Exception as e:
        print(f"❌ 处理文件出错: {e}")
        return None

# ==================== 统计和分析 ====================
def analyze_results(scene_graphs: List[SceneGraph]):
    """分析结果"""
    print("\n📈 结果分析")
    print("-" * 40)

    if not scene_graphs:
        print("❌ 没有可分析的数据")
        return

    # 统计信息
    total_images = len(scene_graphs)
    total_regions = sum(len(sg.regions) for sg in scene_graphs)
    total_relations = sum(len(sg.relations) for sg in scene_graphs)

    # 区域分布
    region_distribution = {}
    component_count = 0
    defect_count = 0

    for sg in scene_graphs:
        for region in sg.regions:
            region_distribution[region.class_name] = region_distribution.get(region.class_name, 0) + 1
            if region.is_component:
                component_count += 1
            if region.is_defect:
                defect_count += 1

    # 关系分布
    relation_distribution = {}
    defect_component_relations = 0
    component_component_relations = 0
    defect_defect_relations = 0

    for sg in scene_graphs:
        for relation in sg.relations:
            relation_distribution[relation.relation] = relation_distribution.get(relation.relation, 0) + 1

            # 统计关系类型
            sub_region = sg.get_region_by_id(relation.subject_id)
            obj_region = sg.get_region_by_id(relation.object_id)

            if sub_region and obj_region:
                if sub_region.is_defect and obj_region.is_component:
                    defect_component_relations += 1
                elif sub_region.is_component and obj_region.is_component:
                    component_component_relations += 1
                elif sub_region.is_defect and obj_region.is_defect:
                    defect_defect_relations += 1

    # 常见关系模式
    pattern_counter = {}
    for sg in scene_graphs:
        for relation in sg.relations:
            sub_region = sg.get_region_by_id(relation.subject_id)
            obj_region = sg.get_region_by_id(relation.object_id)
            if sub_region and obj_region:
                pattern = f"{sub_region.class_name}--{relation.relation}--{obj_region.class_name}"
                pattern_counter[pattern] = pattern_counter.get(pattern, 0) + 1

    print(f"📊 总体统计:")
    print(f"  图像总数: {total_images}")
    print(f"  区域总数: {total_regions}")
    print(f"  构件区域: {component_count} ({component_count/total_regions*100:.1f}%)")
    print(f"  缺陷区域: {defect_count} ({defect_count/total_regions*100:.1f}%)")
    print(f"  关系总数: {total_relations}")
    print(f"  缺陷-构件关系: {defect_component_relations} ({defect_component_relations/total_relations*100:.1f}%)")
    print(f"  构件-构件关系: {component_component_relations} ({component_component_relations/total_relations*100:.1f}%)")
    print(f"  缺陷-缺陷关系: {defect_defect_relations} ({defect_defect_relations/total_relations*100:.1f}%)")

    print(f"\n📍 区域类别分布:")
    # 先显示构件
    print("  构件区域:")
    for class_name in config.COMPONENT_CLASSES:
        count = region_distribution.get(class_name, 0)
        if count > 0:
            percentage = count / total_regions * 100 if total_regions > 0 else 0
            print(f"    {class_name:<20}: {count:>4} ({percentage:>5.1f}%)")

    print("  缺陷区域:")
    for class_name in config.DEFECT_CLASSES:
        count = region_distribution.get(class_name, 0)
        if count > 0:
            percentage = count / total_regions * 100 if total_regions > 0 else 0
            print(f"    {class_name:<20}: {count:>4} ({percentage:>5.1f}%)")

    print(f"\n🔗 关系类别分布:")
    sorted_relations = sorted(relation_distribution.items(), key=lambda x: x[1], reverse=True)
    for relation, count in sorted_relations:
        percentage = count / total_relations * 100 if total_relations > 0 else 0
        print(f"  {relation:<15}: {count:>4} ({percentage:>5.1f}%)")

    print(f"\n🏆 最常见关系模式 (Top 15):")
    sorted_patterns = sorted(pattern_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    for pattern, count in sorted_patterns:
        print(f"  {pattern}")

    # 保存详细统计
    save_detailed_statistics(scene_graphs, region_distribution, relation_distribution, pattern_counter)

def save_detailed_statistics(scene_graphs: List[SceneGraph], region_dist: Dict, relation_dist: Dict, patterns: Dict):
    """保存详细统计"""
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)

    # 统计各类关系的数量
    defect_component_patterns = {}
    component_component_patterns = {}
    defect_defect_patterns = {}

    for pattern, count in patterns.items():
        parts = pattern.split('--')
        if len(parts) == 3:
            sub_class = parts[0]
            relation = parts[1]
            obj_class = parts[2]

            sub_is_defect = sub_class in config.DEFECT_CLASSES
            sub_is_component = sub_class in config.COMPONENT_CLASSES
            obj_is_defect = obj_class in config.DEFECT_CLASSES
            obj_is_component = obj_class in config.COMPONENT_CLASSES

            if sub_is_defect and obj_is_component:
                defect_component_patterns[pattern] = count
            elif sub_is_component and obj_is_component:
                component_component_patterns[pattern] = count
            elif sub_is_defect and obj_is_defect:
                defect_defect_patterns[pattern] = count

    # 保存JSON统计
    stats = {
        "total_images": len(scene_graphs),
        "total_regions": sum(len(sg.regions) for sg in scene_graphs),
        "total_relations": sum(len(sg.relations) for sg in scene_graphs),
        "region_distribution": region_dist,
        "relation_distribution": relation_dist,
        "common_patterns": dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:30]),
        "defect_component_patterns": dict(sorted(defect_component_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
        "component_component_patterns": dict(sorted(component_component_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
        "defect_defect_patterns": dict(sorted(defect_defect_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
        "scene_graphs_summary": [
            {
                "image": sg.image_id,
                "num_regions": len(sg.regions),
                "num_components": len([r for r in sg.regions if r.is_component]),
                "num_defects": len([r for r in sg.regions if r.is_defect]),
                "num_relations": len(sg.relations),
                "defect_types": list(set([r.class_name for r in sg.regions if r.is_defect]))
            }
            for sg in scene_graphs[:20]  # 只保存前20个的摘要信息
        ]
    }

    stats_file = output_dir / "detailed_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n💾 详细统计已保存: {stats_file}")

    # 保存文本报告
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("焊接缺陷SGG系统分析报告\n")
        f.write("=" * 60 + "\n\n")

        total_images = len(scene_graphs)
        total_regions = sum(len(sg.regions) for sg in scene_graphs)
        total_relations = sum(len(sg.relations) for sg in scene_graphs)

        f.write(f"处理图像总数: {total_images}\n")
        f.write(f"总区域数: {total_regions}\n")
        f.write(f"总关系数: {total_relations}\n\n")

        f.write("细粒度区域分布:\n")
        f.write("-" * 40 + "\n")
        # 构件区域
        f.write("构件区域:\n")
        for class_name in sorted(config.COMPONENT_CLASSES):
            count = region_dist.get(class_name, 0)
            if count > 0:
                percentage = count / total_regions * 100 if total_regions > 0 else 0
                f.write(f"  {class_name:<20}: {count:>6} ({percentage:>5.1f}%)\n")

        # 缺陷区域
        f.write("\n缺陷区域:\n")
        for class_name in sorted(config.DEFECT_CLASSES):
            count = region_dist.get(class_name, 0)
            if count > 0:
                percentage = count / total_regions * 100 if total_regions > 0 else 0
                f.write(f"  {class_name:<20}: {count:>6} ({percentage:>5.1f}%)\n")

        f.write("\n二元关系分布:\n")
        f.write("-" * 40 + "\n")
        for relation, count in sorted(relation_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_relations * 100 if total_relations > 0 else 0
            f.write(f"{relation:<15}: {count:>6} ({percentage:>5.1f}%)\n")

        f.write("\n缺陷-构件关系模式 (Top 20):\n")
        f.write("-" * 40 + "\n")
        sorted_dc_patterns = sorted(defect_component_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        for pattern, count in sorted_dc_patterns:
            f.write(f"{pattern}: {count}\n")

        f.write("\n构件-构件关系模式 (Top 20):\n")
        f.write("-" * 40 + "\n")
        sorted_cc_patterns = sorted(component_component_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        for pattern, count in sorted_cc_patterns:
            f.write(f"{pattern}: {count}\n")

    print(f"📝 分析报告已保存: {report_file}")


# ==================== 中文文字绘制辅助函数 ====================
def put_chinese_text(img, text, position, font_size=16, color=(255, 255, 255),
                     bg_color=None, font_path=None):
    """
    在OpenCV图像上绘制中文文字

    Args:
        img: OpenCV图像 (BGR格式)
        text: 要绘制的文字（支持中文）
        position: 文字位置 (x, y)
        font_size: 字体大小
        color: 文字颜色 (B, G, R)
        bg_color: 背景颜色，None表示无背景
        font_path: 字体路径，默认使用系统字体

    Returns:
        绘制后的OpenCV图像
    """
    # 转换 BGR -> RGB (PIL使用RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 尝试加载中文字体
    if font_path is None:
        # 常见系统中文字体路径
        font_candidates = [
            "C:/Windows/Fonts/simhei.ttf",  # Windows 黑体
            "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
            "/System/Library/Fonts/PingFang.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux
        ]
        for candidate in font_candidates:
            if os.path.exists(candidate):
                font_path = candidate
                break

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 计算文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position

    # 绘制背景
    if bg_color is not None:
        # PIL使用RGB，cv2的BGR需要转换
        bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0]) if len(bg_color) == 3 else bg_color
        draw.rectangle([x, y, x + text_width, y + text_height], fill=bg_color_rgb)

    # 绘制文字
    color_rgb = (color[2], color[1], color[0]) if len(color) == 3 else color
    draw.text((x, y), text, font=font, fill=color_rgb)

    # 转换回 BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def put_chinese_text_with_bg(img, text, position, font_size=14,
                             text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """绘制带背景的中文文字"""
    # 先计算文字尺寸（需要临时转换）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_path = None
    for candidate in ["C:/Windows/Fonts/simhei.ttf",
                      "C:/Windows/Fonts/msyh.ttc",
                      "/System/Library/Fonts/PingFang.ttc"]:
        if os.path.exists(candidate):
            font_path = candidate
            break

    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x, y = position

    # 绘制背景
    bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
    draw.rectangle([x - 2, y - 2, x + text_width + 2, y + text_height + 2],
                   fill=bg_color_rgb)

    # 绘制文字
    color_rgb = (text_color[2], text_color[1], text_color[0])
    draw.text((x, y), text, font=font, fill=color_rgb)

    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr, text_width, text_height


# ==================== 可视化 ====================
def generate_visualizations(scene_graphs: List[SceneGraph]):
    """为所有图像生成可视化（支持中文）"""
    print("\n🎨 生成可视化")
    print("-" * 40)

    if not scene_graphs:
        print("❌ 没有可可视化的数据")
        return

    output_dir = Path("inference_results2/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)

    # 颜色映射（BGR格式）
    colors = {
        # 构件区域
        'weld_seam': (0, 0, 255),  # 红色
        'weld_surface': (0, 165, 255),  # 橙色
        'weld_edge': (0, 255, 255),  # 黄色
        'parent_metal': (128, 128, 128),  # 灰色
        'weld_junction': (128, 0, 128),  # 紫色

        # 缺陷区域
        'porosity_region': (255, 0, 0),  # 蓝色
        'crack_line': (0, 255, 0),  # 绿色
        'spatter_cluster': (255, 255, 0),  # 青色
        'undercut_groove': (42, 42, 165),  # 棕色
        'overlap_bump': (203, 192, 255),  # 粉色
        'pit_depression': (0, 255, 255)  # 黄色
    }

    # 中文标签映射
    chinese_names = {
        'weld_seam': '焊缝主体',
        'weld_surface': '焊缝表面',
        'weld_edge': '焊缝边缘',
        'parent_metal': '母材',
        'weld_junction': '焊缝交界',
        'porosity_region': '气孔',
        'crack_line': '裂纹',
        'spatter_cluster': '飞溅',
        'undercut_groove': '咬边',
        'overlap_bump': '重叠',
        'pit_depression': '凹陷'
    }

    count = 0
    total_count = len(scene_graphs)

    for idx, sg in enumerate(scene_graphs):
        try:
            # 读取图像
            img = cv2.imread(sg.image_path)
            if img is None:
                print(f"⚠️  无法读取图像: {sg.image_path}")
                continue

            h, w = img.shape[:2]
            viz_img = img.copy()

            # ========== 绘制构件区域 ==========
            for region in sg.get_component_regions():
                color = colors.get(region.class_name, (255, 255, 255))
                x1, y1, x2, y2 = map(int, region.bbox)

                # 半透明填充
                overlay = viz_img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.15, viz_img, 0.85, 0, viz_img)

                # 边框
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)

                # 英文标签
                en_label = f"C{region.id}:{region.class_name.split('_')[0]}"
                cv2.putText(viz_img, en_label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ========== 绘制缺陷区域 ==========
            for region in sg.get_defect_regions():
                color = colors.get(region.class_name, (255, 255, 255))
                x1, y1, x2, y2 = map(int, region.bbox)

                # 边框（加粗）
                cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 3)

                # 中文标签（使用PIL绘制）
                ch_label = chinese_names.get(region.class_name, region.class_name)
                viz_img = put_chinese_text(
                    viz_img,
                    ch_label,
                    (x2 + 5, y1),
                    font_size=22,
                    color=(255, 255, 255),
                    bg_color=color
                )

            # ========== 绘制关系 ==========
            # 只绘制高置信度关系，避免画面混乱
            high_conf_relations = [r for r in sg.relations if r.confidence > 0.6]
            # 按置信度排序，只取前15条
            high_conf_relations.sort(key=lambda x: x.confidence, reverse=True)
            high_conf_relations = high_conf_relations[:15]

            for relation in high_conf_relations:
                sub_region = sg.get_region_by_id(relation.subject_id)
                obj_region = sg.get_region_by_id(relation.object_id)

                if sub_region and obj_region:
                    sub_center = tuple(map(int, sub_region.center))
                    obj_center = tuple(map(int, obj_region.center))

                    # 箭头颜色
                    if sub_region.is_defect and obj_region.is_component:
                        arrow_color = (255, 0, 255)  # 洋红
                        line_type = 2
                    elif sub_region.is_component and obj_region.is_component:
                        arrow_color = (0, 255, 0)  # 绿色
                        line_type = 1
                    else:
                        arrow_color = (255, 255, 0)  # 青色
                        line_type = 1

                    cv2.arrowedLine(viz_img, sub_center, obj_center,
                                    arrow_color, 2, tipLength=0.05)

                    # 关系标签（高置信度）
                    if relation.confidence > 0.75:
                        mid_x = (sub_center[0] + obj_center[0]) // 2
                        mid_y = (sub_center[1] + obj_center[1]) // 2

                        # 关系名中文映射
                        rel_ch_names = {
                            'contains': '包含', 'adjacent_to': '相邻',
                            'part_of': '属于', 'located_on': '位于',
                            'located_in': '位于', 'along_with': '沿着',
                            'crosses': '穿过', 'surrounds': '环绕',
                            'connected_to': '连接', 'parallel_to': '平行'
                        }
                        rel_text = rel_ch_names.get(relation.relation, relation.relation)

                        viz_img = put_chinese_text(
                            viz_img, rel_text, (mid_x - 10, mid_y - 10),
                            font_size=18, color=arrow_color, bg_color=(0, 0, 0)
                        )

            # ========== 绘制图例 ==========
            legend_x = 10
            legend_y = 30
            font_size = 12

            # 标题
            viz_img = put_chinese_text(
                viz_img, "【构件区域】", (legend_x, legend_y),
                font_size=20, color=(255, 255, 255), bg_color=(0, 0, 0)
            )
            legend_y += 22

            # 构件图例
            for class_name in ['weld_seam', 'weld_surface', 'weld_edge',
                               'parent_metal', 'weld_junction']:
                color = colors.get(class_name, (255, 255, 255))
                cv2.rectangle(viz_img, (legend_x, legend_y),
                              (legend_x + 15, legend_y + 12), color, -1)
                ch_name = chinese_names.get(class_name, class_name)
                viz_img = put_chinese_text(
                    viz_img, ch_name, (legend_x + 20, legend_y - 2),
                    font_size=16, color=(255, 255, 255), bg_color=(50, 50, 50)
                )
                legend_y += 18

            legend_y += 5
            viz_img = put_chinese_text(
                viz_img, "【缺陷区域】", (legend_x, legend_y),
                font_size=20, color=(255, 255, 255), bg_color=(0, 0, 0)
            )
            legend_y += 22

            # 缺陷图例
            for class_name in ['porosity_region', 'crack_line', 'spatter_cluster',
                               'undercut_groove', 'overlap_bump', 'pit_depression']:
                color = colors.get(class_name, (255, 255, 255))
                cv2.rectangle(viz_img, (legend_x, legend_y),
                              (legend_x + 15, legend_y + 12), color, -1)
                ch_name = chinese_names.get(class_name, class_name)
                viz_img = put_chinese_text(
                    viz_img, ch_name, (legend_x + 20, legend_y - 2),
                    font_size=16, color=(255, 255, 255), bg_color=(50, 50, 50)
                )
                legend_y += 18

            legend_y += 5
            viz_img = put_chinese_text(
                viz_img, "【关系类型】", (legend_x, legend_y),
                font_size=20, color=(255, 255, 255), bg_color=(0, 0, 0)
            )
            legend_y += 22

            cv2.line(viz_img, (legend_x, legend_y + 5), (legend_x + 20, legend_y + 5),
                     (255, 0, 255), 2)
            viz_img = put_chinese_text(
                viz_img, "缺陷→构件", (legend_x + 25, legend_y - 2),
                font_size=16, color=(255, 255, 255), bg_color=(50, 50, 50)
            )
            legend_y += 18

            cv2.line(viz_img, (legend_x, legend_y + 5), (legend_x + 20, legend_y + 5),
                     (0, 255, 0), 2)
            viz_img = put_chinese_text(
                viz_img, "构件→构件", (legend_x + 25, legend_y - 2),
                font_size=16, color=(255, 255, 255), bg_color=(50, 50, 50)
            )

            # ========== 图像信息 ==========
            info_text = f"图像: {sg.image_id} | 区域: {len(sg.regions)} | 关系: {len(sg.relations)}"
            # 英文部分用 cv2，中文用 PIL
            cv2.putText(viz_img, f"ID: {sg.image_id}", (w - 200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            viz_img = put_chinese_text(
                viz_img, f"区域数: {len(sg.regions)}", (w - 200, 45),
                font_size=18, color=(255, 255, 255), bg_color=(0, 0, 0)
            )
            viz_img = put_chinese_text(
                viz_img, f"关系数: {len(sg.relations)}", (w - 200, 65),
                font_size=18, color=(255, 255, 255), bg_color=(0, 0, 0)
            )

            # 保存
            viz_path = output_dir / f"{sg.image_id}_viz.png"
            cv2.imwrite(str(viz_path), viz_img)
            count += 1

            if (idx + 1) % 10 == 0:
                print(f"✅ 已生成 {idx + 1}/{total_count} 个可视化图像")

        except Exception as e:
            print(f"❌ 可视化 {sg.image_id} 出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n🎨 可视化生成完成!")
    print(f"  总共生成: {count} 个可视化图像")
    print(f"  保存位置: {output_dir}")

# ==================== 生成HTML报告 ====================
def generate_html_report(scene_graphs: List[SceneGraph]):
    """生成HTML报告"""
    print("\n📋 生成HTML报告")
    print("-" * 40)

    if not scene_graphs:
        print("❌ 没有数据生成报告")
        return

    # 统计信息
    total_images = len(scene_graphs)
    total_regions = sum(len(sg.regions) for sg in scene_graphs)
    total_relations = sum(len(sg.relations) for sg in scene_graphs)

    # 区域分布
    region_dist = {}
    for sg in scene_graphs:
        for region in sg.regions:
            region_dist[region.class_name] = region_dist.get(region.class_name, 0) + 1

    # 关系分布
    relation_dist = {}
    for sg in scene_graphs:
        for relation in sg.relations:
            relation_dist[relation.relation] = relation_dist.get(relation.relation, 0) + 1

    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 计算其他统计
    component_count = sum(1 for sg in scene_graphs for r in sg.regions if r.is_component)
    defect_count = sum(1 for sg in scene_graphs for r in sg.regions if r.is_defect)

    # 生成HTML
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>焊接缺陷SGG系统完整报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ background: white; padding: 25px; border-radius: 10px; 
                  box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                     gap: 20px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; 
                    border-left: 4px solid #667eea; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #667eea; margin: 10px 0; }}
        .stat-label {{ font-size: 16px; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; font-weight: bold; color: #333; }}
        tr:hover {{ background-color: #f9f9f9; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
                     gap: 20px; margin: 20px 0; }}
        .image-card {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; 
                     transition: transform 0.2s; }}
        .image-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .image-card img {{ width: 100%; height: 200px; object-fit: cover; }}
        .image-info {{ padding: 15px; }}
        .highlight {{ background-color: #e7f4ff; padding: 15px; border-radius: 6px; 
                    border-left: 4px solid #2196F3; margin: 20px 0; }}
        .tag {{ display: inline-block; background: #e0e0e0; padding: 4px 8px; 
               border-radius: 4px; margin: 2px; font-size: 12px; }}
        .component {{ color: #4CAF50; font-weight: bold; }}
        .defect {{ color: #F44336; font-weight: bold; }}
        .relation {{ color: #2196F3; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 焊接缺陷细粒度SGG系统完整报告</h1>
        <p>基于场景图生成的焊接缺陷检测与分析系统</p>
        <p>处理时间: {current_time}</p>
    </div>
    
    <div class="section">
        <h2>📊 系统概览</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_images}</div>
                <div class="stat-label">处理图像总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_regions}</div>
                <div class="stat-label">检测区域总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_relations}</div>
                <div class="stat-label">识别关系总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{component_count}</div>
                <div class="stat-label">构件区域数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{defect_count}</div>
                <div class="stat-label">缺陷区域数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_regions/total_images:.1f}</div>
                <div class="stat-label">平均每图区域数</div>
            </div>
        </div>
        
        <div class="highlight">
            <p><strong>🎯 系统特性:</strong></p>
            <ul>
                <li>✅ 处理了 <strong>{total_images}</strong> 张焊接图像</li>
                <li>✅ 检测到 <strong>{total_regions}</strong> 个细粒度区域</li>
                <li>✅ 识别出 <strong>{total_relations}</strong> 个二元关系</li>
                <li>✅ 支持 <strong>5</strong> 种焊接构件类型</li>
                <li>✅ 支持 <strong>6</strong> 种焊接缺陷类型</li>
                <li>✅ 支持 <strong>10</strong> 种空间/语义关系</li>
            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>📍 细粒度区域分布</h2>
        <table>
            <thead>
                <tr>
                    <th>区域类型</th>
                    <th>区域类别</th>
                    <th>数量</th>
                    <th>百分比</th>
                    <th>说明</th>
                </tr>
            </thead>
            <tbody>
'''

    # 添加构件区域行
    for class_name in config.COMPONENT_CLASSES:
        count = region_dist.get(class_name, 0)
        if count > 0:
            percentage = count / total_regions * 100 if total_regions > 0 else 0
            description = {
                'weld_seam': '焊缝主体，呈长条状，位于图像中部',
                'weld_surface': '焊缝表面，焊道成型部分',
                'weld_edge': '焊缝边缘，与母材交界处',
                'parent_metal': '母材，焊缝两侧的金属材料',
                'weld_junction': '焊缝交界，母材与焊缝过渡区域'
            }.get(class_name, class_name)

            html_content += f'''                <tr>
                    <td><span class="component">构件</span></td>
                    <td><strong>{class_name}</strong></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{description}</td>
                </tr>\n'''

    # 添加缺陷区域行
    for class_name in config.DEFECT_CLASSES:
        count = region_dist.get(class_name, 0)
        if count > 0:
            percentage = count / total_regions * 100 if total_regions > 0 else 0
            description = {
                'porosity_region': '气孔区域，集中在焊道区域，形状较规则',
                'crack_line': '裂纹线段，多为直线段，少数曲折',
                'spatter_cluster': '飞溅聚集，焊道两侧母材表面点状突起',
                'undercut_groove': '咬边沟槽，母材与焊缝交界处的损伤',
                'overlap_bump': '重叠凸起，焊缝表面凸起明显的部分',
                'pit_depression': '凹坑区域，形状不规则，焊道成型差时出现'
            }.get(class_name, class_name)

            html_content += f'''                <tr>
                    <td><span class="defect">缺陷</span></td>
                    <td><strong>{class_name}</strong></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{description}</td>
                </tr>\n'''

    html_content += '''            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>🔗 二元关系分布</h2>
        <table>
            <thead>
                <tr>
                    <th>关系类型</th>
                    <th>数量</th>
                    <th>百分比</th>
                    <th>说明</th>
                </tr>
            </thead>
            <tbody>
'''

    # 添加关系分布行
    sorted_relations = sorted(relation_dist.items(), key=lambda x: x[1], reverse=True)
    for relation, count in sorted_relations:
        percentage = count / total_relations * 100 if total_relations > 0 else 0
        description = {
            'contains': '包含关系，如焊缝包含气孔',
            'adjacent_to': '相邻关系，如咬边与焊缝边缘相邻',
            'part_of': '属于关系，如重叠凸起属于焊缝表面',
            'located_on': '位于...之上，如飞溅位于母材之上',
            'located_in': '位于...之内，如气孔位于焊缝之内',
            'along_with': '沿着，如裂纹沿着焊缝方向',
            'crosses': '穿过，如裂纹穿过焊缝',
            'surrounds': '环绕，如飞溅环绕焊缝',
            'connected_to': '连接到，如裂纹连接到咬边',
            'parallel_to': '平行于，如裂纹平行于焊缝边缘'
        }.get(relation, relation)

        html_content += f'''                <tr>
                    <td><span class="relation">{relation}</span></td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                    <td>{description}</td>
                </tr>\n'''

    html_content += '''            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>🖼️ 可视化图像</h2>
        <p>系统为所有图像生成了可视化，以下是部分示例：</p>
        <div class="image-grid">
'''

    # 添加可视化图像（随机选择12个）
    viz_files = list(Path("inference_results/visualizations").glob("*.png"))
    import random
    random.seed(42)  # 固定随机种子以获得可重复结果

    if viz_files:
        selected_files = random.sample(viz_files, min(12, len(viz_files)))
        for viz_file in selected_files:
            image_id = viz_file.stem.replace('_viz', '')
            html_content += f'''            <div class="image-card">
                <img src="visualizations/{viz_file.name}" alt="{image_id}">
                <div class="image-info">
                    <strong>{image_id}</strong><br>
                    <span class="tag">构件区域</span>
                    <span class="tag">缺陷检测</span>
                    <span class="tag">关系识别</span>
                </div>
            </div>\n'''

    html_content += f'''        </div>
        <p style="text-align: center; margin-top: 20px;">
            <strong>总计生成 {len(viz_files)} 个可视化图像</strong>
        </p>
    </div>
    
    <div class="section">
        <h2>🎯 基于焊接知识的规则</h2>
        <div class="highlight">
            <p><strong>缺陷-构件关系规则:</strong></p>
            <ol>
                <li><strong>重叠(overlap)</strong>: 位于焊缝表面，属于焊缝表面，与焊缝边缘相邻</li>
                <li><strong>咬边(undercut)</strong>: 位于焊缝交界处，与母材、焊缝边缘相邻，属于焊缝交界</li>
                <li><strong>气孔(porosity)</strong>: 位于焊缝主体/表面内，属于焊缝主体，气孔之间可能相邻</li>
                <li><strong>裂纹(crack)</strong>: 沿着焊缝主体/边缘，可能穿过焊缝，位于焊缝表面内</li>
                <li><strong>飞溅(spatter)</strong>: 位于母材之上，与焊缝边缘相邻，可能环绕焊缝</li>
                <li><strong>凹坑(pit)</strong>: 位于焊缝表面/主体内，属于焊缝表面，与重叠/咬边相邻</li>
            </ol>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 生成的文件</h2>
        <div class="highlight">
            <p><strong>主要输出文件:</strong></p>
            <ul>
                <li><strong>场景图JSON文件</strong>: dataset/processed/*_sgg.json (共{total_images}个)</li>
                <li><strong>可视化图像</strong>: inference_results/visualizations/*_viz.png (共{len(viz_files)}个)</li>
                <li><strong>详细统计</strong>: inference_results/detailed_statistics.json</li>
                <li><strong>分析报告</strong>: inference_results/analysis_report.txt</li>
                <li><strong>本HTML报告</strong>: inference_results/report.html</li>
            </ul>
        </div>
    </div>
    
   <div class="section" style="text-align: center; color: #666; font-size: 14px;">
        <hr>
        <p>焊接缺陷细粒度SGG系统 | 基于焊接知识的场景图生成</p>
        <p>生成时间: {current_time}</p>
    </div>
</body>
</html>'''

    # 保存HTML文件
    report_path = Path("inference_results") / "report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML报告已生成: {report_path}")
    print(f"   在浏览器中打开: file://{report_path.absolute()}")

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("🚀 焊接缺陷细粒度SGG完整系统")
    print("=" * 60)
    print("基于焊接知识的缺陷-构件关系识别")
    print("-" * 60)

    # 1. 处理所有文件
    scene_graphs = process_all_files()

    if not scene_graphs:
        print("❌ 没有成功处理任何文件")
        return

    print("\n" + "=" * 60)
    print("✅ 所有文件处理完成!")
    print("=" * 60)

    # 2. 分析结果
    analyze_results(scene_graphs)

    # 3. 生成可视化（为所有图像）
    generate_visualizations(scene_graphs)

    # 4. 生成HTML报告
    generate_html_report(scene_graphs)

    # 5. 显示最终统计
    print("\n" + "=" * 60)
    print("🎉 系统运行完成!")
    print("=" * 60)

    total_images = len(scene_graphs)
    total_regions = sum(len(sg.regions) for sg in scene_graphs)
    total_relations = sum(len(sg.relations) for sg in scene_graphs)
    component_count = sum(1 for sg in scene_graphs for r in sg.regions if r.is_component)
    defect_count = sum(1 for sg in scene_graphs for r in sg.regions if r.is_defect)

    print(f"\n📊 最终统计:")
    print(f"  📸 处理图像: {total_images} 张")
    print(f"  📍 检测区域: {total_regions} 个")
    print(f"    ├─ 构件区域: {component_count} 个 ({component_count/total_regions*100:.1f}%)")
    print(f"    └─ 缺陷区域: {defect_count} 个 ({defect_count/total_regions*100:.1f}%)")
    print(f"  🔗 识别关系: {total_relations} 个")
    print(f"  📈 平均每图区域: {total_regions/total_images:.1f}")
    print(f"  📈 平均每图关系: {total_relations/total_images:.1f}")

    print(f"\n📁 生成的文件:")
    print(f"  ├── dataset/processed/              # {total_images}个场景图JSON")
    print(f"  ├── inference_results/              # 分析结果")
    print(f"  │   ├── visualizations/            # {total_images}个可视化图像")
    print(f"  │   ├── detailed_statistics.json   # 详细统计")
    print(f"  │   ├── analysis_report.txt        # 分析报告")
    print(f"  │   └── report.html                # HTML报告")
    print(f"  └── 系统运行完成!")

    # 显示一些示例关系
    if scene_graphs and scene_graphs[0].relations:
        print(f"\n🔍 示例关系 (第一个图像):")
        sg = scene_graphs[0]

        # 缺陷->构件关系
        defect_component_rels = []
        # 构件->构件关系
        component_component_rels = []
        # 缺陷->缺陷关系
        defect_defect_rels = []

        for relation in sg.relations[:10]:  # 只显示前10个
            sub_region = sg.get_region_by_id(relation.subject_id)
            obj_region = sg.get_region_by_id(relation.object_id)
            if sub_region and obj_region:
                rel_str = f"{sub_region.class_name} --[{relation.relation}]--> {obj_region.class_name}"

                if sub_region.is_defect and obj_region.is_component:
                    defect_component_rels.append(rel_str)
                elif sub_region.is_component and obj_region.is_component:
                    component_component_rels.append(rel_str)
                elif sub_region.is_defect and obj_region.is_defect:
                    defect_defect_rels.append(rel_str)

        if defect_component_rels:
            print(f"  缺陷->构件关系:")
            for i, rel in enumerate(defect_component_rels[:3]):
                print(f"    {i+1}. {rel}")

        if component_component_rels:
            print(f"  构件->构件关系:")
            for i, rel in enumerate(component_component_rels[:3]):
                print(f"    {i+1}. {rel}")

        if defect_defect_rels:
            print(f"  缺陷->缺陷关系:")
            for i, rel in enumerate(defect_defect_rels[:3]):
                print(f"    {i+1}. {rel}")

if __name__ == "__main__":
    main()