"""
Neo4j焊接缺陷SGG图数据库导入系统
将焊接缺陷SGG系统的JSON输出转换为Neo4j图数据库
关键设计：同一缺陷类型的所有图片都指向同一个缺陷类型节点
"""

import json
import os
from pathlib import Path
from neo4j import GraphDatabase

print("🔗 Neo4j焊接缺陷SGG图数据库导入系统")
print("=" * 60)

# ==================== 配置 ====================
class Neo4jConfig:
    """Neo4j数据库配置"""
    URI = "bolt://localhost:7687"  # Neo4j默认端口
    USER = "neo4j"                  # 默认用户名
    PASSWORD = "12345678"         # 你的密码

    # 标签定义（缺陷类型作为标签）
    LABELS = {
        'IMAGE': 'Image',
        'COMPONENT': 'Component',
        'DEFECT': 'Defect',

        # 构件标签
        'WELD_SEAM': 'WeldSeam',
        'WELD_SURFACE': 'WeldSurface',
        'WELD_EDGE': 'WeldEdge',
        'PARENT_METAL': 'ParentMetal',
        'WELD_JUNCTION': 'WeldJunction',

        # 缺陷标签
        'POROSITY': 'Porosity',
        'CRACK': 'Crack',
        'SPATTER': 'Spatter',
        'UNDERCUT': 'Undercut',
        'OVERLAP': 'Overlap',
        'PIT': 'Pit'
    }

    # 关系类型
    RELATION_TYPES = {
        'CONTAINS': 'CONTAINS',
        'ADJACENT_TO': 'ADJACENT_TO',
        'PART_OF': 'PART_OF',
        'LOCATED_ON': 'LOCATED_ON',
        'LOCATED_IN': 'LOCATED_IN',
        'ALONG_WITH': 'ALONG_WITH',
        'CROSSES': 'CROSSES',
        'SURROUNDS': 'SURROUNDS',
        'CONNECTED_TO': 'CONNECTED_TO',
        'PARALLEL_TO': 'PARALLEL_TO',
        'HAS_DEFECT': 'HAS_DEFECT',
        'HAS_COMPONENT': 'HAS_COMPONENT',
        'IS_TYPE_OF': 'IS_TYPE_OF'
    }

    # 焊接类别映射
    CLASS_MAPPING = {
        # 构件映射
        'weld_seam': 'WeldSeam',
        'weld_surface': 'WeldSurface',
        'weld_edge': 'WeldEdge',
        'parent_metal': 'ParentMetal',
        'weld_junction': 'WeldJunction',

        # 缺陷映射
        'porosity_region': 'Porosity',
        'crack_line': 'Crack',
        'spatter_cluster': 'Spatter',
        'undercut_groove': 'Undercut',
        'overlap_bump': 'Overlap',
        'pit_depression': 'Pit'
    }

config = Neo4jConfig()

# ==================== Neo4j连接器 ====================
class Neo4jConnector:
    """Neo4j连接器 - 修正版"""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

    def execute_write(self, query, parameters=None):
        with self.driver.session() as session:
            # 使用execute_write方法
            return session.execute_write(
                lambda tx: tx.run(query, parameters or {}).data()
            )

# ==================== 数据加载器 ====================
def load_sgg_data(json_dir):
    """加载所有SGG JSON文件"""
    print(f"📂 加载数据从: {json_dir}")

    json_dir = Path(json_dir)
    if not json_dir.exists():
        print(f"❌ 目录不存在: {json_dir}")
        return []

    json_files = list(json_dir.glob("*_sgg.json"))

    if not json_files:
        print("❌ 未找到JSON文件")
        return []

    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"⚠️  加载 {json_file} 时出错: {e}")

    print(f"✅ 成功加载 {len(all_data)} 个SGG文件")
    return all_data

# ==================== Neo4j导入器 ====================
class Neo4jImporter:
    """Neo4j数据导入器"""

    def __init__(self, connector):
        self.connector = connector

    def clear_database(self):
        """清空数据库"""
        print("🧹 清空数据库...")
        query = "MATCH (n) DETACH DELETE n"
        self.connector.execute_write(query)
        print("✅ 数据库已清空")

    def create_constraints(self):
        """创建约束"""
        print("📝 创建约束...")

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.image_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Porosity) REQUIRE p.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Crack) REQUIRE c.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Spatter) REQUIRE s.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:Undercut) REQUIRE u.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Overlap) REQUIRE o.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pit) REQUIRE p.defect_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ws:WeldSeam) REQUIRE ws.component_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (wsu:WeldSurface) REQUIRE wsu.component_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (we:WeldEdge) REQUIRE we.component_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pm:ParentMetal) REQUIRE pm.component_type IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (wj:WeldJunction) REQUIRE wj.component_type IS UNIQUE"
        ]

        for constraint in constraints:
            try:
                self.connector.execute_write(constraint)
            except Exception as e:
                print(f"⚠️  创建约束时出错: {e}")

        print("✅ 约束已创建")

    def import_all_data(self, sgg_data):
        """导入所有数据到Neo4j"""
        print("🚀 开始导入数据到Neo4j...")

        # 1. 创建缺陷类型节点（每个缺陷类型只有一个节点）
        self._create_defect_type_nodes()

        # 2. 创建构件类型节点（每个构件类型只有一个节点）
        self._create_component_type_nodes()

        # 3. 处理每个图像
        total_images = len(sgg_data)
        for idx, data in enumerate(sgg_data, 1):
            try:
                self._import_single_image(data)
                if idx % 10 == 0:
                    print(f"✅ 已导入 {idx}/{total_images} 个图像")
            except Exception as e:
                print(f"❌ 导入图像 {data.get('image', 'unknown')} 时出错: {e}")

        print(f"🎉 所有数据导入完成! 共处理 {total_images} 个图像")

    def _create_defect_type_nodes(self):
        """创建缺陷类型节点（每个类型只有一个）"""
        print("🏷️  创建缺陷类型节点...")

        defect_types = [
            ('Porosity', '气孔', '集中在焊道区域，形状较规则'),
            ('Crack', '裂纹', '多为直线段，少数曲折'),
            ('Spatter', '飞溅', '焊道两侧母材表面点状突起'),
            ('Undercut', '咬边', '母材与焊缝交界处的损伤'),
            ('Overlap', '重叠', '焊缝表面凸起明显的部分'),
            ('Pit', '凹坑', '形状不规则，焊道成型差时出现')
        ]

        for defect_type, chinese_name, description in defect_types:
            query = """
            MERGE (d:%s {
                defect_type: $defect_type,
                chinese_name: $chinese_name,
                description: $description
            })
            """ % defect_type

            self.connector.execute_write(query, {
                'defect_type': defect_type,
                'chinese_name': chinese_name,
                'description': description
            })

        print("✅ 缺陷类型节点已创建")

    def _create_component_type_nodes(self):
        """创建构件类型节点（每个类型只有一个）"""
        print("🏗️  创建构件类型节点...")

        component_types = [
            ('WeldSeam', '焊缝主体', '呈长条状，位于图像中部'),
            ('WeldSurface', '焊缝表面', '焊道成型部分'),
            ('WeldEdge', '焊缝边缘', '与母材交界处'),
            ('ParentMetal', '母材', '焊缝两侧的金属材料'),
            ('WeldJunction', '焊缝交界', '母材与焊缝过渡区域')
        ]

        for component_type, chinese_name, description in component_types:
            query = """
            MERGE (c:%s {
                component_type: $component_type,
                chinese_name: $chinese_name,
                description: $description
            })
            """ % component_type

            self.connector.execute_write(query, {
                'component_type': component_type,
                'chinese_name': chinese_name,
                'description': description
            })

        print("✅ 构件类型节点已创建")

    def _import_single_image(self, data):
        """导入单个图像的所有数据"""
        image_id = data.get('image', '')
        image_path = data.get('image_path', '')
        image_size = data.get('image_size', [0, 0])

        # 1. 创建图像节点
        self._create_image_node(image_id, image_path, image_size)

        # 2. 创建区域节点和关系
        regions = data.get('regions', [])
        for region in regions:
            self._create_region_node(image_id, region)

        # 3. 创建区域间关系
        relations = data.get('relations', [])
        for relation in relations:
            self._create_region_relation(image_id, relation)

    def _create_image_node(self, image_id, image_path, image_size):
        """创建图像节点"""
        query = """
        MERGE (i:Image {
            image_id: $image_id
        })
        SET i.image_path = $image_path,
            i.width = $width,
            i.height = $height,
            i.name = $image_name
        """

        self.connector.execute_write(query, {
            'image_id': image_id,
            'image_path': str(image_path).replace('\\', '/'),
            'width': image_size[0] if len(image_size) > 0 else 0,
            'height': image_size[1] if len(image_size) > 1 else 0,
            'image_name': Path(image_path).name
        })

    def _create_region_node(self, image_id, region):
        """创建区域节点"""
        region_id = region.get('id', 0)
        region_class = region.get('class', '')
        is_defect = region.get('is_defect', False)
        is_component = region.get('is_component', False)

        # 生成节点ID
        node_id = f"{image_id}_{region_id}"

        if is_defect:
            # 创建缺陷实例节点
            self._create_defect_instance(image_id, node_id, region)
        elif is_component:
            # 创建构件实例节点
            self._create_component_instance(image_id, node_id, region)

    def _create_defect_instance(self, image_id, node_id, region):
        """创建缺陷实例节点"""
        region_class = region.get('class', '')
        defect_label = config.CLASS_MAPPING.get(region_class, region_class)

        # 1. 创建缺陷实例节点
        query = """
        MERGE (d:Defect {
            defect_id: $defect_id
        })
        SET d.class = $class,
            d.bbox = $bbox,
            d.confidence = $confidence,
            d.image_id = $image_id,
            d.original_id = $original_id
        """

        self.connector.execute_write(query, {
            'defect_id': node_id,
            'class': region_class,
            'bbox': json.dumps(region.get('bbox', [])),
            'confidence': region.get('confidence', 0.0),
            'image_id': image_id,
            'original_id': region.get('id', 0)
        })

        # 2. 添加特定缺陷类型标签
        if defect_label in ['Porosity', 'Crack', 'Spatter', 'Undercut', 'Overlap', 'Pit']:
            query_add_label = """
            MATCH (d:Defect {defect_id: $defect_id})
            SET d:%s
            """ % defect_label

            self.connector.execute_write(query_add_label, {
                'defect_id': node_id
            })

        # 3. 创建图像-缺陷实例关系
        query_rel = """
        MATCH (i:Image {image_id: $image_id})
        MATCH (d:Defect {defect_id: $defect_id})
        MERGE (i)-[:HAS_DEFECT]->(d)
        """

        self.connector.execute_write(query_rel, {
            'image_id': image_id,
            'defect_id': node_id
        })

        # 4. 创建缺陷实例-缺陷类型关系（关键步骤：指向同一个缺陷类型节点）
        if defect_label in ['Porosity', 'Crack', 'Spatter', 'Undercut', 'Overlap', 'Pit']:
            query_type_rel = """
            MATCH (d:Defect {defect_id: $defect_id})
            MATCH (dt:%s {defect_type: $defect_type})
            MERGE (d)-[:IS_TYPE_OF]->(dt)
            """ % defect_label

            self.connector.execute_write(query_type_rel, {
                'defect_id': node_id,
                'defect_type': defect_label
            })

        # 5. 添加属性
        self._add_node_attributes('Defect', node_id, region)

    def _create_component_instance(self, image_id, node_id, region):
        """创建构件实例节点"""
        region_class = region.get('class', '')
        component_label = config.CLASS_MAPPING.get(region_class, region_class)

        # 1. 创建构件实例节点
        query = """
        MERGE (c:Component {
            component_id: $component_id
        })
        SET c.class = $class,
            c.bbox = $bbox,
            c.confidence = $confidence,
            c.image_id = $image_id,
            c.original_id = $original_id
        """

        self.connector.execute_write(query, {
            'component_id': node_id,
            'class': region_class,
            'bbox': json.dumps(region.get('bbox', [])),
            'confidence': region.get('confidence', 0.0),
            'image_id': image_id,
            'original_id': region.get('id', 0)
        })

        # 2. 添加特定构件类型标签
        if component_label in ['WeldSeam', 'WeldSurface', 'WeldEdge', 'ParentMetal', 'WeldJunction']:
            query_add_label = """
            MATCH (c:Component {component_id: $component_id})
            SET c:%s
            """ % component_label

            self.connector.execute_write(query_add_label, {
                'component_id': node_id
            })

        # 3. 创建图像-构件实例关系
        query_rel = """
        MATCH (i:Image {image_id: $image_id})
        MATCH (c:Component {component_id: $component_id})
        MERGE (i)-[:HAS_COMPONENT]->(c)
        """

        self.connector.execute_write(query_rel, {
            'image_id': image_id,
            'component_id': node_id
        })

        # 4. 创建构件实例-构件类型关系
        if component_label in ['WeldSeam', 'WeldSurface', 'WeldEdge', 'ParentMetal', 'WeldJunction']:
            query_type_rel = """
            MATCH (c:Component {component_id: $component_id})
            MATCH (ct:%s {component_type: $component_type})
            MERGE (c)-[:IS_TYPE_OF]->(ct)
            """ % component_label

            self.connector.execute_write(query_type_rel, {
                'component_id': node_id,
                'component_type': component_label
            })

        # 5. 添加属性
        self._add_node_attributes('Component', node_id, region)

    def _add_node_attributes(self, node_type, node_id, region):
        """添加节点属性"""
        exclude_fields = ['id', 'class', 'bbox', 'confidence', 'is_component', 'is_defect']

        for key, value in region.items():
            if key not in exclude_fields and value is not None:
                # 安全地设置属性
                query = f"""
                MATCH (n:{node_type} {{defect_id: $node_id}}) 
                SET n.{key} = $value
                """ if node_type == 'Defect' else f"""
                MATCH (n:{node_type} {{component_id: $node_id}}) 
                SET n.{key} = $value
                """

                try:
                    self.connector.execute_write(query, {
                        'node_id': node_id,
                        'value': str(value)
                    })
                except Exception as e:
                    # 如果属性名无效，使用带引号的属性名
                    safe_key = f"`{key}`"
                    query_safe = f"""
                    MATCH (n:{node_type} {{defect_id: $node_id}}) 
                    SET n.{safe_key} = $value
                    """ if node_type == 'Defect' else f"""
                    MATCH (n:{node_type} {{component_id: $node_id}}) 
                    SET n.{safe_key} = $value
                    """

                    try:
                        self.connector.execute_write(query_safe, {
                            'node_id': node_id,
                            'value': str(value)
                        })
                    except:
                        # 如果还是失败，跳过这个属性
                        continue

    def _create_region_relation(self, image_id, relation):
        """创建区域间关系"""
        sub_id = f"{image_id}_{relation.get('subject_id', 0)}"
        obj_id = f"{image_id}_{relation.get('object_id', 0)}"
        rel_type = relation.get('relation', '')
        confidence = relation.get('confidence', 0.0)

        # 映射关系类型
        rel_type_mapping = {
            'contains': 'CONTAINS',
            'adjacent_to': 'ADJACENT_TO',
            'part_of': 'PART_OF',
            'located_on': 'LOCATED_ON',
            'located_in': 'LOCATED_IN',
            'along_with': 'ALONG_WITH',
            'crosses': 'CROSSES',
            'surrounds': 'SURROUNDS',
            'connected_to': 'CONNECTED_TO',
            'parallel_to': 'PARALLEL_TO'
        }

        neo4j_rel_type = rel_type_mapping.get(rel_type, rel_type.upper())

        # 先尝试查找subject节点（可能是Defect或Component）
        query = """
        MATCH (subj)
        WHERE (subj:Defect AND subj.defect_id = $sub_id)
           OR (subj:Component AND subj.component_id = $sub_id)
        MATCH (obj)
        WHERE (obj:Defect AND obj.defect_id = $obj_id)
           OR (obj:Component AND obj.component_id = $obj_id)
        MERGE (subj)-[r:%s]->(obj)
        SET r.confidence = $confidence,
            r.image_id = $image_id,
            r.original_type = $original_type
        """ % neo4j_rel_type

        try:
            self.connector.execute_write(query, {
                'sub_id': sub_id,
                'obj_id': obj_id,
                'confidence': confidence,
                'image_id': image_id,
                'original_type': rel_type
            })
        except Exception as e:
            print(f"⚠️  创建关系时出错: {e}")

# ==================== 查询工具 ====================
class Neo4jQueryTool:
    """Neo4j查询工具"""

    def __init__(self, connector):
        self.connector = connector

    def show_statistics(self):
        """显示统计信息"""
        print("\n📊 图数据库统计信息")
        print("-" * 40)

        queries = [
            ("图像数量", "MATCH (i:Image) RETURN count(i) as count"),
            ("缺陷实例总数", "MATCH (d:Defect) RETURN count(d) as count"),
            ("构件实例总数", "MATCH (c:Component) RETURN count(c) as count"),
            ("气孔实例数", "MATCH (d:Porosity) RETURN count(d) as count"),
            ("裂纹实例数", "MATCH (d:Crack) RETURN count(d) as count"),
            ("飞溅实例数", "MATCH (d:Spatter) RETURN count(d) as count"),
            ("咬边实例数", "MATCH (d:Undercut) RETURN count(d) as count"),
            ("重叠实例数", "MATCH (d:Overlap) RETURN count(d) as count"),
            ("凹坑实例数", "MATCH (d:Pit) RETURN count(d) as count"),
            ("总关系数", "MATCH ()-[r]->() RETURN count(r) as count")
        ]

        for label, query in queries:
            try:
                result = self.connector.execute_query(query)
                if result:
                    print(f"  {label}: {result[0]['count']}")
            except Exception as e:
                print(f"  {label}: 查询错误 - {e}")

    def show_defect_type_connections(self):
        """显示缺陷类型连接情况"""
        print("\n🔗 缺陷类型连接情况")
        print("-" * 40)

        query = """
        MATCH (dt)<-[:IS_TYPE_OF]-(d:Defect)
        RETURN dt.defect_type as defect_type, count(d) as instance_count
        ORDER BY instance_count DESC
        """

        try:
            results = self.connector.execute_query(query)
            for result in results:
                print(f"  {result['defect_type']}: {result['instance_count']} 个实例指向该缺陷类型节点")
        except Exception as e:
            print(f"  ❌ 查询错误: {e}")

    def show_component_type_connections(self):
        """显示构件类型连接情况"""
        print("\n🔗 构件类型连接情况")
        print("-" * 40)

        query = """
        MATCH (ct)<-[:IS_TYPE_OF]-(c:Component)
        RETURN ct.component_type as component_type, count(c) as instance_count
        ORDER BY instance_count DESC
        """

        try:
            results = self.connector.execute_query(query)
            for result in results:
                print(f"  {result['component_type']}: {result['instance_count']} 个实例指向该构件类型节点")
        except Exception as e:
            print(f"  ❌ 查询错误: {e}")

    def find_images_with_defect(self, defect_type):
        """查找包含特定缺陷的图像"""
        print(f"\n🔍 查找包含 {defect_type} 的图像")
        print("-" * 40)

        query = """
        MATCH (i:Image)-[:HAS_DEFECT]->(d:Defect)-[:IS_TYPE_OF]->(dt:%s)
        RETURN i.image_id, i.name, count(d) as defect_count
        ORDER BY defect_count DESC
        LIMIT 10
        """ % defect_type

        try:
            results = self.connector.execute_query(query)
            if not results:
                print(f"  未找到包含 {defect_type} 的图像")
            else:
                for result in results:
                    print(f"  图像: {result['i.image_id']} ({result['i.name']}) - {result['defect_count']} 个{defect_type}")
        except Exception as e:
            print(f"  ❌ 查询错误: {e}")

    def show_common_relations(self):
        """显示常见关系"""
        print("\n🔄 常见关系类型")
        print("-" * 40)

        query = """
        MATCH ()-[r]->()
        RETURN type(r) as relation_type, count(r) as count
        ORDER BY count DESC
        LIMIT 10
        """

        try:
            results = self.connector.execute_query(query)
            for result in results:
                print(f"  {result['relation_type']}: {result['count']} 次")
        except Exception as e:
            print(f"  ❌ 查询错误: {e}")

    def find_defect_patterns(self):
        """查找缺陷模式"""
        print("\n🎯 缺陷-构件关系模式")
        print("-" * 40)

        query = """
        MATCH (d:Defect)-[r]->(c:Component)
        RETURN d.class as defect_class, type(r) as relation, c.class as component_class, count(r) as frequency
        ORDER BY frequency DESC
        LIMIT 10
        """

        try:
            results = self.connector.execute_query(query)
            for result in results:
                print(f"  {result['defect_class']} --[{result['relation']}]--> {result['component_class']}: {result['frequency']} 次")
        except Exception as e:
            print(f"  ❌ 查询错误: {e}")

# ==================== 主函数 ====================
def main():
    """主函数"""
    print("🔗 Neo4j焊接缺陷SGG图数据库导入系统")
    print("=" * 60)

    try:
        # 1. 初始化连接
        connector = Neo4jConnector(
            config.URI,
            config.USER,
            config.PASSWORD
        )
        print("✅ Neo4j连接成功")

        # 2. 加载数据
        sgg_data = load_sgg_data("dataset/processed")
        if not sgg_data:
            print("❌ 没有数据可导入")
            connector.close()
            return

        # 3. 创建导入器
        importer = Neo4jImporter(connector)

        # 4. 询问是否清空数据库
        clear_choice = input("是否清空数据库并重新创建？(y/N): ").strip().lower()
        if clear_choice == 'y':
            importer.clear_database()
            importer.create_constraints()
        else:
            print("⚠️  使用现有数据库，不会清空数据")

        # 5. 导入数据
        importer.import_all_data(sgg_data)

        # 6. 查询和分析
        query_tool = Neo4jQueryTool(connector)
        query_tool.show_statistics()
        query_tool.show_defect_type_connections()
        query_tool.show_component_type_connections()
        query_tool.show_common_relations()
        query_tool.find_defect_patterns()

        # 7. 示例查询
        print("\n🔍 示例查询:")
        print("-" * 40)

        # 查找包含气孔的图像
        query_tool.find_images_with_defect("Porosity")

        # 查找包含咬边的图像
        query_tool.find_images_with_defect("Undercut")

        # 8. 关闭连接
        connector.close()
        print("\n✅ Neo4j导入完成!")
        print("   可以打开Neo4j Browser查看图数据库:")
        print("   http://localhost:7474")

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        print("请检查:")
        print("  1. Neo4j数据库是否正在运行")
        print("  2. 用户名和密码是否正确")
        print("  3. 数据库URI是否正确")

# ==================== 执行 ====================
if __name__ == "__main__":
    main()