import os
from neo4j import GraphDatabase
import csv
import ast
import re
import json
from collections import defaultdict

# -----------------------------------------------------------
# 🛠️ 配置区域
# -----------------------------------------------------------
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
NEO4J_URL = "bolt://localhost:7687"

current_dir = os.path.dirname(os.path.abspath(__file__))
NODES_CSV_PATH = os.path.join(current_dir, "融合后节点1.csv")
RELATIONS_CSV_PATH = os.path.join(current_dir, "融合后关系_带注意力.csv")


def safe_parse_properties(prop_str):
    """安全解析属性字符串"""
    if not prop_str or prop_str == "{}":
        return {}

    if isinstance(prop_str, dict):
        return prop_str

    try:
        return json.loads(prop_str)
    except:
        pass

    try:
        fixed_str = prop_str
        fixed_str = re.sub(r'(\w+):', r'"\1":', fixed_str)
        fixed_str = fixed_str.replace('\\', '\\\\')
        return ast.literal_eval(fixed_str)
    except:
        pass

    result = {}
    try:
        id_match = re.search(r'"id":\s*"([^"]+)"', prop_str)
        if id_match:
            result["id"] = id_match.group(1)

        name_match = re.search(r'"name":\s*"([^"]+)"', prop_str)
        if name_match:
            result["name"] = name_match.group(1)

        imageid_match = re.search(r'"imageid":\s*"([^"]+)"', prop_str)
        if imageid_match:
            result["imageid"] = imageid_match.group(1)

    except:
        pass

    return result


def parse_node_type(node_type_str):
    """解析节点类型列表"""
    try:
        if not node_type_str:
            return ["__Node__"]

        parts = re.findall(r'([^[\]",\s]+)', node_type_str)
        if parts:
            return list(set(parts))
        return ["__Node__"]
    except:
        return ["__Node__"]


def read_and_merge_nodes():
    """
    读取节点数据，按节点名称合并
    """
    print(f"📂 正在读取节点文件：{NODES_CSV_PATH}")

    # 按节点名称聚合
    merged_nodes = {}  # key=节点名称, value={labels, properties, imageids}
    image_nodes = defaultdict(set)  # key=imageid, value=节点名称集合

    try:
        with open(NODES_CSV_PATH, 'r', encoding='gbk') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                node_id = row.get("节点ID", "").strip()
                node_type_str = row.get("节点类型", "")
                node_name = row.get("节点名称", "").strip()
                prop_str = row.get("所有属性", "{}")

                if not node_name:  # 使用节点名称作为聚合键
                    # 如果没有节点名称，尝试从属性中获取
                    properties = safe_parse_properties(prop_str)
                    node_name = properties.get("name", "")
                    if not node_name:
                        print(f"⚠️ 跳过缺少节点名称的行 (ID: {node_id})")
                        continue

                # 解析属性
                properties = safe_parse_properties(prop_str)

                # 提取imageid
                imageid = properties.get("imageid", "")

                # 按节点名称聚合
                if node_name not in merged_nodes:
                    # 解析标签
                    labels = parse_node_type(node_type_str)

                    merged_nodes[node_name] = {
                        "labels": labels,
                        "properties": {
                            "name": node_name,
                            "triplet_source_id": properties.get("triplet_source_id", "")
                        },
                        "imageids": set()
                    }

                # 收集该节点关联的imageid
                if imageid:
                    merged_nodes[node_name]["imageids"].add(imageid)
                    image_nodes[imageid].add(node_name)

        print(f"✅ 节点合并完成：{len(merged_nodes)} 个唯一节点名称")
        print(f"✅ 发现 {len(image_nodes)} 个唯一图片节点")

        return merged_nodes, image_nodes

    except FileNotFoundError:
        print(f"❌ 找不到文件：{NODES_CSV_PATH}")
        return None, None
    except Exception as e:
        print(f"❌ 读取文件出错: {e}")
        return None, None


def read_and_merge_relations(merged_nodes):
    """
    读取关系数据，去重并确保每个节点对之间只有一条关系
    """
    print(f"\n📂 正在读取关系文件：{RELATIONS_CSV_PATH}")

    # 需要建立原始节点ID到节点名称的映射
    # 先读取所有节点，建立ID到名称的映射
    id_to_name = {}
    try:
        with open(NODES_CSV_PATH, 'r', encoding='gbk') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                node_id = row.get("节点ID", "").strip()
                node_name = row.get("节点名称", "").strip()
                if node_id and node_name:
                    id_to_name[node_id] = node_name
                elif node_id:
                    # 如果没有节点名称，从属性中获取
                    prop_str = row.get("所有属性", "{}")
                    properties = safe_parse_properties(prop_str)
                    name = properties.get("name", "")
                    if name:
                        id_to_name[node_id] = name
    except Exception as e:
        print(f"⚠️ 建立ID到名称映射时出错: {e}")

    # 存储去重后的关系
    unique_relations = {}

    try:
        with open(RELATIONS_CSV_PATH, 'r', encoding='utf-8-sig') as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                start_id = row.get("起始节点ID", "").strip()
                end_id = row.get("结束节点ID", "").strip()
                rel_type = row.get("关系类型", "").strip()

                if not start_id or not end_id or not rel_type:
                    continue

                # 将节点ID映射到节点名称
                start_name = id_to_name.get(start_id, start_id)
                end_name = id_to_name.get(end_id, end_id)

                # 跳过自环关系
                if start_name == end_name:
                    continue

                # 创建关系的唯一键
                relation_key = (start_name, end_name, rel_type)

                # 只保留第一条关系（去重）
                if relation_key not in unique_relations:
                    unique_relations[relation_key] = {
                        "start": start_name,
                        "end": end_name,
                        "type": rel_type
                    }

        print(f"✅ 关系去重完成：得到 {len(unique_relations)} 条唯一关系")
        return list(unique_relations.values())

    except FileNotFoundError:
        print(f"❌ 找不到文件：{RELATIONS_CSV_PATH}")
        return []
    except Exception as e:
        print(f"❌ 读取关系文件出错: {e}")
        return []


def import_nodes(driver, merged_nodes):
    """导入合并后的节点"""
    print(f"\n🚀 开始导入节点...")

    with driver.session() as session:
        total = len(merged_nodes)
        success = 0

        for node_name, node_info in merged_nodes.items():
            try:
                labels = node_info["labels"]
                properties = node_info["properties"]

                # 构建标签字符串
                if labels and labels != ["__Node__"]:
                    label_str = ":".join([f"`{label}`" for label in labels])
                else:
                    label_str = "__Node__"

                # 创建节点，使用节点名称作为唯一标识
                cypher = f"""
                    MERGE (n:{label_str} {{name: $node_name}})
                    SET n.triplet_source_id = $triplet_source_id
                """

                session.run(cypher,
                            node_name=node_name,
                            triplet_source_id=properties.get("triplet_source_id", ""))
                success += 1

                if success % 100 == 0:
                    print(f"✅ 已导入 {success}/{total} 个节点")

            except Exception as e:
                print(f"❌ 节点 {node_name} 导入失败: {e}")

        print(f"\n📊 节点导入完成！成功导入 {success}/{total} 个节点")


def import_image_nodes(driver, image_nodes):
    """导入图片节点，并建立属于关系"""
    print(f"\n🖼️ 开始导入图片节点和关系...")

    with driver.session() as session:
        total_images = len(image_nodes)
        success_images = 0
        success_relations = 0

        for imageid, defect_names in image_nodes.items():
            try:
                # 创建图片节点
                cypher_image = """
                    MERGE (img:Image {id: $imageid})
                    SET img.name = $imageid
                """
                session.run(cypher_image, imageid=imageid)
                success_images += 1

                # 为每个缺陷节点建立"属于"关系
                for defect_name in defect_names:
                    try:
                        cypher_rel = """
                            MATCH (img:Image {id: $imageid})
                            MATCH (defect {name: $defect_name})
                            MERGE (img)-[:属于]->(defect)
                        """
                        session.run(cypher_rel, imageid=imageid, defect_name=defect_name)
                        success_relations += 1
                    except Exception as e:
                        print(f"❌ 建立关系失败 {imageid} -> {defect_name}: {e}")

                if success_images % 50 == 0:
                    print(f"✅ 已导入 {success_images}/{total_images} 个图片节点")

            except Exception as e:
                print(f"❌ 图片节点 {imageid} 导入失败: {e}")

        print(f"\n📊 图片节点导入完成！成功导入 {success_images} 个图片节点，{success_relations} 条属于关系")


def import_relations(driver, relations):
    """导入去重后的关系"""
    print(f"\n🔗 开始导入去重后的关系...")

    if not relations:
        print("⚠️ 没有需要导入的关系")
        return

    with driver.session() as session:
        total = len(relations)
        success = 0
        failed = 0

        for relation in relations:
            try:
                # 使用MERGE确保关系唯一性
                cypher = f"""
                    MATCH (a {{name: $start_name}})
                    MATCH (b {{name: $end_name}})
                    MERGE (a)-[r:`{relation['type']}`]->(b)
                """

                session.run(cypher,
                            start_name=relation['start'],
                            end_name=relation['end'])
                success += 1

                if success % 100 == 0:
                    print(f"✅ 已导入 {success}/{total} 条关系")

            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"❌ 关系导入失败 {relation['start']} -[{relation['type']}]-> {relation['end']}: {e}")

        print(f"\n📊 关系导入完成！成功导入 {success}/{total} 条关系，失败 {failed} 条")


def create_indexes(driver):
    """创建索引以提升查询性能"""
    print("\n🔧 创建索引...")

    with driver.session() as session:
        try:
            # 为节点名称创建索引
            session.run("CREATE INDEX node_name_index IF NOT EXISTS FOR (n) ON (n.name)")
            print("✅ 创建节点名称索引")

            # 为图片节点创建特定索引
            session.run("CREATE INDEX image_id_index IF NOT EXISTS FOR (n:Image) ON (n.id)")
            print("✅ 创建图片节点ID索引")

        except Exception as e:
            print(f"⚠️ 索引创建失败: {e}")


def verify_import(driver):
    """验证导入结果"""
    print("\n🔍 验证导入结果...")

    with driver.session() as session:
        try:
            # 统计所有节点
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"📊 节点总数: {node_count}")

            # 统计图片节点
            result = session.run("MATCH (n:Image) RETURN count(n) as count")
            image_count = result.single()["count"]
            print(f"📊 图片节点数: {image_count}")

            # 统计所有关系
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            print(f"📊 关系总数: {rel_count}")

            # 统计属于关系
            result = session.run("MATCH ()-[r:属于]->() RETURN count(r) as count")
            belong_count = result.single()["count"]
            print(f"📊 属于关系数: {belong_count}")

            # 显示节点名称示例
            result = session.run("MATCH (n) RETURN n.name as name, labels(n) as labels LIMIT 10")
            print("\n📊 节点示例（前10个）:")
            for record in result:
                print(f"  {record['name']} - {record['labels']}")

        except Exception as e:
            print(f"⚠️ 验证失败: {e}")


# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 知识图谱导入 - 节点名称合并版本")
    print("=" * 60)

    # 1. 读取并合并节点数据
    merged_nodes, image_nodes = read_and_merge_nodes()

    if merged_nodes is None:
        print("\n❌ 节点数据读取失败，程序退出")
        exit(1)

    # 2. 读取并去重关系数据
    relations = read_and_merge_relations(merged_nodes)

    # 3. 连接数据库
    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
        driver.verify_connectivity()
        print(f"✅ 成功连接到 Neo4j")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        exit(1)

    # 4. 导入节点
    import_nodes(driver, merged_nodes)

    # 5. 导入图片节点和属于关系
    import_image_nodes(driver, image_nodes)

    # 6. 创建索引
    create_indexes(driver)

    # 7. 导入去重后的关系
    import_relations(driver, relations)

    # 8. 验证结果
    verify_import(driver)

    # 9. 关闭连接
    driver.close()
    print("\n✅ 所有操作完成！数据库连接已关闭")