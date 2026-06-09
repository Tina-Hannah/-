import os
from neo4j import GraphDatabase
import csv
import ast
import re

# -----------------------------------------------------------
# 🛠️ 配置区域 (保持不变)
# -----------------------------------------------------------
NEO4J_USER = "neo4j"
NEO4J_PASS = "12345678"
NEO4J_URL = "bolt://localhost:7687"

# 获取当前脚本路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# ⭐ 修改 1：文件名改成 wgx.csv
CSV_PATH = os.path.join(current_dir, "relationship.csv")


def parse_property_str(prop_str):
    """解析属性字符串"""
    try:
        prop_str = re.sub(r'(\w+):', r'"\1":', prop_str)
        prop_str = re.sub(r':\s*([a-zA-Z_]\w*)(?=[,}])', r': "\1"', prop_str)
        return ast.literal_eval(prop_str)
    except Exception:
        return {}


def import_text_relationships():
    print(f"📂 正在读取文本关系文件：{CSV_PATH}")

    try:
        driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASS))
        driver.verify_connectivity()
        print(f"✅ 成功连接到 Neo4j (端口: 12687)")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    with driver.session() as session:
        try:
            # 保持 utf-8-sig
            with open(CSV_PATH, 'r', encoding='utf-8-sig') as csv_file:
                reader = csv.DictReader(csv_file)

                total = 0
                success = 0

                for row in reader:
                    total += 1
                    clean_row = {k.strip(): v for k, v in row.items() if k}

                    start_id = clean_row.get("起始节点ID")
                    end_id = clean_row.get("结束节点ID")
                    rel_type = clean_row.get("关系类型")
                    prop_str = clean_row.get("关系属性", "{}")

                    if not start_id or not end_id or not rel_type:
                        continue

                    properties = parse_property_str(prop_str)

                    # 如果有关系ID也存进去
                    if "关系ID" in clean_row:
                        properties["rel_id"] = clean_row["关系ID"]

                    # ⭐ Cypher 逻辑：
                    # 匹配刚才导入的文本节点 (id 属性)，然后建立关系
                    cypher = f"""
                        MATCH (a {{id: $start_id}})
                        MATCH (b {{id: $end_id}})
                        MERGE (a)-[r:`{rel_type}`]->(b)
                        SET r += $props
                    """

                    try:
                        session.run(cypher, start_id=start_id, end_id=end_id, props=properties)
                        success += 1
                    except Exception as e:
                        print(f"❌ 第 {total} 行关系导入失败: {e}")

                print(f"\n📊 文本关系导入结束！扫描 {total} 行，成功建立 {success} 条连接")

        except FileNotFoundError:
            print(f"❌ 找不到文件：{CSV_PATH}")
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")

    driver.close()


if __name__ == "__main__":
    import_text_relationships()