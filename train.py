"""
训练脚本 - 基于复用的train_egtr.py框架
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.weld_config import config
from data.dataset import WeldingDataset


def main():
    parser = argparse.ArgumentParser(description="焊接缺陷SGG模型训练")

    parser.add_argument("--data_dir", type=str, default=config.PROCESSED_DIR,
                        help="训练数据目录")
    parser.add_argument("--model_dir", type=str, default="models",
                        help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="学习率")
    parser.add_argument("--lr_backbone", type=float, default=config.LR_BACKBONE,
                        help="主干网络学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载工作线程数")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--debug", action='store_true',
                        help="调试模式（使用少量数据）")

    args = parser.parse_args()

    # 执行训练
    train_model(args)


def train_model(args):
    """训练模型"""
    print("=" * 60)
    print("焊接缺陷细粒度SGG模型训练")
    print("=" * 60)

    # 创建模型目录
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    # 加载数据集
    print(f"\n加载数据集: {args.data_dir}")
    dataset = WeldingDataset(
        data_dir=args.data_dir,
        mode="processed"
    )

    if len(dataset) == 0:
        print("错误: 未找到训练数据")
        return

    # 分割数据集
    print("分割数据集...")
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=0.7,
        val_ratio=0.15
    )

    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")

    # 数据集统计
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()

    print(f"\n训练集统计:")
    print(f"  总区域数: {train_stats['total_regions']}")
    print(f"  总关系数: {train_stats['total_relations']}")
    print(f"  平均每图区域数: {train_stats['avg_regions_per_image']:.1f}")
    print(f"  平均每图关系数: {train_stats['avg_relations_per_image']:.1f}")

    print(f"\n验证集统计:")
    print(f"  总区域数: {val_stats['total_regions']}")
    print(f"  总关系数: {val_stats['total_relations']}")
    print(f"  平均每图区域数: {val_stats['avg_regions_per_image']:.1f}")
    print(f"  平均每图关系数: {val_stats['avg_relations_per_image']:.1f}")

    # 这里可以集成实际的模型训练代码
    # 基于复用的train_egtr.py框架

    print("\n" + "=" * 60)
    print("训练功能代码框架")
    print("=" * 60)

    print("""
    基于复用的train_egtr.py框架，需要以下步骤:

    1. 数据准备:
       - 将场景图数据转换为DETR格式
       - 创建数据加载器
       - 数据增强（随机裁剪、翻转等）

    2. 模型初始化:
       - 基于Deformable DETR架构
       - 修改分类头：11个区域类别 + 10个关系类别
       - 加载预训练权重

    3. 训练配置:
       - 优化器：AdamW
       - 学习率调度器
       - 损失函数：区域分类损失 + 关系分类损失

    4. 训练循环:
       for epoch in range(args.epochs):
           # 训练阶段
           train_one_epoch(model, train_loader, optimizer, epoch)

           # 验证阶段
           validate(model, val_loader, epoch)

           # 保存检查点

    5. 评估:
       - 在测试集上评估
       - 计算mAP等指标

    具体实现需要集成原train_egtr.py的代码框架。
    """)

    # 保存训练配置
    save_training_config(args, model_dir)

    print(f"\n训练配置已保存到: {model_dir}")
    print("请根据实际需求实现训练代码。")


def save_training_config(args, model_dir: Path):
    """保存训练配置"""
    config_dict = {
        "data_dir": args.data_dir,
        "model_dir": str(model_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "region_classes": config.REGION_CLASSES,
        "relation_classes": config.RELATION_CLASSES,
        "label_mapping": config.LABEL_MAPPING
    }

    import json
    config_file = model_dir / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    print(f"训练配置文件: {config_file}")


if __name__ == "__main__":
    main()