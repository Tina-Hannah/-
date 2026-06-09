"""
焊接缺陷细粒度SGG系统 - 主程序
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config.weld_config import config
from data.dataset import WeldingDataset
from data.preprocess import preprocess_dataset
from data.annotation_tool import run_annotation_tool
from models.weld_sgg import WeldSGGSystem
from utils.visualize import plot_statistics, create_dashboard


def main():
    parser = argparse.ArgumentParser(
        description="焊接缺陷细粒度SGG系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 预处理数据
  python main.py preprocess --data_dir dataset/welding_images --output_dir dataset/processed

  # 2. 人工标注关系（可选）
  python main.py annotate

  # 3. 训练模型（使用自动生成的关系）
  python main.py train --data_dir dataset/processed --epochs 50

  # 4. 推理预测
  python main.py infer --image_dir test_images --output_dir results

  # 5. 评估系统
  python main.py evaluate --data_dir dataset/processed --model_path models/best_model.pth
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 数据预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理原始数据')
    preprocess_parser.add_argument("--data_dir", type=str, default=config.DATA_DIR,
                                   help="原始数据目录（包含图像和Labelme标注）")
    preprocess_parser.add_argument("--output_dir", type=str, default=config.PROCESSED_DIR,
                                   help="预处理输出目录")
    preprocess_parser.add_argument("--generate_relations", action='store_true', default=True,
                                   help="自动生成关系（默认启用）")
    preprocess_parser.add_argument("--split_dataset", action='store_true', default=True,
                                   help="分割数据集为训练/验证/测试集")

    # 标注工具命令
    annotate_parser = subparsers.add_parser('annotate', help='启动关系标注工具')
    annotate_parser.add_argument("--data_dir", type=str, default=config.DATA_DIR,
                                 help="数据目录")
    annotate_parser.add_argument("--output_dir", type=str, default=config.ANNOTATED_DIR,
                                 help="标注输出目录")

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练SGG模型')
    train_parser.add_argument("--data_dir", type=str, default=config.PROCESSED_DIR,
                              help="训练数据目录")
    train_parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                              help="训练轮数")
    train_parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                              help="批次大小")
    train_parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                              help="学习率")
    train_parser.add_argument("--model_dir", type=str, default="models",
                              help="模型保存目录")

    # 推理命令
    infer_parser = subparsers.add_parser('infer', help='推理预测')
    infer_parser.add_argument("--image_dir", type=str, required=True,
                              help="图像目录")
    infer_parser.add_argument("--model_path", type=str, default=None,
                              help="模型路径（如使用训练好的模型）")
    infer_parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR,
                              help="输出目录")
    infer_parser.add_argument("--visualize", action='store_true', default=True,
                              help="生成可视化结果")
    infer_parser.add_argument("--dashboard", action='store_true', default=True,
                              help="生成HTML仪表板")

    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估系统性能')
    eval_parser.add_argument("--data_dir", type=str, default=config.PROCESSED_DIR,
                             help="评估数据目录")
    eval_parser.add_argument("--model_path", type=str, default=None,
                             help="模型路径（如使用训练好的模型）")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation_results",
                             help="评估结果目录")

    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析数据集')
    analyze_parser.add_argument("--data_dir", type=str, default=config.PROCESSED_DIR,
                                help="数据目录")
    analyze_parser.add_argument("--output_dir", type=str, default="analysis_results",
                                help="分析结果目录")

    args = parser.parse_args()

    if args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'annotate':
        annotate_relations(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'infer':
        infer_predictions(args)
    elif args.command == 'evaluate':
        evaluate_system(args)
    elif args.command == 'analyze':
        analyze_dataset(args)
    else:
        parser.print_help()


def preprocess_data(args):
    """预处理数据"""
    print("=" * 60)
    print("焊接缺陷数据预处理")
    print("=" * 60)

    # 检查数据目录
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    # 预处理数据集
    dataset_stats = preprocess_dataset(
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        generate_relations=args.generate_relations,
        split_dataset=args.split_dataset
    )

    print("\n预处理完成!")
    print(f"处理图像数: {dataset_stats['total_images']}")
    print(f"总区域数: {dataset_stats['total_regions']}")
    print(f"总关系数: {dataset_stats['total_relations']}")
    print(f"输出目录: {args.output_dir}")


def annotate_relations(args):
    """标注关系"""
    print("=" * 60)
    print("焊接缺陷关系标注工具")
    print("=" * 60)

    run_annotation_tool(args.data_dir, args.output_dir)


def train_model(args):
    """训练模型"""
    print("=" * 60)
    print("焊接缺陷SGG模型训练")
    print("=" * 60)

    # 加载数据集
    print(f"加载数据集: {args.data_dir}")
    dataset = WeldingDataset(
        data_dir=args.data_dir,
        mode="processed"
    )

    if len(dataset) == 0:
        print("错误: 未找到训练数据")
        return

    # 分割数据集
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=0.7,
        val_ratio=0.15
    )

    print(f"\n训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"测试集: {len(test_dataset)} 张图像")

    # 这里可以集成实际的模型训练代码
    # 基于复用的train_egtr.py框架

    print("\n训练功能待实现...")
    print("可以使用以下代码框架:")
    print("""
    # 1. 创建数据加载器
    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=args.batch_size)

    # 2. 初始化模型（复用Deformable DETR架构）
    model = WeldSGGModel(
        num_regions=len(config.REGION_CLASSES),
        num_relations=len(config.RELATION_CLASSES)
    )

    # 3. 训练循环
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, epoch, args)
        validate(model, val_loader, epoch, args)

    # 4. 保存模型
    save_model(model, args.model_dir)
    """)


def infer_predictions(args):
    """推理预测"""
    print("=" * 60)
    print("焊接缺陷SGG推理预测")
    print("=" * 60)

    # 检查图像目录
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"错误: 图像目录不存在: {image_dir}")
        return

    # 获取图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))

    if not image_paths:
        print(f"错误: 在 {image_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_paths)} 张图像")

    # 初始化SGG系统
    sgg_system = WeldSGGSystem(
        use_spatial_rules=True,
        use_semantic_rules=True,
        use_region_detector=False  # 使用标注数据，而不是检测模型
    )

    # 批量处理图像
    scene_graphs = sgg_system.process_batch([str(p) for p in image_paths])

    # 导出结果
    output_dir = Path(args.output_dir)
    summary = sgg_system.export_results(scene_graphs, str(output_dir))

    # 生成可视化
    if args.visualize:
        print("\n生成可视化结果...")
        # 可视化已在export_results中生成

    # 生成仪表板
    if args.dashboard:
        print("生成HTML仪表板...")
        dashboard_path = create_dashboard(scene_graphs, str(output_dir))
        print(f"仪表板: file://{dashboard_path.absolute()}")

    print("\n推理完成!")
    print(f"输出目录: {output_dir}")
    print(f"处理图像数: {len(scene_graphs)}")
    print(f"总关系数: {sum(len(sg.relations) for sg in scene_graphs)}")


def evaluate_system(args):
    """评估系统"""
    print("=" * 60)
    print("焊接缺陷SGG系统评估")
    print("=" * 60)

    # 加载数据集
    dataset = WeldingDataset(
        data_dir=args.data_dir,
        mode="processed"
    )

    if len(dataset) == 0:
        print("错误: 未找到评估数据")
        return

    # 初始化SGG系统
    sgg_system = WeldSGGSystem(
        use_spatial_rules=True,
        use_semantic_rules=True
    )

    # 评估
    print("正在评估系统性能...")
    metrics = sgg_system.evaluate_on_dataset(args.data_dir)

    # 打印结果
    from evaluation.metrics import print_metrics
    print_metrics(metrics)

    # 保存评估结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    import json
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存: {metrics_file}")


def analyze_dataset(args):
    """分析数据集"""
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)

    # 加载数据集
    dataset = WeldingDataset(
        data_dir=args.data_dir,
        mode="processed"
    )

    if len(dataset) == 0:
        print("错误: 未找到数据")
        return

    # 获取统计信息
    stats = dataset.get_statistics()

    # 打印统计信息
    print(f"\n数据集大小: {stats['total_images']} 张图像")
    print(f"总区域数: {stats['total_regions']}")
    print(f"总关系数: {stats['total_relations']}")
    print(f"总缺陷数: {stats['total_defects']}")

    print("\n区域分布:")
    for cls_name, count in sorted(stats['region_distribution'].items()):
        percentage = count / stats['total_regions'] * 100
        print(f"  {cls_name:<20}: {count:>4} ({percentage:>5.1f}%)")

    print("\n关系分布:")
    for rel_name, count in sorted(stats['relation_distribution'].items()):
        total = sum(stats['relation_distribution'].values())
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {rel_name:<20}: {count:>4} ({percentage:>5.1f}%)")

    print(f"\n平均每图区域数: {stats['avg_regions_per_image']:.1f}")
    print(f"平均每图关系数: {stats['avg_relations_per_image']:.1f}")
    print(f"平均每图缺陷数: {stats['avg_defects_per_image']:.1f}")

    print("\n最常见关系模式:")
    for pattern, count in stats['common_patterns'][:10]:
        print(f"  {pattern}")

    # 保存分析结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    import json
    stats_file = output_dir / "dataset_analysis.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 生成图表
    chart_file = output_dir / "analysis_charts.png"
    plot_statistics(stats, str(chart_file))

    print(f"\n分析结果已保存到: {output_dir}")
    print(f"统计文件: {stats_file}")
    print(f"图表文件: {chart_file}")


if __name__ == "__main__":
    main()