"""
推理脚本 - 单独的推理模块
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from models.weld_sgg import WeldSGGSystem
from utils.visualize import create_dashboard


def main():
    parser = argparse.ArgumentParser(description="焊接缺陷SGG推理")

    parser.add_argument("--image_dir", type=str, required=True,
                        help="图像目录")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="输出目录")
    parser.add_argument("--model_path", type=str, default=None,
                        help="模型路径（可选，如使用训练好的模型）")
    parser.add_argument("--use_region_detector", action='store_true',
                        help="使用区域检测器（而不是标注数据）")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批处理大小")
    parser.add_argument("--visualize", action='store_true', default=True,
                        help="生成可视化结果")
    parser.add_argument("--dashboard", action='store_true', default=True,
                        help="生成HTML仪表板")

    args = parser.parse_args()

    # 执行推理
    run_inference(args)


def run_inference(args):
    """执行推理"""
    print("=" * 60)
    print("焊接缺陷细粒度SGG推理")
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
        use_region_detector=args.use_region_detector
    )

    # 批量处理图像
    print("\n开始处理图像...")
    scene_graphs = []

    # 分批处理
    batch_size = args.batch_size
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"处理批次 {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}")

        batch_sgs = sgg_system.process_batch([str(p) for p in batch_paths])
        scene_graphs.extend(batch_sgs)

    # 导出结果
    output_dir = Path(args.output_dir)
    summary = sgg_system.export_results(scene_graphs, str(output_dir))

    # 生成仪表板
    if args.dashboard:
        print("\n生成HTML仪表板...")
        dashboard_path = create_dashboard(scene_graphs, str(output_dir))
        print(f"仪表板: file://{dashboard_path.absolute()}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)

    print(f"\n输出目录: {output_dir.absolute()}")
    print(f"处理图像数: {len(scene_graphs)}")
    print(f"总区域数: {sum(len(sg.regions) for sg in scene_graphs)}")
    print(f"总关系数: {sum(len(sg.relations) for sg in scene_graphs)}")

    print("\n区域分布:")
    for cls_name, count in sorted(summary["region_distribution"].items()):
        total = sum(summary["region_distribution"].values())
        percentage = count / total * 100 if total > 0 else 0
        print(f"  {cls_name:<20}: {count:>4} ({percentage:>5.1f}%)")

    print("\n最常见关系模式:")
    for pattern, count in summary["common_patterns_overall"][:5]:
        print(f"  {pattern}")


if __name__ == "__main__":
    main()