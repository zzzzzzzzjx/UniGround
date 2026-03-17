#!/usr/bin/env python3
"""
将文件夹下的数字命名子文件夹均分到指定数量的目标文件夹中
"""
import os
import shutil
import argparse
from pathlib import Path


def copy_tree_with_error_handling(src, dst):
    """
    递归复制目录树，跳过损坏的符号链接但复制所有其他内容
    """
    os.makedirs(dst, exist_ok=True)
    errors = []

    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        try:
            if os.path.islink(src_path):
                # 处理符号链接
                link_target = os.readlink(src_path)
                os.symlink(link_target, dst_path)
            elif os.path.isdir(src_path):
                # 递归复制目录
                copy_tree_with_error_handling(src_path, dst_path)
            else:
                # 复制文件
                shutil.copy2(src_path, dst_path)
        except (OSError, IOError) as e:
            errors.append(f"    跳过损坏的文件/链接: {item} ({e})")
            continue

    return errors


def distribute_folders(source_dir, num_groups, output_dir=None, mode='copy'):
    """
    将源目录下的数字命名子文件夹均分到指定数量的组中

    Args:
        source_dir: 源文件夹路径
        num_groups: 要分成的组数
        output_dir: 输出目录（默认为源目录下的distributed文件夹）
        mode: 'copy' 或 'move'，决定是复制还是移动文件夹
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"错误: 源目录 {source_dir} 不存在")
        return

    # 获取所有数字命名的子文件夹
    subfolders = []
    for item in source_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            subfolders.append(item)

    if not subfolders:
        print(f"错误: 在 {source_dir} 中没有找到数字命名的子文件夹")
        return

    # 按数字排序
    subfolders.sort(key=lambda x: int(x.name))
    print(f"找到 {len(subfolders)} 个子文件夹: {[f.name for f in subfolders]}")

    # 设置输出目录
    if output_dir is None:
        output_path = source_path / "distributed"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # 计算每组的数量
    total = len(subfolders)
    base_count = total // num_groups
    remainder = total % num_groups

    # 分配文件夹
    current_idx = 0
    for group_num in range(1, num_groups + 1):
        # 前 remainder 组多分配一个
        count = base_count + (1 if group_num <= remainder else 0)

        # 创建目标组文件夹
        target_group = output_path / str(group_num)
        target_group.mkdir(parents=True, exist_ok=True)

        print(f"\n组 {group_num}: 分配 {count} 个文件夹")

        # 分配文件夹到当前组
        for i in range(count):
            if current_idx >= total:
                break

            source_folder = subfolders[current_idx]
            target_folder = target_group / source_folder.name

            if mode == 'move':
                shutil.move(str(source_folder), str(target_folder))
                print(f"  移动: {source_folder.name} -> {group_num}/{source_folder.name}")
            else:  # copy
                if target_folder.exists():
                    shutil.rmtree(target_folder)
                errors = copy_tree_with_error_handling(str(source_folder), str(target_folder))
                print(f"  复制: {source_folder.name} -> {group_num}/{source_folder.name}")
                if errors:
                    for error in errors:
                        print(error)

            current_idx += 1

    print(f"\n完成! 输出目录: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='将数字命名的子文件夹均分到指定数量的组中'
    )
    parser.add_argument(
        'source_dir',
        help='源文件夹路径'
    )
    parser.add_argument(
        'num_groups',
        type=int,
        help='要分成的组数'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='输出目录（默认为源目录下的distributed文件夹）'
    )
    parser.add_argument(
        '--mode',
        choices=['copy', 'move'],
        default='copy',
        help='操作模式: copy(复制) 或 move(移动)，默认为copy'
    )

    args = parser.parse_args()

    distribute_folders(
        args.source_dir,
        args.num_groups,
        args.output_dir,
        args.mode
    )


if __name__ == '__main__':
    main()
