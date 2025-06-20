import os
import time
import concurrent.futures

# 引入强大的进度条库 tqdm
# 如果你没有安装，请先在命令行运行: pip install tqdm
try:
    import tqdm
except ImportError:
    print("错误：缺少 'tqdm' 库。")
    print("请先通过命令 'pip install tqdm' 安装它，然后再运行此脚本。")
    exit()

# --- 可配置参数 ---
# 1. 定义要检查的图片后缀名 (已根据你的要求简化)
#    使用元组比列表稍快，用于 .endswith()
IMAGE_EXTENSIONS = ('.jpg', '.jpeg')

# 2. 配置并发线程数
#    None 会让程序自动选择合适的线程数，通常是 CPU核心数 * 5
#    对于I/O密集型任务，可以设置得更高，如 16, 32, 甚至 64，可以自行测试以获得最佳性能
MAX_WORKERS = 8


def check_folder_for_jpg(folder_path: str) -> tuple[str, bool]:
    """
    高效地检查单个文件夹是否包含JPG图片。
    设计为在多线程环境中安全运行。

    Args:
        folder_path: 要检查的子文件夹路径。

    Returns:
        一个元组 (文件夹路径, 是否包含JPG)。
        例如: ('/path/to/folder_a', True)
    """
    try:
        # os.scandir 是最高效的目录遍历方式
        for entry in os.scandir(folder_path):
            # 检查是否是文件，并且名字以 .jpg 或 .jpeg 结尾 (忽略大小写)
            if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                # 只要找到一张，就立刻返回，不再继续查找
                return folder_path, True

        # 如果遍历完整个文件夹都没找到，则说明不包含
        return folder_path, False

    except OSError as e:
        # 处理可能出现的权限问题等错误
        print(f"\n警告：无法访问文件夹 {folder_path}，已跳过。原因: {e}")
        # 出错时，我们默认它“包含”图片，以避免错误地将其列为空文件夹
        return folder_path, True


def find_empty_jpg_folders_fast(parent_folder: str) -> list[str]:
    """
    使用带进度条的多线程方法，超快速地查找不含JPG的子文件夹。
    """
    if not os.path.isdir(parent_folder):
        print(f"错误：提供的路径 '{parent_folder}' 不是一个有效的文件夹。")
        return []

    empty_folders = []

    # --- 第 1 步：快速收集所有需要检查的子文件夹 ---
    print(f"[*] 正在扫描根目录: {parent_folder}")
    print("[*] 第 1 步: 正在快速收集所有子文件夹...")
    try:
        with os.scandir(parent_folder) as entries:
            subfolders_to_check = [entry.path for entry in entries if entry.is_dir()]
    except OSError as e:
        print(f"\n错误：无法读取顶级文件夹 {parent_folder}。原因: {e}")
        return []

    if not subfolders_to_check:
        print("[-] 在指定目录下没有找到任何子文件夹。")
        return []

    total_folders = len(subfolders_to_check)
    print(f"[+] 找到 {total_folders} 个子文件夹。")
    print(f"[*] 第 2 步: 开始并行检查 (使用 {MAX_WORKERS or '自动'} 个线程)...")

    # --- 第 2 步：使用线程池和进度条并行处理 ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到线程池
        futures = [executor.submit(check_folder_for_jpg, folder) for folder in subfolders_to_check]

        # 使用 tqdm 创建一个进度条来实时展示处理进度
        progress_bar = tqdm.tqdm(concurrent.futures.as_completed(futures),
                                 total=total_folders,
                                 desc="检查进度",
                                 unit="个文件夹")

        for future in progress_bar:
            try:
                path, contains_jpg = future.result()
                # 如果检查结果为“不包含”，则将其加入列表
                if not contains_jpg:
                    empty_folders.append(path)
            except Exception as e:
                # 捕获在任务执行中可能发生的任何其他异常
                print(f"\n处理文件夹时发生意外错误: {e}")

    return sorted(empty_folders)  # 对结果排序，让输出更整洁


if __name__ == "__main__":
    # --- 请在这里修改你要扫描的大文件夹路径！ ---
    # Windows 示例: target_folder = r'E:\MyLargePhotoCollection'
    # macOS/Linux 示例: target_folder = '/data/photos/archive'
    target_folder = 'D:/Mydownload/tiaozhanbei/VideoMAEv2-master/dataset/something-something-v2/20bn-something-something-v2-frames'  # '.' 表示当前脚本所在的文件夹，请务必修改为你自己的路径

    print("=" * 60)
    print("  高速空图片文件夹扫描脚本  ")
    print("=" * 60)

    start_time = time.perf_counter()

    result_folders = find_empty_jpg_folders_fast(target_folder)

    end_time = time.perf_counter()

    print("\n" + "=" * 60)
    print("🎉 扫描完成！")
    print(f"总计用时: {end_time - start_time:.4f} 秒")

    if result_folders:
        print(f"✅ 共找到 {len(result_folders)} 个不含 JPG 图片的文件夹:")
        # 将结果保存到文件中，方便后续处理
        output_file = "empty_folders_list.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for folder in result_folders:
                print(f"  - {folder}")
                f.write(folder + '\n')
        print(f"\n[i] 详细列表已保存到文件: {os.path.abspath(output_file)}")
    else:
        print("✅ 所有子文件夹中都包含 JPG 图片，或者没有找到子文件夹。")
    print("=" * 60)