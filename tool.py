import os
import re

def remove_leading_zeros(root_dir, subfolders=["imgs", "masks"], prefix="case_"):
    """
    遍历 root_dir 下的 subfolders，将所有文件名形如
      prefix + 0*NUM + ext
    的，重命名为
      prefix + NUM + ext
    其中 NUM 前面的多余 0 会被删除。

    Args:
        root_dir (str): predictions 根目录
        subfolders (list): 子目录列表，默认 ["imgs","masks"]
        prefix (str): 文件名前缀，默认 "case_"
    """
    # 正则：前缀 + 任意多 0 + 数字 + 扩展名
    pattern = re.compile(rf"^({re.escape(prefix)})0*(\d+)(\.[^.]+)$")

    for sub in subfolders:
        folder = os.path.join(root_dir, sub)
        if not os.path.isdir(folder):
            print(f"跳过，不存在目录: {folder}")
            continue

        for fn in os.listdir(folder):
            m = pattern.match(fn)
            if not m:
                continue

            old_path = os.path.join(folder, fn)
            new_name = f"{m.group(1)}{int(m.group(2))}{m.group(3)}"
            new_path = os.path.join(folder, new_name)

            # 防止重名冲突
            if os.path.exists(new_path):
                print(f"目标已存在，跳过: {new_name}")
                continue

            os.rename(old_path, new_path)
            print(f"重命名: {fn} → {new_name}")


if __name__ == "__main__":
    # 举例：假设你的 prediction 根目录是 ./predictions/UnetPlusPlus
    base = "./predictions"
    remove_leading_zeros(base, subfolders=["imgs", "masks"])
