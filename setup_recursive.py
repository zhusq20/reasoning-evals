import os
import subprocess

def install_packages_recursively(folder):
    """
    递归遍历文件夹，找到所有包含 setup.py 的目录并使用 pip 安装。
    """
    for root, dirs, files in os.walk(folder):  # 使用 os.walk 遍历所有子目录
        if 'setup.py' in files:  # 如果当前目录包含 setup.py
            print(f"Installing package in {root}...")
            try:
                # 使用 pip 安装当前目录
                subprocess.check_call(['pip', 'install', root])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install package in {root}. Error: {e}")
        else:
            print(f"Skipping {root}, no setup.py found.")

if __name__ == "__main__":
    # 假设脚本在父目录中运行
    parent_folder = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    print(f"Starting installation in: {parent_folder}")
    install_packages_recursively(parent_folder)
    print("Installation complete.")