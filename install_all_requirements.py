import os
import subprocess

def install_requirements_recursively(folder):
    """
    递归查找并安装所有 requirements.txt 文件中的依赖。
    """
    for root, dirs, files in os.walk(folder):  # 遍历所有子目录
        if 'requirements.txt' in files:  # 如果当前目录包含 requirements.txt
            requirements_path = os.path.join(root, 'requirements.txt')
            print(f"Installing dependencies from: {requirements_path}...")
            try:
                # 使用 pip 安装 requirements.txt
                subprocess.check_call(['pip', 'install', '-r', requirements_path])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install dependencies from {requirements_path}. Error: {e}")
        else:
            print(f"Skipping {root}, no requirements.txt found.")

if __name__ == "__main__":
    # 假设脚本在父目录中运行
    parent_folder = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    print(f"Starting installation in: {parent_folder}")
    install_requirements_recursively(parent_folder)
    print("Installation complete.")