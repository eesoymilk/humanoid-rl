import pandas as pd
import matplotlib.pyplot as plt
import os

ALGOS = ['SAC', 'PPO', 'TD3', 'DDPG', 'A2C']
WRAPPED = ['no_wrapped', 'wrapped']

def plot_csv_files(directory_path):
    # 檢查目錄中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            # 完整的檔案路徑
            file_path = os.path.join(directory_path, filename)
            
            # 讀取 CSV 文件
            data = pd.read_csv(file_path)
            
            # 繪製圖表
            plt.figure(figsize=(10, 6))
            plt.plot(data['Step'], data['Value'], label=f'{filename[:-4]}')  # 使用文件名作為圖例標籤
            
            # 圖表標題和軸標籤
            y_label = ' '.join(filename[filename.index('tag-') + 4:-4].split('_'))
            plt.xlabel('Step')
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)
            
            # 儲存為 PNG
            png_path = os.path.join(directory_path, f"{filename[:-4]}.png")
            plt.savefig(png_path)
            plt.close()  # 關閉當前圖表以節省記憶體
    
def go_throght_dir(base_dir):
    for algo in ALGOS:
        for wrapped in WRAPPED:
            plot_csv_files(os.path.join(base_dir, 'FINAL_DATA', 'DATA_CSV', f'{algo}_{wrapped}'))
            
def plot_comparison(base_dir, tags):
    plt.figure(figsize=(15, 10))
    
    for algo in ALGOS:
        for wrapped in WRAPPED:
            directory_path = os.path.join(base_dir, 'FINAL_DATA', 'DATA_CSV', f'{algo}_{wrapped}')
            for filename in os.listdir(directory_path):
                if filename.endswith('.csv') and any(tag in filename for tag in tags):
                    file_path = os.path.join(directory_path, filename)
                    data = pd.read_csv(file_path)
                    
                    for tag in tags:
                        if tag in filename:
                            plt.plot(data['Step'], data['Value'], label=f'{algo}_{wrapped}_{tag}')

    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, f'comparison_{tags[0]}.png'))
    plt.close()

if __name__ == '__main__':
    tags = ['ep_len_mean', 'ep_rew_mean']
    plot_comparison(os.getcwd(), [tags[0]])
    plot_comparison(os.getcwd(), [tags[1]])
    
    go_throght_dir(os.getcwd())