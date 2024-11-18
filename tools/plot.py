import matplotlib.pyplot as plt
import re
import numpy as np
import os

def create_result_folder(log_filename, dataset):
    # 獲取日誌檔案的基本名稱（不含副檔名）
    base_name = os.path.splitext(os.path.basename(log_filename))[0]
    
    # 創建 result 資料夾（如果不存在）
    result_folder = "result" + '/' + dataset
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    # 創建以日誌檔案名命名的子資料夾
    result_subfolder = os.path.join(result_folder, f"result_{base_name}")
    if not os.path.exists(result_subfolder):
        os.makedirs(result_subfolder)
    
    return result_subfolder

def parse_log_file(filename, source):
    # 存儲解析後的數據
    training_data = {
        'epochs': [],
        'train_loss': [],
        'source_acc': [],
        'iterations': [],
        'finetune_loss': [],
        'finetune_acc': []
    }
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # 使用動態 source 變數解析訓練階段數據
        train_match = re.search(rf"Epoch (\d+), train loss = ([\d.]+), source_acc = {{'{source}': ([\d.]+)}}", line)
        if train_match:
            training_data['epochs'].append(int(train_match.group(1)))
            training_data['train_loss'].append(float(train_match.group(2)))
            training_data['source_acc'].append(float(train_match.group(3)))
            
        # 解析微調階段數據
        finetune_match = re.search(r'Iteration (\d+), loss = ([\d.]+), acc = ([\d.]+)', line)
        if finetune_match:
            training_data['iterations'].append(int(finetune_match.group(1)))
            training_data['finetune_loss'].append(float(finetune_match.group(2)))
            training_data['finetune_acc'].append(float(finetune_match.group(3)))
    
    return training_data


def plot_training_progress(data, result_folder):
    # 創建一個2x2的子圖
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 訓練損失曲線
    ax1.plot(data['epochs'], data['train_loss'], 'b-', label='Training Loss')
    ax1.set_title('Training Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # 源域準確率曲線
    ax2.plot(data['epochs'], data['source_acc'], 'g-', label='Source Accuracy')
    ax2.set_title('Source Domain Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # 微調損失曲線
    ax3.plot(data['iterations'], data['finetune_loss'], 'r-', label='Finetune Loss')
    ax3.set_title('Finetune Loss vs. Iterations')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    ax3.legend()
    
    # 微調準確率曲線
    ax4.plot(data['iterations'], data['finetune_acc'], 'purple', label='Finetune Accuracy')
    ax4.set_title('Finetune Accuracy vs. Iterations')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存圖片到結果資料夾
    plot_path = os.path.join(result_folder, 'training_progress.png')
    plt.savefig(plot_path)
    plt.close()
    
    # 保存數值數據到 CSV 文件
    save_data_to_csv(data, result_folder)
    
    return plot_path

def save_data_to_csv(data, result_folder):
    # 保存訓練階段數據
    training_file = os.path.join(result_folder, 'training_data.csv')
    with open(training_file, 'w') as f:
        f.write('Epoch,Train Loss,Source Accuracy\n')
        for i in range(len(data['epochs'])):
            f.write(f"{data['epochs'][i]},{data['train_loss'][i]},{data['source_acc'][i]}\n")
    
    # 保存微調階段數據
    finetune_file = os.path.join(result_folder, 'finetune_data.csv')
    with open(finetune_file, 'w') as f:
        f.write('Iteration,Finetune Loss,Finetune Accuracy\n')
        for i in range(len(data['iterations'])):
            f.write(f"{data['iterations'][i]},{data['finetune_loss'][i]},{data['finetune_acc'][i]}\n")

def main():
    # 創建結果資料夾
    source = "physics"
    target = "cs"
    gnn = "GAT"

    # 獲取日誌文件名
    log_filename = "logs_"+source+"_"+target+"_1000_400_"+gnn+".log"
    
    result_folder = create_result_folder(log_filename, dataset=source + "+" + target)
    
    # 解析數據並繪製圖表，將 source 作為參數傳遞
    training_data = parse_log_file(log_filename, source)
    plot_path = plot_training_progress(training_data, result_folder)
    
    print(f"結果已保存在資料夾: {result_folder}")
    print(f"圖表路徑: {plot_path}")
    print("同時生成了 training_data.csv 和 finetune_data.csv 檔案，包含完整的數值數據")

if __name__ == "__main__":
    main()
