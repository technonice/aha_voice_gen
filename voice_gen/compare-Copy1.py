import subprocess
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import time

import csv
import os
import re

def extract_similarity_score(output):
    """从命令输出中提取相似度分数"""
    lines = output.split('\n')
    for line in lines:
        if 'similarity score between two input wavs is' in line:
            match = re.search(r'is\s+([0-9.]+)', line)
            if match:
                return float(match.group(1))
    return None

def get_processed_indices(csv_file):
    """获取已经处理过的索引"""
    if not os.path.exists(csv_file):
        return set()
    
    try:
        df = pd.read_csv(csv_file)
        return set(df['index'].tolist())
    except:
        return set()
        
def process_single_pair_with_retry(args, max_retries=3):
    """带重试机制的单个音频对处理"""
    model_id, ref_file, syn_file, i, group_name = args
    
    for attempt in range(max_retries):
        try:
            cmd = [
                'python', './speakerlab/bin/infer_sv.py',
                '--model_id', model_id,
                '--wavs', ref_file, syn_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            similarity = extract_similarity_score(result.stdout) if result.returncode == 0 else None
            
            if similarity is not None:
                return i, similarity, None
            else:
                time.sleep(1)  # 等待后重试
                
        except Exception as e:
            if attempt == max_retries - 1:
                return i, 'ERROR', str(e)
            time.sleep(2)  # 等待后重试
    
    return i, 'ERROR', 'Max retries exceeded'

def smart_add_comparison_group(model_id, folder_a, folder_b, a_name, b_name, output_csv, 
                              group_name, start_idx=1, end_idx=200, max_workers=4, 
                              batch_size=50, max_retries=3):
    """
    智能多线程版本，支持分批处理和进度保存
    """
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    
    # 读取或创建CSV
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=['index', 'reference_file', 'synthesized_file'])
    
    if group_name not in df.columns:
        df[group_name] = None

    # 获取已处理的索引
    # processed_indices = set(df[df[group_name].notna()]['index'].tolist()) if group_name in df.columns else set()
    numeric_mask = pd.to_numeric(df[group_name], errors='coerce').notna()
    processed_indices = set(df[numeric_mask]['index'].tolist())
    
    # 准备任务（跳过已处理的）
    tasks = []
    for i in range(start_idx, end_idx + 1):
        if i in processed_indices:
            continue
            
        ref_file = os.path.join(folder_a, f"{a_name}{i}.wav")
        syn_file = os.path.join(folder_b, f"{b_name}{i}.wav")
        
        if all(os.path.exists(f) for f in [ref_file, syn_file]):
            tasks.append((model_id, ref_file, syn_file, i, group_name))

    print(f"{group_name} 需要处理 {len(tasks)} 个文件（跳过 {len(processed_indices)} 个已处理文件）")

    # 分批处理以避免内存问题
    for batch_start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[batch_start:batch_start + batch_size]
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(batch_tasks), desc=f"{group_name} 批次 {batch_start//batch_size + 1}") as pbar:
                for result in executor.map(lambda x: process_single_pair_with_retry(x, max_retries), batch_tasks):
                    batch_results.append(result)
                    pbar.update(1)
                    pbar.set_postfix({'索引': result[0], '分数': result[1]})

        # 更新DataFrame并保存
        for i, score, error in batch_results:
            if error:
                print(f"错误处理索引 {i}: {error}")
            
            if i in df['index'].values:
                df.loc[df['index'] == i, group_name] = score
            else:
                new_row = {
                    'index': i,
                    'reference_file': f"reference_{i}.wav",
                    'synthesized_file': f"synthesized_speech_{i}.wav",
                    group_name: score
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 每批结束后保存进度
        df.to_csv(output_csv, index=False)
        print(f"已保存批次 {batch_start//batch_size + 1} 的进度")

    print(f"{group_name} 全部处理完成!")

# 使用智能版本
if __name__ == "__main__":
    #[可配置]
    MODEL_ID = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    OUTPUT_CSV = "../input_similarity.csv"

    smart_add_comparison_group(
        MODEL_ID,
        "../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/", #对比文件目录a
        "../pre_deal/speech_enhancement/MossFormerGAN_SE_16K", #对比文件目录b
        "reference_",#a目录下文件前缀
        "reference_",#b目录下文件前缀
        OUTPUT_CSV,#输出的csv位置
        "speech_enhancement",#记录栏名称
        start_idx=1,
        end_idx=200,
        max_workers=20,
        batch_size=100,
        max_retries=3
    )

    smart_add_comparison_group(
        MODEL_ID,
        "../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/",
        "../pre_deal/speech_separation/s1",
        "reference_",
        "reference_",
        OUTPUT_CSV,
        "speech_separation_s1",
        start_idx=1,
        end_idx=200,
        max_workers=20,
        batch_size=100,
        max_retries=3
    )
    
    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/",
    #     "../pre_deal/enhancement/mix_dnor",
    #     "reference_",
    #     "reference_",
    #     OUTPUT_CSV,
    #     "mix_dnor",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/",
    #     "../pre_deal/enhancement/mix_normalized",
    #     "reference_",
    #     "reference_",
    #     OUTPUT_CSV,
    #     "mix_normalized",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../index-tts/aigc_speech_generation_tasks/",
    #     "./resample/3_output/",
    #     "reference_",
    #     "synthesized_speech_",
    #     OUTPUT_CSV,
    #     "3_output",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../index-tts/aigc_speech_generation_tasks/",
    #     "./resample/4_output/",
    #     "reference_",
    #     "synthesized_speech_",
    #     OUTPUT_CSV,
    #     "4_output",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../index-tts/aigc_speech_generation_tasks/",
    #     "./resample/5_output/",
    #     "reference_",
    #     "synthesized_speech_",
    #     OUTPUT_CSV,
    #     "5_output",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../index-tts/aigc_speech_generation_tasks/",
    #     "./resample/8_output/",
    #     "reference_",
    #     "synthesized_speech_",
    #     OUTPUT_CSV,
    #     "8_output",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

    # smart_add_comparison_group(
    #     MODEL_ID,
    #     "../index-tts/aigc_speech_generation_tasks/",
    #     "./resample/9_output/",
    #     "reference_",
    #     "synthesized_speech_",
    #     OUTPUT_CSV,
    #     "9_output",
    #     start_idx=1,
    #     end_idx=200,
    #     max_workers=20,
    #     batch_size=100,
    #     max_retries=3
    # )

# # 1. 顺序处理多个组（每个组内多线程）
# batch_add_comparison_groups(MODEL_ID, group_configs, OUTPUT_CSV)

# # 2. 并行处理多个组（组间和组内都并行）
# parallel_batch_add_groups(MODEL_ID, group_configs, OUTPUT_CSV)

# # 3. 智能处理（带重试和进度保存）
# smart_add_comparison_group(MODEL_ID, ...)