from clearvoice import ClearVoice
import os
def process_directory(input_dir, output_dir, task='speech_enhancement'):
    # 初始化模型
    cv = ClearVoice(
        task=task,
        model_names=['MossFormerGAN_SE_16K'] if task == 'speech_enhancement' else 
                   ['MossFormer2_SS_16K'] if task == 'speech_separation' else
                   ['AV_MossFormer2_TSE_16K']
    )
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有音频文件
    audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.wav', '.mp4', '.avi'))]
    
    # 批量处理
    for audio_file in audio_files:
        input_path = os.path.join(input_dir, audio_file)
        cv(
            input_path=input_path,
            online_write=True,
            output_path=output_dir
        )
        print(f"Processed: {audio_file}")

# 使用示例
#语音增强
process_directory(
    #[可配置]
    input_dir = '../../voice_gen/pre_deal/fix',
    output_dir = '../../voice_gen/pre_deal/speech_enhancement',
    task='speech_enhancement'
)

#人声分离
process_directory(
    #[可配置]
    input_dir = '../../voice_gen/pre_deal/fix',
    output_dir = '../../voice_gen/pre_deal/speech_separation',
    task='speech_separation'
)