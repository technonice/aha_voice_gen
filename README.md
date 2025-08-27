# aha_voice_gen
2025天池-智能语音交互认证生成赛-aha!

## 使用配置（算力平台 AutoDL）
	- 镜像 PyTorch  2.3.0 Python  3.12(ubuntu22.04) CUDA  12.1
	- GPU RTX 4090(24GB) * 1
	- CPU 16 vCPU Intel(R) Xeon(R) Gold 6430
	- 内存120GB

---

## 开源模型
	- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) 处理数据
	- [index-tts1.5](https://github.com/index-tts/index-tts) 语音克隆
	- [ERes2NetV2说话人确认-中文-通用-200k-Spkrs](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary) 确认人说话 

#### 简略过程介绍
- 对**参考音频**进行简单分析，其中包含<u>中英双语人声、嘈杂环境、多人声、方言、超短音频、超长音频等等</u>
- 对音频进行适当的截断，我是用的<u>方法是统一截断</u>，**假如**需要防止截断中音频出现最后人声说话不完全可能需要，检测人声语音空白间隔处，然后就近截断的方法。
- 对于**多人声**，并不清楚比赛的比对音频是对首个人声音纹进行比对标准，还是多人整体的效果对比。我使用的是<u>**截断首个人声音频进行处理**</u>。
- 而输出音频的**参考文本**中，出现<u>英文缩写、度量单位等</u>情况，在index-tts中虽然有训练到部分可以正常生成，但还是不能完全成功。因此，我的**解决方案**是，检索参考文本中，非中文字符、非标点符号等情况的序列，由于获取的序列过少，就手动修改了内容（如'ADS'->'A D S'）,index-tts做了类似的适配，中间加空格则很好处理，而且也可以加拼音之类的方式修改错误读音。**对于真实大量样本的话**，需要让小型的语言模型听从指令修改正确反馈。
- 对音频使用上面的简单截断等处理后，<u>使用**ClearerVoice-Studio处理**数据</u>，其功能主要用到语音增强和人声分离，使用说话人确认进行比对效果发现语音增强绝大部分更好，不过语音增强部分中音频出现效果极差，则用人声部分代替。
- 对语音增强后使用<u>**归一化**</u>（峰值、峰值和响度）这两种。
- 处理后的数据用于<u>**index-tts**</u>的参考音频，并使用不同的参数进行调整得到不同组的音频数据。
- 然后对不同组的音频进行<u>**说话人确认**</u>来比对最开始的参考音频，看谁的相似度高后打包到一个文件夹中，得到比较好的结果。如果需要比对更好的结果，可能需要根据<u>**说话人确认**</u>中的子类评分来适配比赛中对应的优选项，再处理。

##### 具体过程：
解压压缩包,压缩包内有 （后缀非copy的文件是之前程序的草稿，可能不适配，尽量按顺序走）
	- voice_gen/AISumerCamp_audio_generation_fight
	- ……
后续的克隆仓库都是在voice_gen这个目录下

```shell
cd voice_gen
```
###### 克隆参考集
```bash
git clone https://www.modelscope.cn/datasets/Datawhale/AISumerCamp_audio_generation_fight.git
```

###### [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) 处理数据

```shell
# 克隆仓库
git clone https://github.com/modelscope/ClearerVoice-Studio.git

# 生成Conda环境和安装依赖
cd ClearerVoice-Studio 
conda create --prefix ./ClearerVoice-Studio python=3.8
conda activate ./ClearerVoice-Studio 
conda install -c conda-forge pyworld

pip install -r requirements.txt


# git模型下载，请确保已安装git lfs
mkdir -p checkpoints
git clone https://www.modelscope.cn/iic/ClearerVoice-Studio.git checkpoints

#离开环境回到voice_gen目录
conda deactivate
cd ..
```

###### [index-tts1.5](https://github.com/index-tts/index-tts) 语音克隆


```shell
# 克隆仓库
git clone https://github.com/index-tts/index-tts.git

# 生成Conda环境和安装依赖
cd index-tts
conda create --prefix ./index-tts python=3.10
conda activate ./index-tts
apt-get install ffmpeg
# or use conda to install ffmpeg
conda install -c conda-forge ffmpeg

# Install `IndexTTS` as a package:
pip install -e .

#Recommended for China users. 如果下载速度慢，可以使用镜像：
export HF_ENDPOINT="https://hf-mirror.com"

#Download models by `huggingface-cli`:
huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints


#离开环境回到voice_gen目录
conda deactivate
cd ..
```


###### [ERes2NetV2说话人确认-中文-通用-200k-Spkrs](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary) 确认人说话 


```shell
git clone https://github.com/alibaba-damo-academy/3D-Speaker.git

cd 3D-Speaker
conda create --prefix ./3D-Speaker python=3.8
conda activate ./3D-Speaker
pip install -r requirements.txt


pip install modelscope
# 配置模型名称并指定wav路径，wav路径可以是单个wav，也可以包含多条wav路径的list文件
model_id=iic/speech_eres2netv2_sv_zh-cn_16k-common
pip install pyannote.audio addict datasets

#测试
python speakerlab/bin/infer_sv.py --model_id $model_id --wavs ../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/reference_1.wav ../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/reference_2.wav
#显示类似的就成功
[INFO]: The similarity score between two input wavs is 0.0431

#离开环境回到voice_gen目录
conda deactivate
cd ..
```


---
当文件中有下面这个标志，说明那部分是可以配置的
`#[可配置]`

##### 打开voice_gen/pre_deal_voicedata-Copy1.ipynb 执行全部
- 1.将200个参考音频中长于30秒的截断成30秒
- 2.将多人声音频剪裁为首个人声目标 后覆盖1 得到 voice_gen/pre_deal/fix

##### 文本修正csv :voice_gen/modify_text_not_only_chinese.csv
这是通过autodl-tmp/voice_gen/check-Copy1.ipynb 中程序（以这个开头的
`#检查不止是中文的文本，量太少了手动改了`
）获得后手动修改，如果数据量过多可以通过使用其他小型语言模型逐个修改

##### 增强和人声分离 [42,44] 合并（可通过说话人确认来判断哪个更优）得到voice_gen/pre_deal/enhancement/mix

```shell
cp ./test2-Copy1.py ./ClearerVoice-Studio/clearvoice/test2-Copy1.py
cd ./ClearerVoice-Studio
conda activate ./ClearerVoice-Studio 
#运行语音增强和人声分离
python ./clearvoice/test2-Copy1.py

#离开环境回到voice_gen目录
conda deactivate
cd ..

#这是小批量的，若是大批量，需要说话人确认比对后进行处理(由于说话人确认是我最后一天才弄的，来不及，而且提交的结果也比较差)
mkdir -p ./pre_deal/enhancement/mix/
cp ./pre_deal/speech_enhancement/MossFormerGAN_SE_16K/* ./pre_deal/enhancement/mix/
cp -f ./pre_deal/speech_separation/MossFormer2_SS_16K/reference_42_s1.wav ./pre_deal/enhancement/mix/reference_42.wav
cp -f ./pre_deal/speech_separation/MossFormer2_SS_16K/reference_44_s1.wav ./pre_deal/enhancement/mix/reference_44.wav

```


##### 说话人确认比较 compare-Copy1.py
```bash
#当前位置是voice_gen
#将生成的人声分离后缀是s1的批量复制到去了s1
for file in ./pre_deal/speech_separation/MossFormer2_SS_16K/*_s1.wav; do
    cp "$file" "./pre_deal/speech_separation/s1/$(basename "$file" _s1.wav).wav"
done
```

```shell
cp ./compare-Copy1.py ./3D-Speaker/compare-Copy1.py
cd 3D-Speaker
conda activate ./3D-Speaker

python compare-Copy1.py

#离开环境回到voice_gen目录
conda deactivate
cd ..

#然后可以查看./input_similarity.csv 对应哪个更优（人声增强和人声分离）

```


##### 📈 基本统计信息:

| 采样方法 | 均值 | 标准差 | 最小值 | 最大值 | Q1 | Q3 | 样本数 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| speechenhan | 0.900 | 0.120 | 0.122 | 0.999 | 0.858 | 0.982 | 200 |
| speechsepar | 0.859 | 0.127 | 0.372 | 0.991 | 0.812 | 0.951 | 200 |
##### 🏆 排名统计 :

| 采样方法 | 较优样本比例 | 样本数 |
| ---- | ---- | ---- |
| speechenhan | 84.5 | 200 |
| speechsepar | 15.5 | 200 |

![[violin_full 1.png]]

##### 归一化（峰值和 峰值响度归一化）voice_gen/post_deal_voicedata-Copy1.ipynb 执行全部
得到
- voice_gen/pre_deal/enhancement/mix_normalized 
- voice_gen/pre_deal/enhancement/mix_dnor

##### 语音克隆 voice_gen/voice_gen_data_deal_new-setting-Copy1.ipynb

主要方法就是调不同的

```python
short_audio_kwargs = {
    "top_p": 0.9,      # 更高的top_p增加多样性但保持质量
    "top_k": 20,       # 更低的top_k减少随机性
    "temperature": 0.90, # 更低的temperature减少随机性，使输出更确定
}
```
然后通过 生成多个输出后，使用确认说话人比较相似度，获取最高的合并在一起，由于是在最后一天才写相似度方面，所以并没有总结调参策略

复现就是，逐个按下面调参修改，修改一下位置为
![[code2.png]]
![[code1.png]]

```python
#1
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/1
base_kwargs = {
    "do_sample": True,
    "top_p": 0.8,
    "top_k": 30,
    "temperature": 1.0,
    "length_penalty": 0.0,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "output_scores": True,
}
short_audio_kwargs = {
    "top_p": 0.8,      # 更高的top_p增加多样性但保持质量
    "top_k": 30,       # 更低的top_k减少随机性
    "temperature": 1.0, # 更低的temperature减少随机性，使输出更确定
}
#2
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/2
base_kwargs = {
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 25,
    "temperature": 0.9,
    "length_penalty": 0.0,
    "num_beams": 20,
    "repetition_penalty": 10.0,
    "output_scores": True,
}
short_audio_kwargs = {
    "top_p": 0.9,      
    "top_k": 20,       
    "temperature": 0.8, 
}
#3
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/3
short_audio_kwargs = {
    "top_p": 0.8,      
    "top_k": 20,       
    "temperature": 0.8, 
}
#4
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/4
short_audio_kwargs = {
    "top_p": 0.8,      
    "top_k": 15,       
    "temperature": 0.8, 
}
#5
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/5
short_audio_kwargs = {
    "top_p": 0.75,      
    "top_k": 15,       
    "temperature": 0.75, 
}
#6
参考音频 ./pre_deal/enhancement/mix_dnor
输出音频 ./new_data/compare/6
short_audio_kwargs = {
    "top_p": 0.9,      
    "top_k": 20,       
    "temperature": 0.80, 
}
缺失的文件编号（2 个）：
[144, 159]

```
最终生成的几组数据都在 voice_gen/new_data/compare/ 
然后需要从新采样来匹配说话人确认模型 使用 voice_gen/resample-Copy1.ipynb
如果输入参数不同需要修正，以及根据数量不同修改

```python
    input_directory = './new_data/compare/1_output/'
    output_directory = './new_data/resample/1_output/'
    batch_resample_directory(input_directory, output_directory, target_sample_rate)
```
采样后，说话人确认比较 compare-Copy2.py

```shell
cp ./compare-Copy2.py ./3D-Speaker/compare-Copy2.py
cd 3D-Speaker
conda activate ./3D-Speaker
```

```python
#有多少组就用多少个
smart_add_comparison_group(
        MODEL_ID,
        "../AISumerCamp_audio_generation_fight/aigc_speech_generation_tasks/", #对比文件目录a
        "../new_data/resample/1_output/", #对比文件目录b
        "reference_",#a目录下文件前缀
        "synthesized_speech_",#b目录下文件前缀
        OUTPUT_CSV,#输出的csv位置
        "1_output",#记录栏名称
        start_idx=1,
        end_idx=200,
        max_workers=20,
        batch_size=100,
        max_retries=3
    )

```

```shell
python compare-Copy2.py

#离开环境回到voice_gen目录
conda deactivate
cd ..

#然后可以查看./output_similarity.csv 对应哪个更优

```
##### 得到最优的音频和统计 autodl-tmp/voice_gen/compare-Copy1.ipynb

```python
#根据需要修改
 FOLDER_MAPPING = {
        '1_output': "./new_data/compare/1_output",
        '2_output': "./new_data/compare/2_output",
        '3_output': "./new_data/compare/3_output",
        '4_output': "./new_data/compare/4_output",
        '5_output': "./new_data/compare/5_output",
        '6_output': "./new_data/compare/6_output",
    }

```
最优音频在voice_gen/best_audio_results
最后运行 autodl-tmp/voice_gen/check-post-Copy1.ipynb 检查是否有漏


```
mkdir result
cp ./best_audio_results/* result/
cp result.csv result/
zip -r result.zip result
```
好了 完成了

给出对比图![[ranking_bar_full.png]]![[ranking_heatmap_full.png]]![[violin_full.png]]

###### 复盘
第一次参加这种比赛，没有设计好实践规划，导致最后一天才匆忙写说话人确认部分比对方面，之前想的方案是：寻找ClearerVoice-Studio和index-tts内部文件是否有类似方法提供后无果搁置了。再加上中途出现了些bug，使得比对只比对了一对音频的结果就提交了。

# 致谢
感谢本次比赛的各个主办方提供了高质量的竞赛平台与组织保障，使我们能够在公平、开放的环境中开展算法研究与实践。感谢 Datawhale 给予我们的帮助与支持，让我们可以更了解赛事与学习。同时，衷心感谢 AutoDL 平台提供的算力资源支持，为模型训练与实验验证提供了重要保障。最后，再次向向各方支持致以最衷心的感谢！
