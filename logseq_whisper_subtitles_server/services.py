import yt_dlp
from datetime import timedelta
import whisper
import uuid
import os
import subprocess
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from huggingface_hub import hf_hub_download
import requests
import json
from openai import OpenAI
import re
import time
import pysrt
import ass

# 定义常量
EN_SEGMENT_SYMBOLS = ['.', '?', '!']
PUNCTUATION = ['，', '。', '？', '！']
DEFAULT_MIN_LENGTH = 100  # set to 0 to disable merging Segments
DEFAULT_MODEL_SIZE = "base"

# FunASR 模型列表
FUNASR_MODELS = [
    "SenseVoiceSmall", "paraformer-zh", "paraformer-zh-streaming", "paraformer-en",
    "conformer-en", "ct-punc", "fsmn-vad", "fsmn-kws", "fa-zh", "cam++",
    "Qwen-Audio", "Qwen-Audio-Chat", "emotion2vec+large"
]

# 添加诊断信息
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 检查 CUDA 是否可用，如可用则强制使用
device = "cuda" if torch.cuda.is_available() else "cpu"
# 为 FunASR 模型设置 device
funasr_device = "cuda:0" if device.startswith("cuda") else device
print(f"Using device: {device}")

# 将 Whisper 模型加载移动到函数中
whisper_models = {}

def load_whisper_model(model_size=DEFAULT_MODEL_SIZE):
    if model_size not in whisper_models:
        print(f"Loading {model_size} whisper model...")
        whisper_models[model_size] = whisper.load_model(model_size).to(device)
        print(f"Loading {model_size} whisper model done.")
        print(f"Model device: {next(whisper_models[model_size].parameters()).device}")
    return whisper_models[model_size]

# 添加一个字典来缓存已加载的 FunASR 模型
funasr_models = {}

# 在文件顶部添加这个函数
def clean_text(text):
    """
    移除时间戳和标点，只保留字母数字字符
    """
    return ''.join(char for char in re.sub(r'\{\{timestamp \d+\}\}', '', text) if char.isalnum())

def load_funasr_model(model_name, model_source):
    """Load FunASR model
    加载 FunASR 模型"""
    if model_name not in funasr_models:
        print(f"Loading FunASR model: {model_name} from {model_source}")
        try:
            # 使用官方示例中的参数，并添加 disable_update=True
            funasr_models[model_name] = AutoModel(
                model=model_name,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 60000},
                punc_model="ct-punc",
                spk_model="cam++",  # 如果需要说话人分离，可以取消注释
                device=funasr_device,
                disable_update=True  # 禁用更新检查
            )
            print(f"Successfully loaded FunASR model: {model_name}")
        except Exception as e:
            print(f"Error loading FunASR model: {str(e)}")
            raise
    return funasr_models[model_name]

def is_audio_file(filename):
    """Check if the file is an audio file
    检查文件是否为音频文件"""
    audio_extensions = ['.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a', '.wma']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in audio_extensions

def extract_audio_from_local_video(video_path):
    """Extract audio from a local video file
    从本地视频文件中提取音频"""
    audio_output_path = os.path.join('local', f'local_audio_{uuid.uuid4().hex}.mp3')
    if not os.path.exists('local'):
        os.makedirs('local')
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        '-vn',
        audio_output_path
    ]
    try:
        print("Converting local video to audio ...")
        subprocess.run(command)
        print("Converting local video to audio done.")
    except subprocess.CalledProcessError as e:
        print("Converting local video to audio failed.")
        raise RuntimeError(f"Failed to convert local video to audio: {e.stderr.decode()}") from e

    return audio_output_path

def download_video(video_url):
    """Download video and extract audio
    下载视频并提取音频"""
    print(f"Downloading the video: {video_url} into audio ...")
    vid = uuid.uuid4().hex
    if not os.path.exists('videos'):
        os.makedirs('videos')
    audio_name = os.path.join('videos', f'video_audio_{vid}.mp3')

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': audio_name,
        'keepvideo': False,
        'postprocessor_args': [
            '-ar', '16000'
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Downloading the video: {video_url} into audio done.")
        
        if not os.path.exists(audio_name):
            possible_name = audio_name + '.mp3'
            if os.path.exists(possible_name):
                os.rename(possible_name, audio_name)
                print(f"Renamed {possible_name} to {audio_name}")
            else:
                raise FileNotFoundError(f"Could not find the downloaded audio file: {audio_name}")
        
        print(f"Audio file path: {audio_name}")
        print(f"Audio file exists: {os.path.exists(audio_name)}")
        print(f"Audio file size: {os.path.getsize(audio_name)} bytes")
        
        return os.path.abspath(audio_name)
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        raise

def replace_punctuation(text):
    """Replace English punctuation with Chinese punctuation
    替换英文标点为中文标点"""
    text = text.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
    return text

def segment_text_with_ollama(text, model="qwen2.5:3b", max_length=1500, ollama_endpoint="http://localhost:11434"):
    """
    使用 Ollama 的模型对文本进行分段
    """
    print("原始文本:")
    print(text)
    print(f"文本长度: {len(text)}")

    prompt = f"""
    请针对在```符号中的文本做分段，不要增加或删减，每个段落的长度大约为{max_length}个字符。确保每个段落的意思是完整的，每个段落用一个换行符分隔。分段前后的标点符号可以不同，但文字必须完全一致，不允许有任何差异。

    ```
    {text}
    ```
    """

    print(f"发送请求到 Ollama API: {ollama_endpoint}")
    response = requests.post(f'{ollama_endpoint}/api/generate', 
                             json={
                                 "model": model,
                                 "prompt": prompt,
                                 "stream": False
                             })
    
    if response.status_code == 200:
        result = response.json()
        segmented_text = result['response'].strip()
        
        print("Ollama 返回的分段文本:")
        print(segmented_text)
        
        # 验证分段前后的文本是否完全一致
        original_text = ''.join(char for char in text if char.isalnum())
        segmented_text_no_punct = ''.join(char for char in segmented_text if char.isalnum())
        
        if original_text == segmented_text_no_punct:
            print("分段成功，文本内容完全一致")
            return segmented_text.split('\n')
        else:
            print("警告：分段后的文本与原文本不一致")
            print("原始文本（无标点）:")
            print(original_text)
            print("分段后文本（无标点）:")
            print(segmented_text_no_punct)
            return [text]  # 如果不一致，返回原始文本作为单个段落
    else:
        print(f"Ollama API 调用失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return [text]  # 如果 API 调用失败，返回原始文本作为单个段落

def split_text(text, max_length):
    """
    将长文本切割成不超过指定长度的片段，在时间戳之前进行分割
    """
    # 使用正则表达式匹配时间戳和文本内容
    chunks = re.findall(r'(.*?)(\{\{timestamp \d+\}\}.*?)(?=\{\{timestamp \d+\}\}|$)', text, re.DOTALL)
    
    result = []
    current_chunk = ""
    for previous_text, timestamped_text in chunks:
        if len(current_chunk) + len(previous_text) + len(timestamped_text) <= max_length:
            current_chunk += previous_text + timestamped_text
        else:
            if current_chunk:
                result.append(current_chunk.strip())
            current_chunk = timestamped_text
        
        # 如果当前块已经超过最大长度，立即添加到结果中
        if len(current_chunk) > max_length:
            result.append(current_chunk.strip())
            current_chunk = ""

    if current_chunk:
        result.append(current_chunk.strip())

    return result

def segment_text_with_openai(text, **kwargs):
    print("Entering segment_text_with_openai function")
    print(f"API Keys: {kwargs.get('api_keys')}")
    print(f"Models: {kwargs.get('models')}")
    print(f"API Endpoints: {kwargs.get('api_endpoints')}")
    print(f"Priority: {kwargs.get('priority')}")
    print(f"Enable Rotation: {kwargs.get('enable_rotation')}")
    print(f"Max Length: {kwargs.get('max_length')}")
    print(f"Max Segment Lengths: {kwargs.get('max_segment_lengths')}")
    print(f"Original full text: {text}")
    
    if not kwargs.get('api_keys') or not kwargs.get('models') or not kwargs.get('api_endpoints') or not kwargs.get('priority'):
        print("Warning: Some OpenAI API settings are missing.")
        return [{"start": 0, "text": text}], None

    # 创建一个优先级到字母的映射
    priority_letters = {i: chr(65 + i) for i in range(5)}  # A, B, C, D, E
    
    # 创建一个优先级到索引的映射
    priority_map = {int(p): i for i, p in enumerate(kwargs.get('priority'))}
    
    # 根据优先级排序 API 设置
    valid_settings = list(zip(range(1, len(kwargs.get('api_keys')) + 1), kwargs.get('api_keys'), kwargs.get('models'), kwargs.get('api_endpoints')))
    api_settings = sorted(valid_settings, key=lambda x: priority_map.get(x[0], len(kwargs.get('priority'))))
    
    print(f"Sorted API settings: {api_settings}")

    # 根据是否启用轮询来决定使用哪个长度进行分割
    if kwargs.get('enable_rotation'):
        split_length = kwargs.get('max_length', 1500)
        print(f"Using max_length for splitting: {split_length}")
    else:
        # 使用api_settings中第一组（优先级最高）的max_segment_length
        first_setting = api_settings[0] if api_settings else None
        if first_setting and len(first_setting) > 0:
            setting_number = first_setting[0]
            split_length = kwargs.get('max_segment_lengths', [])[setting_number - 1] if kwargs.get('max_segment_lengths') and setting_number <= len(kwargs.get('max_segment_lengths')) else kwargs.get('max_length', 1500)
        else:
            split_length = kwargs.get('max_length', 1500)
        print(f"Using max_segment_length for splitting: {split_length}")

    chunks = split_text(text, split_length)
    print(f"Number of chunks after splitting: {len(chunks)}")
    
    if not chunks:
        print("Warning: No chunks to process. Returning original text.")
        return [{"start": 0, "text": text}], None

    segmented_chunks = []
    total_chunks = len(chunks)
    rotation_message = None

    def is_segmentation_acceptable(original, segmented, tolerance, unit):
        original_length = len(original)
        segmented_length = len(segmented)
        if original_length == 0:
            return True, 0  # 如果原文为空，认为分段是可接受的
        if unit == "percent":
            difference_percent = abs(original_length - segmented_length) / original_length * 100
            return difference_percent <= tolerance, difference_percent
        else:  # characters
            difference_chars = abs(original_length - segmented_length)
            return difference_chars <= tolerance, difference_chars

    latest_segment_last_timestamp = 0
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{total_chunks}")
        print(f"Original chunk text: {chunk}")
        
        for attempt, (setting_number, api_key, model, api_endpoint) in enumerate(api_settings, 1):
            print(f"Using max length for setting {setting_number}: {split_length}")
            prompt = f"""
            将以下文本分成多个段落。遵循以下规则：
            1. 保持原文的所有内容，绝对不要删除、增加或修改任何文字。
            2. 只在自然的句子边界进行分段。
            3. 确保每个段落的意思是完整的。
            4. 每个段落的字数一般应在50~150字，有特殊情况也可以根据情况动态调整。
            5. 直接返回分段后的文本，每个段落用一个换行符分隔。
            6. 不要添加任何额外的解释、编号或标记。
            7. 分段前后的标点符号可以不同，但文字必须完全一致，不允许有任何差异。
            8. 请务必保留每个段落开头的时间戳标记（形如 {{{{timestamp 123}}}}），但删除段落中间的时间戳标记。
            9. 如果段落开头没有时间戳，根据前后邻近的时间戳，计算出当前段落的时间戳，并添加到段落开头。

            原文本：
            {chunk}
            """

            success = False
            priority_letter = priority_letters[attempt - 1]
            print(f"Attempt {attempt}: Sending request to OpenAI API (Priority: {priority_letter}, OpenAI Setting: {setting_number}, Endpoint: {api_endpoint}, Model: {model})")
            
            try:
                client = OpenAI(api_key=api_key, base_url=api_endpoint)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a text segmentation assistant. Only return the segmented text without any additional information."},
                        {"role": "user", "content": prompt}
                    ]
                )
                print(f"Received response from OpenAI API using model: {model}")
                segmented_chunk = response.choices[0].message.content.strip()
                print(f"Segmented chunk: {segmented_chunk}")
                
                processed_segments = process_segments_with_timestamps(segmented_chunk,latest_segment_last_timestamp)
                
                print(f"Segmented chunk text: {processed_segments}")
                
                original_text_no_punct = clean_text(chunk)
                segmented_text_no_punct = clean_text(''.join([seg["text"] for seg in processed_segments]))
                
                is_acceptable, difference = is_segmentation_acceptable(original_text_no_punct, segmented_text_no_punct, kwargs.get('segmentation_tolerance'), kwargs.get('segmentation_tolerance_unit'))
                
                if is_acceptable:
                    print(f"Chunk {i}: Segmentation successful, within tolerance")
                    print(f"Difference: {difference} {'%' if kwargs.get('segmentation_tolerance_unit') == 'percent' else 'characters'}")
                    segmented_chunks.extend(processed_segments)
                    success = True
                    rotation_message = f"Using OpenAI API setting {setting_number} (Priority {priority_letter}) with model {model}"
                    break
                else:
                    print(f"Chunk {i}: Warning - Segmented text difference exceeds tolerance")
                    print(f"Difference: {difference} {'%' if kwargs.get('segmentation_tolerance_unit') == 'percent' else 'characters'}")
                    if not kwargs.get('enable_rotation'):
                        break
                    print("Rotation enabled, trying next API setting")
            except Exception as e:
                print(f"Chunk {i}, Attempt {attempt}: OpenAI API 调用失败 (Priority: {priority_letter}, OpenAI Setting: {setting_number}, Endpoint: {api_endpoint}, Model: {model}): {str(e)}")
                if not kwargs.get('enable_rotation'):
                    break
                print("Rotation enabled, trying next API setting")

        if not success:
            print(f"All API calls failed or exceeded tolerance for chunk {i}/{total_chunks}. Using original chunk.")
            segmented_chunks.extend(process_segments_with_timestamps(chunk,latest_segment_last_timestamp))
        
        latest_segment_last_timestamp = segmented_chunks[-1]['start']

    # 用于日志显示的整体比较
    original_text = clean_text(text)
    segmented_text = clean_text(''.join([seg["text"] for seg in segmented_chunks]))
    
    if not original_text:
        print("Warning: Original text is empty after cleaning.")
        return segmented_chunks, rotation_message

    is_overall_acceptable, overall_difference = is_segmentation_acceptable(original_text, segmented_text, kwargs.get('segmentation_tolerance'), kwargs.get('segmentation_tolerance_unit'))
    
    print("\nFinal comparison (for logging purposes only):")
    print(f"Original full text length: {len(original_text)}")
    print(f"Segmented full text length: {len(segmented_text)}")
    print(f"Overall difference: {overall_difference} {'%' if kwargs.get('segmentation_tolerance_unit') == 'percent' else 'characters'}")
    print(f"Overall segmentation {'within' if is_overall_acceptable else 'exceeds'} tolerance")

    return segmented_chunks, rotation_message

def process_transcription(audio_path, transcribe_func, post_process_func=None, **kwargs):
    """
    通用的转录处理函数
    
    :param audio_path: 音频文件路径
    :param transcribe_func: 转录函数（Whisper 或 FunASR）
    :param post_process_func: 后处理函数（可选）
    :param kwargs: 其他参数
    :return: 处理后的转录结果
    """
    print(f"Processing audio file: {audio_path}")
    
    # 转录
    transcription = transcribe_func(audio_path, **kwargs)
    
    # 如果有后处理函数，执行后处理
    if post_process_func:
        transcription = post_process_func(transcription)
    
    # 处理结果
    result = []
    if isinstance(transcription, dict) and 'segments' in transcription:
        # Whisper 格式
        for segment in transcription['segments']:
            start_time = int(segment['start'])
            text = segment['text'].strip()
            if text:
                result.append({
                    "start": start_time,
                    "text": text
                })
    else:
        # FunASR 格式
        for segment in transcription:
            if isinstance(segment, dict):
                if 'sentence_info' in segment:
                    # 处理包含 sentence_info 的情况
                    for sentence in segment['sentence_info']:
                        start_time_seconds = int(float(sentence.get("start", 0)) / 1000)  # 转换为秒
                        text = sentence["text"].strip()
                        if text:
                            result.append({
                                "start": start_time_seconds,
                                "text": text
                            })
                elif 'text' in segment:
                    # 处理包含 text 的情况
                    start_time_seconds = int(float(segment.get("start", 0)) / 1000)  # 转换为秒
                    text = segment["text"].strip()
                    if text:
                        result.append({
                            "start": start_time_seconds,
                            "text": text
                        })
            elif isinstance(segment, str):
                # 如果 segment 是字符串，假设它是纯文本，没有时间戳
                result[-1]['text'] += segment.strip()
    
    return result

def whisper_transcribe(audio_path, model_size=DEFAULT_MODEL_SIZE, **kwargs):
    """
    Whisper 转录函数
    """
    model = load_whisper_model(model_size)
    print(f"Transcribing with Whisper model: {model_size}")
    transcribe = model.transcribe(audio=audio_path, **kwargs)
    return transcribe

def funasr_transcribe(audio_path, model, hotword_file_path='', hotwords='', **kwargs):
    """
    FunASR 转录函数
    """
    print(f"Transcribing with FunASR model: {model}")
    
    # 处理热词
    hotword_list = []
    if hotword_file_path:
        try:
            with open(hotword_file_path, 'r', encoding='utf-8') as f:
                hotword_list.extend(f.read().splitlines())
        except Exception as e:
            print(f"Error reading hotword file: {str(e)}")
    
    if hotwords:
        hotword_list.extend(hotwords.split(','))
    
    hotword_list = list(set(hotword_list))  # 去重
    
    if hotword_list:
        print(f"Using hotwords: {hotword_list}")
        kwargs['hotword'] = ' '.join(hotword_list)  # 将热词列表转换为空格分隔的字符串
    
    print(f"FunASR generate kwargs: {kwargs}")  # 添加这行来打印所有参数
    res = model.generate(input=audio_path, **kwargs)
    # print(f"FunASR transcription result: {res}")  # 添加这行来打印原始结果
    return res

def segment_text(text, segment_model, segmentation_params):
    if segment_model == 'openai':
        return segment_text_with_openai(
            text,
            **segmentation_params
        )
    elif segment_model == 'ollama':
        segments = segment_text_with_ollama(
            text,
            model=segmentation_params['ollama_model'],
            max_length=segmentation_params['max_length'],
            ollama_endpoint=segmentation_params['ollama_endpoint']
        )
        return segments, None
    else:
        raise ValueError(f"Unsupported segment model: {segment_model}")

def transcribe_audio(audio_path, min_length=DEFAULT_MIN_LENGTH, model_type="whisper", model_size=DEFAULT_MODEL_SIZE, zh_type='zh-cn', funasr_model_name=None, funasr_model_source=None, segment_model="ollama", perform_segmentation=False, hotword_file_path='', hotwords='', **segmentation_params):
    """Transcribe audio file
    转录音频文件"""
    print("Entering transcribe_audio function")
    print(f"Model Type: {model_type}")
    print(f"Segment Model: {segment_model}")
    print(f"Perform Segmentation: {perform_segmentation}")
    print(f"OpenAI API Keys: {segmentation_params.get('api_keys')}")
    print(f"OpenAI Models: {segmentation_params.get('models')}")
    print(f"OpenAI API Endpoints: {segmentation_params.get('api_endpoints')}")
    print(f"OpenAI Priority: {segmentation_params.get('priority')}")
    print(f"Enable OpenAI Rotation: {segmentation_params.get('enable_rotation')}")
    print(f"Max Length: {segmentation_params.get('max_length')}")

    if not min_length:
        min_length = DEFAULT_MIN_LENGTH

    if model_type == "whisper":
        if not model_size:
            model_size = DEFAULT_MODEL_SIZE

        kwargs = {
            'verbose': True,
            'fp16': (device=="cuda")
        }
        if zh_type.strip() == 'zh-cn':
            print("Transcribing Chinese simplified audio ...")
            kwargs['initial_prompt'] = "对于普通话句子，以中文简体输出"

        result = process_transcription(audio_path, whisper_transcribe, model_size=model_size, **kwargs)

    elif model_type == "funasr":
        if not funasr_model_name:
            funasr_model_name = "paraformer-zh"
        if not funasr_model_source:
            funasr_model_source = "modelscope"

        funasr_model = load_funasr_model(funasr_model_name, funasr_model_source)
        result = process_transcription(audio_path, funasr_transcribe, model=funasr_model, hotword_file_path=hotword_file_path, hotwords=hotwords)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    full_text_with_timestamp = ""
    for segment in result:
        start_time_seconds = segment['start']
        text = segment['text']
        full_text_with_timestamp += f"{{{{timestamp {start_time_seconds}}}}} {text}\n"
    
    if perform_segmentation:
        segmented_result, rotation_message = segment_text(full_text_with_timestamp, segment_model, segmentation_params)
        
        # 确保分段结果是正确的格式
        final_result = []
        for segment in segmented_result:
            if isinstance(segment, dict) and 'start' in segment and 'text' in segment:
                final_result.append(segment)
            elif isinstance(segment, str):
                # 如果是字符串，尝试解析时间戳和文本
                match = re.match(r'\{\{timestamp (\d+)\}\} (.*)', segment)
                if match:
                    start_time = int(match.group(1))
                    text = match.group(2)
                    final_result.append({"start": start_time, "text": text})
                else:
                    print(f"Warning: Unable to parse segment: {segment}")
        
        result = final_result
    else:
        rotation_message = None

    print("最终分段结果:")
    for item in result:
        print(f"{{{{timestamp {item['start']}}}}} {item['text']}")
    
    return result, rotation_message

# 新增函数：获取可用的 Ollama 模型
def get_ollama_models(ollama_endpoint="http://localhost:11434"):
    try:
        response = requests.get(f'{ollama_endpoint}/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
        else:
            print(f"获取 Ollama 模型列表失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"获取 Ollama 模型列表时出错: {str(e)}")
        return []

def preload_funasr_models():
    """
    预加载常用的 FunASR 模型
    """
    print("Preloading FunASR models...")
    models_to_preload = ["paraformer-zh"]  # 可以根据需要添加更多模型
    for model in models_to_preload:
        try:
            load_funasr_model(model, "modelscope")
            print(f"Successfully preloaded FunASR model: {model}")
        except Exception as e:
            print(f"Failed to preload FunASR model {model}: {str(e)}")
    print("FunASR model preloading completed.")

# 预加载常用的 Whisper 和 FunASR 模型
def preload_models():
    # 预加载 Whisper 模型
    load_whisper_model(DEFAULT_MODEL_SIZE)
    
    # 预加载 FunASR 模型
    preload_funasr_models()
    
    print("Model preloading completed.")

# 在服务器启动时调用这个函数
# preload_models()

def convert_subtitle_to_transcription(subtitle_path):
    _, ext = os.path.splitext(subtitle_path)
    ext = ext.lower()

    if ext == '.srt':
        return convert_srt(subtitle_path)
    elif ext == '.ass':
        return convert_ass(subtitle_path)
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}")

def convert_srt(subtitle_path):
    subs = pysrt.open(subtitle_path)
    segments = []
    for sub in subs:
        segments.append({
            "start": sub.start.ordinal // 1000,  # Convert to seconds
            "text": sub.text.replace('\n', ' ')
        })
    return {"segments": segments}

def convert_ass(subtitle_path):
    with open(subtitle_path, encoding='utf-8-sig') as f:
        ass_file = ass.parse(f)
    
    segments = []
    for event in ass_file.events:
        if event.is_comment:
            continue
        segments.append({
            "start": int(event.start.total_seconds()),
            "text": event.text
        })
    return {"segments": segments}

def process_segments_with_timestamps(text,latest_segment_last_timestamp):
    """
    处理带有时间戳的文本段落，返回标准化的段落列表
    """
    result = []
    lines = text.split('\n')
    first_timestamp = None
    for line in lines:
        if line.strip():  # 确保行非空
            match = re.match(r'(\{\{timestamp (\d+)\}\})(.*)', line, re.DOTALL)
            if match:
                timestamp, start_time, content = match.groups()
                content_without_timestamps = re.sub(r'\{\{timestamp \d+\}\}', '', content)
                result.append({"start": int(start_time), "text": content_without_timestamps.strip()})
                if first_timestamp is None:
                    first_timestamp = int(start_time)
            else:
                if result:
                    # 如果没有匹配的timestamp，将文本追加到上一个segment
                    result[-1]["text"] += " " + line.strip()
                elif first_timestamp is not None:
                    # 如果是第一个没有timestamp的segment，使用第一个找到的timestamp
                    result.append({"start": first_timestamp, "text": line.strip()})
                else:
                    # 如果是第一个segment且没有timestamp，创建一个新的segment
                    result.append({"start": 0, "text": line.strip()})
    
    # 如果第一行没有时间戳，但后面找到了时间戳，更新第一个段落的start
    if result and result[0]["start"] == 0:
        if first_timestamp is not None:
            result[0]["start"] = first_timestamp
        else:
            result[0]["start"] = latest_segment_last_timestamp
    
    return result