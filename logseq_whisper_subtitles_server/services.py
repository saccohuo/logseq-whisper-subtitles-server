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

# 检查 CUDA 是否可用，如果可用则强制使用
device = "cuda" if torch.cuda.is_available() else "cpu"
# 为 FunASR 模型设置 device
funasr_device = "cuda:0" if device.startswith("cuda") else device
print(f"Using device: {device}")

print("Loading base whisper model...")
whisper_models = {
    DEFAULT_MODEL_SIZE: whisper.load_model(DEFAULT_MODEL_SIZE).to(device)
}
print("Loading base whisper model done.")
print(f"Model device: {next(whisper_models[DEFAULT_MODEL_SIZE].parameters()).device}")

# 添加一个字典来缓存已加载的 FunASR 模型
funasr_models = {}

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
        'keepvideo': True,
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
            else:
                raise FileNotFoundError(f"Could not find the downloaded audio file: {audio_name}")
        
        return audio_name
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        raise

def replace_punctuation(text):
    """Replace English punctuation with Chinese punctuation
    替换英文标点为中文标点"""
    text = text.replace(",", "，").replace(".", "。").replace("?", "？").replace("!", "！")
    return text

def segment_text_with_ollama(text, model="qwen2.5:3b", max_length=100, ollama_endpoint="http://localhost:11434"):
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

def split_text(text, max_length=1500):
    """
    将长文本切割成不超过指定长度的片段，以句子为单位，保留时间戳
    """
    sentences = re.split(r'(。|！|？|\.|\!|\?)', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        # 检查句子是否以时间戳开头
        timestamp_match = re.match(r'(\{\{timestamp \d+\}\})', sentence)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            sentence_content = sentence[len(timestamp):]
        else:
            timestamp = ""
            sentence_content = sentence

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = timestamp + sentence_content  # 确保新chunk以时间戳开头（如果有的话）

    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def segment_text_with_openai(text, max_length=100, api_keys=None, models=None, api_endpoints=None, priority=None, enable_rotation=False):
    """
    使用 OpenAI API 对文本进行分段，支持多组 API 设置和轮询
    """
    print("Entering segment_text_with_openai function")
    print(f"API Keys: {api_keys}")
    print(f"Models: {models}")
    print(f"API Endpoints: {api_endpoints}")
    print(f"Priority: {priority}")
    print(f"Enable Rotation: {enable_rotation}")
    
    if not api_keys or not models or not api_endpoints or not priority:
        print("Warning: Some OpenAI API settings are missing.")
        return [text], None  # 返回原始文本作为单个段落

    # 创建一个优先级到字母的映射
    priority_letters = {i: chr(65 + i) for i in range(5)}  # A, B, C, D, E
    
    # 创建一个优先级到索引的映射
    priority_map = {int(p): i for i, p in enumerate(priority) }
    
    # 根据优先级排序 API 设置
    valid_settings = list(zip(range(1, len(api_keys) + 1), api_keys, models, api_endpoints))
    api_settings = sorted(valid_settings, key=lambda x: priority_map.get(x[0], len(priority)))
    
    print(f"Sorted API settings: {api_settings}")

    chunks = split_text(text)
    segmented_chunks = []
    total_chunks = len(chunks)
    rotation_message = None

    def clean_text(text):
        # 移除时间戳和标点，只保留字母数字字符
        return ''.join(char for char in re.sub(r'\{\{timestamp \d+\}\}', '', text) if char.isalnum())

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{total_chunks} ({i/total_chunks*100:.2f}%)")
        print(f"Original chunk: {chunk}")
        
        prompt = f"""
        将以下文本分成多个段落，每个段落的长度约为{max_length}个字。
        遵循以下规则：
        1. 保持原文的所有内容，绝对不要删除、增加或修改任何文字。
        2. 只在自然的句子边界进行分段。
        3. 确保每个段落的意思是完整的。
        4. 直接返回分段后的文本，每个段落用一个换行符分隔。
        5. 不要添加任何额外的解释、编号或标记。
        6. 分段前后的标点符号可以不同，但文字必须完全一致，不允许有任何差异。
        7. 保留每个段落开头的时间戳标记（形如 {{{{timestamp 123}}}}），但删除段落中间的时间戳标记。

        原文本：
        {chunk}
        """

        success = False
        for attempt, (setting_number, api_key, model, api_endpoint) in enumerate(api_settings, 1):
            priority_letter = priority_letters[attempt - 1]  # A for first attempt, B for second, etc.
            try:
                client = OpenAI(api_key=api_key, base_url=api_endpoint)
                print(f"Attempt {attempt}: Sending request to OpenAI API (Priority: {priority_letter}, OpenAI Setting: {setting_number}, Endpoint: {api_endpoint}, Model: {model})")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a text segmentation assistant. Only return the segmented text without any additional information."},
                        {"role": "user", "content": prompt}
                    ]
                )
                print(f"Received response from OpenAI API using model: {model}")
                segmented_chunk = response.choices[0].message.content.strip()
                
                # 处理返回的文本，只保留段落开头的时间戳
                processed_segments = []
                for segment in segmented_chunk.split('\n'):
                    match = re.match(r'(\{\{timestamp \d+\}\})(.*)', segment, re.DOTALL)
                    if match:
                        timestamp, content = match.groups()
                        content_without_timestamps = re.sub(r'\{\{timestamp \d+\}\}', '', content)
                        processed_segments.append(f"{timestamp}{content_without_timestamps}")
                    else:
                        processed_segments.append(segment)
                
                # 验证分段前后的文本是否完全一致（忽略时间戳和标点）
                original_text_no_punct = clean_text(chunk)
                segmented_text_no_punct = clean_text(''.join(processed_segments))
                
                print(f"Original text (cleaned): {original_text_no_punct}")
                print(f"Segmented text (cleaned): {segmented_text_no_punct}")
                print(f"Segmented text (with formatting):")
                for seg in processed_segments:
                    print(seg)
                
                if original_text_no_punct == segmented_text_no_punct:
                    print(f"Chunk {i}: Segmentation successful, text content matches")
                    segmented_chunks.extend(processed_segments)
                    success = True
                    rotation_message = f"Using OpenAI API setting {setting_number} (Priority {priority_letter}) with model {model}"
                    break
                else:
                    print(f"Chunk {i}: Warning - Segmented text does not match original text")
                    if not enable_rotation:
                        break
                    print("Rotation enabled, trying next API setting")
            except Exception as e:
                print(f"Chunk {i}, Attempt {attempt}: OpenAI API 调用失败 (Priority: {priority_letter}, OpenAI Setting: {setting_number}, Endpoint: {api_endpoint}, Model: {model}): {str(e)}")
                if not enable_rotation:
                    break
                print("Rotation enabled, trying next API setting")

        if not success:
            print(f"All API calls failed for chunk {i}/{total_chunks}. Using original chunk.")
            segmented_chunks.append(chunk)

    # 验证整体分段前后的文本是否完全一致（忽略时间戳和标点）
    original_text = clean_text(text)
    segmented_text = clean_text(''.join(segmented_chunks))
    
    print("Final comparison:")
    print(f"Original text (full, cleaned): {original_text}")
    print(f"Segmented text (full, cleaned): {segmented_text}")
    
    if original_text == segmented_text:
        print("Overall segmentation successful, text content matches")
        return segmented_chunks, rotation_message
    else:
        print("Warning: Overall segmented text does not match original text")
        return [text], rotation_message  # 如果不一致，返回原始文本作为单个段落

def transcribe_audio(audio_path, min_length=DEFAULT_MIN_LENGTH, model_type="whisper", model_size=DEFAULT_MODEL_SIZE, zh_type='zh-cn', funasr_model_name=None, funasr_model_source=None, segment_model="ollama", ollama_model="qwen2.5:3b", ollama_endpoint="http://localhost:11434", openai_api_keys=None, openai_models=None, openai_api_endpoints=None, openai_priority=None, enable_openai_rotation=False):
    """Transcribe audio file
    转录音频文件"""
    print("Entering transcribe_audio function")
    print(f"Model Type: {model_type}")
    print(f"Segment Model: {segment_model}")
    print(f"OpenAI API Keys: {openai_api_keys}")
    print(f"OpenAI Models: {openai_models}")
    print(f"OpenAI API Endpoints: {openai_api_endpoints}")
    print(f"OpenAI Priority: {openai_priority}")
    print(f"Enable OpenAI Rotation: {enable_openai_rotation}")
    
    if not min_length:
        min_length = DEFAULT_MIN_LENGTH

    if model_type == "whisper":
        if not model_size:
            model_size = DEFAULT_MODEL_SIZE

        if model_size not in whisper_models:
            print(f"Loading {model_size} whisper model...")
            whisper_models[model_size] = whisper.load_model(model_size).to(device)

        model = whisper_models[model_size]

        print("Using Whisper model: ", model_size)
        print(f"Model device: {next(model.parameters()).device}")

        if zh_type.strip() == 'zh-cn':
            print("Transcribing Chinese simplified audio ...")
            transcribe = model.transcribe(audio=audio_path, verbose=True, initial_prompt="对于普通话句子，以中文简体输出", fp16=(device=="cuda"))
        else:
            transcribe = model.transcribe(audio=audio_path, verbose=True, fp16=(device=="cuda"))

        segments = transcribe['segments']
        detect_language = transcribe.get('language', '')

        res = []
        for segment in segments:
            start_time = int(float(segment['start']))
            text = segment['text'].strip()
            res.append({
                "start": start_time,
                "text": text
            })
        return res, None  # 返回结果和 None 作为 rotation_message

    elif model_type == "funasr":
        print(f"Using FunASR model: {funasr_model_name}")
        model = load_funasr_model(funasr_model_name, funasr_model_source)
        res = model.generate(
            input=audio_path,
            batch_size_s=300,
        )
        
        print("FunASR 生成结果:")
        # print(json.dumps(res, indent=2, ensure_ascii=False))
        
        full_text_with_timestamp = ""
        full_text_without_timestamp = ""
        
        for seg in res:
            if 'sentence_info' in seg:
                for sentence in seg['sentence_info']:
                    start_time_seconds = int(sentence["start"] / 1000)  # 将毫秒转换为整数秒
                    text = rich_transcription_postprocess(sentence["text"])
                    full_text_with_timestamp += f"{{{{timestamp {start_time_seconds}}}}} {text}\n"
                    full_text_without_timestamp += f"{text}\n"
            else:
                start_time_seconds = int(float(seg.get("start", 0)) / 1000)  # 将毫秒转换为整数秒
                text = rich_transcription_postprocess(seg["text"])
                full_text_with_timestamp += f"{{{{timestamp {start_time_seconds}}}}} {text}\n"
                full_text_without_timestamp += f"{text}\n"
        
        print("完整转录文本（带时间戳）:")
        print(full_text_with_timestamp)
        print("完整转录文本（不带时间戳）:")
        print(full_text_without_timestamp)
        
        # 使用选定的模型进行分段
        if segment_model == "openai":
            segments, rotation_message = segment_text_with_openai(full_text_with_timestamp, api_keys=openai_api_keys, models=openai_models, api_endpoints=openai_api_endpoints, priority=openai_priority, enable_rotation=enable_openai_rotation)
        else:
            segments = segment_text_with_ollama(full_text_with_timestamp, model=ollama_model, ollama_endpoint=ollama_endpoint)
            rotation_message = None
        
        # 处理分段后的文本，提取时间戳
        result = []
        for segment in segments:
            match = re.match(r'\{\{timestamp (\d+)\}\} (.+)', segment.strip())
            if match:
                start_time = int(match.group(1))
                text = match.group(2).strip()
                if text:  # 只有当文本非空时才添加到结果中
                    result.append({
                        "start": start_time,
                        "text": text
                    })
            else:
                print(f"Warning: Could not find timestamp for segment: {segment}")
        
        # 验证分段前后的文本是否一致（忽略时间戳和标点）
        def clean_text(text):
            return ''.join(char for char in re.sub(r'\{\{timestamp \d+\}\}', '', text) if char.isalnum())

        original_text = clean_text(full_text_without_timestamp)
        segmented_text = clean_text(''.join([item['text'] for item in result]))
        
        if original_text != segmented_text:
            print("Warning: Segmented text does not match original text")
            print("Original text (cleaned):", original_text)
            print("Segmented text (cleaned):", segmented_text)
            print("Using original text without segmentation")
            # 如果不一致，使用原始文本
            result = []
            for line in full_text_with_timestamp.split('\n'):
                match = re.match(r'\{\{timestamp (\d+)\}\} (.+)', line.strip())
                if match:
                    start_time = int(match.group(1))
                    text = match.group(2).strip()
                    if text:
                        result.append({
                            "start": start_time,
                            "text": text
                        })
        
        print("最终分段结果:")
        for item in result:
            print(f"{{{{timestamp {item['start']}}}}} {item['text']}")
        
        return result, rotation_message

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

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

# 预加载常用的 FunASR 模型
def preload_funasr_models():
    model = "paraformer-zh"
    try:
        load_funasr_model(model, "modelscope")
        print(f"Preloaded FunASR model: {model}")
        print("FunASR model preloading completed.")
    except Exception as e:
        print(f"Failed to preload FunASR model {model}: {str(e)}")

# 在服务器启动时调用这个函数
preload_funasr_models()