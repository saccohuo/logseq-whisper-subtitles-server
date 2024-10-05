from flask import Flask, request, jsonify
from services import transcribe_audio, download_video, extract_audio_from_local_video, is_audio_file, FUNASR_MODELS, get_ollama_models, preload_models, convert_subtitle_to_transcription, segment_text_with_openai, segment_text_with_ollama, segment_text, process_segments_with_timestamps, summarize_text

import re
import os
import traceback
import json

app = Flask(__name__)

def extract_segmentation_params(request_form):
    segmentation_params = {
        'max_length': int(request_form.get('max_length', 1500)),
        'max_segment_lengths': [int(request_form.get(f'openai_max_segment_length{i}', 0)) for i in range(1, 6)],
        'api_keys': [request_form.get(f'openai_api_key{i}', '') for i in range(1, 6)],
        'models': [request_form.get(f'openai_models{i}', '') for i in range(1, 6)],
        'api_endpoints': [request_form.get(f'openai_api_endpoint{i}', '') for i in range(1, 6)],
        'priority': request_form.get('openai_priority', '1,2,3,4,5').split(','),
        'enable_rotation': request_form.get('enable_openai_rotation', 'false').lower() == 'true',
        'segmentation_tolerance': float(request_form.get('segmentation_tolerance', 5)),
        'segmentation_tolerance_unit': request_form.get('segmentation_tolerance_unit', 'percent'),
        'ollama_model': request_form.get('ollama_model', 'qwen2.5:3b'),
        'ollama_endpoint': request_form.get('ollama_endpoint', 'http://localhost:11434'),
        'hotwords': request_form.get('hotwords', ''),  # 添加hotwords
    }
    
    print(f"OpenAI Models(extract_segmentation_params): {segmentation_params['models']}")
    print(f"api_keys(extract_segmentation_params): {segmentation_params['api_keys']}")
    print(f"api_endpoints(extract_segmentation_params): {segmentation_params['api_endpoints']}")
    print(f"Hotwords(extract_segmentation_params): {segmentation_params['hotwords']}")  # 打印hotwords

    # 处理共享的 OpenAI API 设置
    use_shared_openai_api_key = request_form.get('use_shared_openai_api_key', 'false').lower() == 'true'
    shared_openai_api_key = request_form.get('shared_openai_api_key', '')
    use_shared_openai_api_endpoint = request_form.get('use_shared_openai_api_endpoint', 'false').lower() == 'true'
    shared_openai_api_endpoint = request_form.get('shared_openai_api_endpoint', '')

    if use_shared_openai_api_key:
        segmentation_params['api_keys'] = [shared_openai_api_key] * 5
    if use_shared_openai_api_endpoint:
        segmentation_params['api_endpoints'] = [shared_openai_api_endpoint] * 5

    return segmentation_params

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        print("Received transcribe request")
        print(f"Request form data: {request.form}")
        
        model_type = request.form.get('model_type', 'whisper')
        if model_type not in ['whisper', 'funasr']:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        perform_segmentation = request.form.get('perform_segmentation', 'false').lower() == 'true'
        print(f"Perform Segmentation: {perform_segmentation}")
        segment_model = request.form.get('segment_model', 'ollama')
        
        segmentation_params = extract_segmentation_params(request.form)
        
        print(f"Max Length: {segmentation_params['max_length']}")
        print(f"Max Segment Lengths: {segmentation_params['max_segment_lengths']}")
        print(f"OpenAI Models: {segmentation_params['models']}")
        print(f"OpenAI API Endpoints: {segmentation_params['api_endpoints']}")
        print(f"OpenAI Priority: {segmentation_params['priority']}")
        print(f"Enable OpenAI Rotation: {segmentation_params['enable_rotation']}")
        print(f"Segmentation Tolerance: {segmentation_params['segmentation_tolerance']}")
        print(f"Segmentation Tolerance Unit: {segmentation_params['segmentation_tolerance_unit']}")
        print(f"Ollama Model: {segmentation_params['ollama_model']}")
        print(f"Ollama Endpoint: {segmentation_params['ollama_endpoint']}")
        print(f"Segment Model: {segment_model}")
        print(f"Hotwords: {segmentation_params['hotwords']}")  # 打印hotwords

        text = request.form.get('text', '')
        print(f"Received text: {text}")

        result = []
        rotation_message = None
        source = 'unknown'

        # 检查是否是字幕文件
        is_subtitle = re.search(r'\.(srt|ass|vtt)(\]\]|\))', text, re.IGNORECASE) is not None
        print(f"Is Subtitle: {is_subtitle}")

        if is_subtitle:
            # 处理字幕文件的情况
            try:
                subtitle_path = extract_subtitle_file_path(text)
                print(f"Subtitle Path: {subtitle_path}")
                if not subtitle_path:
                    raise ValueError("No subtitle file path found in the block content")
                
                subtitle_content = convert_subtitle_to_transcription(subtitle_path)
                print(f"Subtitle Content: {subtitle_content}")
                text = '\n'.join([f"{{{{timestamp {seg['start']}}}}} {seg['text']}" for seg in subtitle_content['segments']])
                source = 'subtitle'
            except json.JSONDecodeError:
                # 如果不是 JSON，假设它已经是正确格式的文本
                source = 'unknown'
            
            print(f"Processed subtitle text: {text}")  # 添加这行来打印处理后的文本
            
            result, rotation_message = process_and_segment_text(text, perform_segmentation, segment_model, segmentation_params)
        else:
            # 处理其他情况（YouTube、Bilibili、本地文件等）
            youtube_match = re.search(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+', text)
            bilibili_match = re.search(r'https?://(?:www\.)?bilibili\.com/video/[\w-]+', text)
            local_file_match = re.search(r'([^\s]+\.(?:mp4|avi|mov|mkv|flv|wmv|mp3|wav|m4a|flac|ogg))', text, re.IGNORECASE)

            if youtube_match or bilibili_match:
                video_url = youtube_match.group() if youtube_match else bilibili_match.group()
                print(f"Detected video URL: {video_url}")
                audio_path = download_video(video_url)
                source = "youtube" if youtube_match else "bilibili"
                result, rotation_message = transcribe_audio(
                    audio_path, 
                    model_type=model_type,
                    segment_model=segment_model,
                    perform_segmentation=perform_segmentation,
                    **segmentation_params
                )
            elif local_file_match:
                local_path = local_file_match.group(1)
                print(f"Detected local file: {local_path}")
                if is_audio_file(local_path):
                    audio_path = local_path
                else:
                    audio_path = extract_audio_from_local_video(local_path)
                source = "local"
                result, rotation_message = transcribe_audio(
                    audio_path, 
                    model_type=model_type,
                    segment_model=segment_model,
                    perform_segmentation=perform_segmentation,
                    **segmentation_params
                )
            else:
                print("No valid video URL or local file detected. Processing as plain text.")
                result, rotation_message = process_and_segment_text(text, perform_segmentation, segment_model, segmentation_params)

        print(f"Transcription result: {result}")
        print(f"Rotation message: {rotation_message}")

        response = {
            "status": "completed",
            "source": source,
            "segments": result
        }

        if rotation_message:
            response["openai_rotation_message"] = rotation_message

        return jsonify(response)

    except Exception as e:
        print(f"Error in transcribe function: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "logseq-whisper-subtitle-server error: " + str(e),
            "source": "",
            "segments": []
        })

def process_and_segment_text(text, perform_segmentation_flag, segment_model, segmentation_params):
    if perform_segmentation_flag:
        return segment_text(text, segment_model, segmentation_params)
    else:
        return process_segments_with_timestamps(text,0), None

@app.route('/segment', methods=['POST'])
def segment_text_route():
    try:
        text = request.form.get('text')
        segment_model = request.form.get('segment_model', 'ollama')
        segmentation_params = extract_segmentation_params(request.form)
        
        segments, _ = segment_text(text, segment_model, segmentation_params)
        print(f"Segments: {segments}")
        return jsonify({"segments": segments})
    except Exception as e:
        print(f"Error in segment_text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/funasr_models', methods=['GET'])
def get_funasr_models():
    """Get available FunASR models"""
    return jsonify(FUNASR_MODELS)

@app.route('/ollama_models', methods=['GET'])
def get_available_ollama_models():
    """Get available Ollama models"""
    ollama_endpoint = request.args.get('endpoint', 'http://localhost:11434')
    models = get_ollama_models(ollama_endpoint)
    return jsonify(models)

@app.route('/convert_subtitle', methods=['POST'])
def convert_subtitle():
    try:
        subtitle_path = request.form.get('subtitle_path')
        print(f"Received subtitle path: {subtitle_path}")
        if not subtitle_path:
            return jsonify({"error": "No subtitle file path provided"}), 400

        # 获取 Logseq 图谱路径
        graph_path = request.form.get('graph_path', '')
        
        # 如果是相对路径，将其转换为绝对路径
        if subtitle_path.startswith('../'):
            subtitle_path = os.path.join(graph_path, subtitle_path[3:])
        elif not os.path.isabs(subtitle_path):
            subtitle_path = os.path.join(graph_path, subtitle_path)

        print(f"Attempting to convert subtitle file: {subtitle_path}")

        if not os.path.exists(subtitle_path):
            return jsonify({"error": f"Subtitle file not found: {subtitle_path}"}), 404

        result = convert_subtitle_to_transcription(subtitle_path)
        return jsonify(result)
    except Exception as e:
        print(f"Error in convert_subtitle: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_subtitle_file_path(content):
    # 匹配 orgmode 链接格式
    logseq_link_match = re.search(r'\[\[(.*?\.(?:srt|ass|vtt))\]\[.*?\]\]', content)
    if logseq_link_match:
        return logseq_link_match.group(1)

    # 匹配普通的 Markdown 链接格式
    markdown_link_match = re.search(r'\[(.*?)\]\((.*?\.(?:srt|ass|vtt))\)', content)
    if markdown_link_match:
        return markdown_link_match.group(2)

    return None

@app.route('/summarize', methods=['POST'])
def summarize():
    print(f"Begin summarize")
    try:
        text = request.form.get('text')
        api_setting_priority = request.form.get('api_setting_priority', '1,2,3,4,5')
        graph_path = request.form.get('graph_path', '')
        print(f"Received text: {text}")
        print(f"Received api_setting_priority: {api_setting_priority}")
        print(f"Received graph_path: {graph_path}")
        
        if not text:
            return jsonify({"error": "No text provided for summarization"}), 400

        # 检查是否是字幕文件
        subtitle_path = extract_subtitle_file_path(text)
        if subtitle_path:
            print(f"Detected subtitle file: {subtitle_path}")
            # 如果是相对路径，将其转换为绝对路径
            if subtitle_path.startswith('../'):
                subtitle_path = os.path.join(graph_path, subtitle_path[3:])
            elif not os.path.isabs(subtitle_path):
                subtitle_path = os.path.join(graph_path, subtitle_path)

            print(f"Full subtitle path: {subtitle_path}")
            if not os.path.exists(subtitle_path):
                return jsonify({"error": f"Subtitle file not found: {subtitle_path}"}), 404

            subtitle_content = convert_subtitle_to_transcription(subtitle_path)
            text = '\n'.join([f"{{{{timestamp {seg['start']}}}}} {seg['text']}" for seg in subtitle_content['segments']])
            print(f"Converted subtitle to text, length: {len(text)}")

        segmentation_params = extract_segmentation_params(request.form)
        print(f"Segmentation Params: {segmentation_params}")
        summary = summarize_text(text, api_setting_priority, **segmentation_params)
        return jsonify({"summary": summary})
    except Exception as e:
        print(f"Error in summarize function: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Summarization error: {str(e)}"}), 500

# 在服务器启动之前加载模型
# preload_models()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5014)