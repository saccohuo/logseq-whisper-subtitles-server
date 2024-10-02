from flask import Flask, request, jsonify
from services import transcribe_audio, download_video, extract_audio_from_local_video, is_audio_file, FUNASR_MODELS, get_ollama_models, preload_funasr_models
import re
import os
import traceback

app = Flask(__name__)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        text = request.form['text'].strip()
        min_length = request.form.get('min_length', '')
        model_type = request.form.get('model_type', 'whisper')
        model_size = request.form.get('model_size', '') if model_type == 'whisper' else None
        graph_path = request.form.get('graph_path', '')
        zh_type = request.form.get('zh_type', 'zh-cn')
        funasr_model_name = request.form.get('funasr_model_name')
        funasr_model_source = request.form.get('funasr_model_source', 'modelscope')
        ollama_model = request.form.get('ollama_model', 'qwen2.5:3b')
        ollama_endpoint = request.form.get('ollama_endpoint', 'http://localhost:11434')
        segment_model = request.form.get('segment_model', 'ollama')
        openai_api_key = request.form.get('openai_api_key')
        openai_model = request.form.get('openai_model')
        openai_api_endpoint = request.form.get('openai_api_endpoint')

        source = None
        audio_path = None
        youtube_pattern = r"https://www\.youtube\.com/watch\?v=[a-zA-Z0-9_-]+|https://youtu\.be/[a-zA-Z0-9_-]+"
        bilibili_pattern = r"https://www\.bilibili\.com/video/[a-zA-Z0-9_-]+"
        youtube_match = re.search(youtube_pattern, text)
        bilibili_match = re.search(bilibili_pattern, text)

        local_file_pattern = r'(!\[.*?\]\((.*?)\))|(\{\{renderer :[a-zA-Z]+, (.*?)\}\})|\[\[(.*?)\]\[.*?\]\]'
        local_file_match = re.search(local_file_pattern, text)

        if youtube_match or bilibili_match:
            video_url = youtube_match.group() if youtube_match else bilibili_match.group()
            audio_path = download_video(video_url)
            source = "youtube" if youtube_match else "bilibili"
        elif local_file_match:
            if local_file_match.group(2) is not None:
                local_file_path = local_file_match.group(2)
            elif local_file_match.group(4) is not None:
                local_file_path = local_file_match.group(4)
            elif local_file_match.group(5) is not None:
                local_file_path = local_file_match.group(5)
            else:
                return jsonify({
                    "source": "",
                    "segments": [],
                    "error": "No local file path found"
                })

            if local_file_path.startswith("http") or local_file_path.startswith("https"):
                print("This is a URL, not a local file")
                return jsonify({
                    "source": "",
                    "segments": [],
                    "error": "This is a URL, not a local file"
                })

            source = "local"
            if local_file_path.startswith("../"):
                local_file_path = os.path.join(graph_path, local_file_path[3:])

            audio_path = local_file_path
            if not is_audio_file(local_file_path):
                audio_path = extract_audio_from_local_video(local_file_path)
            print(f"Extracted file path: {local_file_path}")
        else:
            # 处理直接文本输入
            source = "text"
            audio_path = "text_input.txt"
            with open(audio_path, "w", encoding="utf-8") as f:
                f.write(text)

        # 直接执行转录
        result = transcribe_audio(audio_path, min_length, model_type, model_size, zh_type, 
                                  funasr_model_name, funasr_model_source, 
                                  segment_model=segment_model,
                                  ollama_model=ollama_model, 
                                  ollama_endpoint=ollama_endpoint,
                                  openai_api_key=openai_api_key,
                                  openai_model=openai_model,
                                  openai_api_endpoint=openai_api_endpoint)

        return jsonify({
            "status": "completed",
            "source": source,
            "segments": result
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": "logseq-whisper-subtitle-server error: " + str(e),
            "source": "",
            "segments": []
        })

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

# 在服务器启动之前预加载模型
preload_funasr_models()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5014)