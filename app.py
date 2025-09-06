import os
import uuid
import json
import subprocess
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'csr'}

# Linux/macOS:
CPP_EXECUTABLE_PATH = './process_csr_app' 
# Windows :
#CPP_EXECUTABLE_PATH = './process_csr.exe'

# --- Flask应用初始化 ---           
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024  # 限制上传大小为160MB


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """检查文件扩展名"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- 路由定义 ---
@app.route('/')
def index():
    """提供前端的 index.html 页面"""
    return send_from_directory('.', 'index.html')

@app.route('/api/process-csr', methods=['POST'])
def process_csr_file():
    """处理CSR文件上传和计算的API端点"""
    if 'file' not in request.files:
        return jsonify({"error": "请求中未找到文件部分"}), 400
    
    file = request.files['file']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "未选择文件或文件类型不允许"}), 400

    # 生成唯一的临时文件名来保存上传的文件
    unique_filename = str(uuid.uuid4()) + ".csr"
    temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    try:
        file.save(temp_filepath)

  
        command = [CPP_EXECUTABLE_PATH, temp_filepath]
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True  
        )

        # 解析C++程序的JSON输出并返回给前端
        output_data = json.loads(result.stdout)
        return jsonify(output_data)

    except subprocess.CalledProcessError as e:
        # 捕获C++程序执行失败的错误
        return jsonify({
            "error": "C++程序执行失败", 
            "details": e.stderr.strip()
        }), 500
        
    except FileNotFoundError:
        return jsonify({"error": f"服务器错误: 找不到C++可执行文件 '{CPP_EXECUTABLE_PATH}'"}), 500

    finally:
      
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- 启动应用 ---
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)


