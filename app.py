import os
import json
import boto3
from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return ''.join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Load sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Create Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def get_cosine(job, resume):
    job_vec = embedder.encode([job])[0]
    resume_vec = embedder.encode([resume])[0]
    return cosine_similarity([job_vec], [resume_vec])[0][0]

def get_summary(job, resume):
    prompt_message = {
        "role": "user",
        "content": f"Why is this candidate a good fit?\n\nJob:\n{job}\n\nResume:\n{resume[:1500]}"
    }

    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [prompt_message],
            "max_tokens": 200,
            "temperature": 0.7
        })
    )

    result = json.loads(response['body'].read())
    return result['content'][0]['text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    job = request.form['job_description']
    uploaded_files = request.files.getlist('resumes')

    results = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            resume_text = extract_text(filepath)
            score = get_cosine(job, resume_text)
            summary = get_summary(job, resume_text)
            results.append({
                'name': filename,
                'score': round(score * 100, 2),  # Keep as float for sorting
                'summary': summary
            })

    # 🔽 Sort by descending similarity score
    results.sort(key=lambda x: x['score'], reverse=True)

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
