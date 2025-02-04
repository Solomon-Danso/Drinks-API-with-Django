import os
import pypdf
import docx
import json
import markdown
import sqlite3
import speech_recognition as sr
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
import numpy as np

# from moviepy.editor import AudioFileClip

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Extract text from Markdown (.md)
def extract_text_from_md(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        return markdown.markdown(f.read())

# Extract text from JSON
def extract_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return json.dumps(data, indent=2)  # Convert to readable text

# Extract text from database (SQLite example)
def extract_text_from_db(db_path, query="SELECT * FROM documents"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return "\n".join([str(row) for row in rows])

# Convert speech from audio files to text
def extract_text_from_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

# # Extract text from video (convert to audio first)
# def extract_text_from_video(video_path):
#     audio_path = "temp_audio.wav"
#     video = AudioFileClip(video_path)
#     video.audio.write_audiofile(audio_path)
#     text = extract_text_from_audio(audio_path)
#     os.remove(audio_path)
#     return text

# Extract text from images using OCR
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

# Load all documents
def load_documents(folder_path):
    all_texts = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            all_texts.append(extract_text_from_pdf(file_path))
        elif file.endswith(".docx"):
            all_texts.append(extract_text_from_docx(file_path))
        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                all_texts.append(f.read())
        elif file.endswith(".md"):
            all_texts.append(extract_text_from_md(file_path))
        elif file.endswith(".json"):
            all_texts.append(extract_text_from_json(file_path))
        elif file.endswith(".db"):
            all_texts.append(extract_text_from_db(file_path))
        elif file.endswith(".mp3") or file.endswith(".wav"):
            all_texts.append(extract_text_from_audio(file_path))

        # elif file.endswith(".mp4") or file.endswith(".avi"):
        #     all_texts.append(extract_text_from_video(file_path))

        elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            all_texts.append(extract_text_from_image(file_path))
    return all_texts

# Load all documents
documents = load_documents("Sources/")

# Generate Embedding 
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast model
embeddings = model.encode(documents)

# Save embeddings

np.save("document_embeddings.npy", embeddings)

# Fine Tune 
model_name = "all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Convert documents to tokenized inputs
inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500
)

trainer = Trainer(model=model, args=training_args, train_dataset=inputs)
trainer.train()

# Save fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")



