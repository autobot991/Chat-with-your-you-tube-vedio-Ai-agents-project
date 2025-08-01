import streamlit as st
import json
import os
import time
import sys
from dotenv import load_dotenv
import requests
import yt_dlp
from pathlib import Path
import re

from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_token = os.getenv('ASSEMBLY_AI_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

base_url = "https://api.assemblyai.com/v2"
headers = {
    "authorization": api_token,
    "content-type": "application/json"
}

# ---------- Streamlit App Configuration ----------
st.set_page_config(layout="wide", page_title="🎙️ AudioChat AI", page_icon="🔊")
st.title("🎙️ Talk to Your YouTube Videos with AI")

with st.sidebar:
    st.header("App Controls")
    st.markdown("Upload a YouTube video/audio, transcribe it, and ask questions about the content.")
    st.markdown("---")
    input_source = st.text_input("🔗 Enter the YouTube video URL")
    download_format = st.radio("📥 Select download format", ["Audio", "Video"], horizontal=True)
    if st.button("🧹 Reset App"):
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))
        st.experimental_rerun()

qa_chain = None

@st.cache_data
def load_transcription(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

@st.cache_data
def load_word_timestamps(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def download_media(url, download_format):
    try:
        os.makedirs('temp', exist_ok=True)
        ydl_metadata = yt_dlp.YoutubeDL().extract_info(url, download=False)
        title = re.sub(r'[\\/:*?"<>|’“”：]', '', ydl_metadata.get("title", "downloaded"))
        ydl_metadata["title"] = title

        ydl_opts = {
            'outtmpl': f'temp/{title}.%(ext)s',
        }

        if download_format == "Audio":
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            })
        else:
            ydl_opts.update({
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                'merge_output_format': 'mp4',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'keepvideo': True,
            })

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_filename = f"{title}.mp3"
        video_filename = f"{title}.mp4" if download_format == 'Video' else None
        logger.info(f"Successfully downloaded {download_format.lower()}: {audio_filename if download_format == 'Audio' else video_filename}")
        return audio_filename, video_filename
    except Exception as e:
        logger.error(f"Error downloading media: {str(e)}")
        st.error(f"Error downloading media: {str(e)}")
        return None, None

def assemblyai_stt(audio_filename):
    try:
        audio_path = os.path.join('temp', audio_filename)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        with open(audio_path, "rb") as f:
            response = requests.post(base_url + "/upload", headers=headers, data=f)
        response.raise_for_status()
        upload_url = response.json()["upload_url"]
        data = {"audio_url": upload_url}
        url = base_url + "/transcript"
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        transcript_id = response.json()['id']
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        while True:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()
            if transcription_result['status'] == 'completed':
                break
            elif transcription_result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")
            else:
                time.sleep(3)
        transcription_text = transcription_result['text']
        word_timestamps = transcription_result['words']
        os.makedirs('docs', exist_ok=True)
        with open('docs/transcription.txt', 'w') as file:
            file.write(transcription_text)
        with open('docs/word_timestamps.json', 'w') as file:
            json.dump(word_timestamps, file)
        logger.info("Successfully transcribed audio with word-level timestamps")
        return transcription_text, word_timestamps
    except Exception as e:
        logger.error(f"Error in speech-to-text conversion: {str(e)}")
        st.error(f"Error in speech-to-text conversion: {str(e)}")
        return None, None

@st.cache_resource
def setup_qa_chain():
    try:
        loader = TextLoader('docs/transcription.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()
        chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        word_timestamps = load_word_timestamps('docs/word_timestamps.json')
        return qa_chain, word_timestamps
    except Exception as e:
        logger.error(f"Error setting up QA chain: {str(e)}")
        st.error(f"Error setting up QA chain: {str(e)}")
        return None, None

def find_relevant_timestamps(answer, word_timestamps):
    relevant_timestamps = []
    answer_words = answer.lower().split()
    for word_info in word_timestamps:
        if word_info['text'].lower() in answer_words:
            relevant_timestamps.append(word_info['start'])
    return relevant_timestamps

def generate_summary(transcription):
    chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )
    summary_prompt = PromptTemplate(
        input_variables=["transcription"],
        template="Summarize the following transcription in 3-5 sentences:\n\n{transcription}"
    )
    summary_chain = LLMChain(llm=chat, prompt=summary_prompt)
    summary = summary_chain.run(transcription)
    return summary

# ---------- Main App Logic ----------
if input_source:
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("Downloading media..."):
            audio_filename, video_filename = download_media(input_source, download_format)

        if audio_filename:
            video_path = os.path.join("temp", video_filename) if video_filename else None
            if download_format == "Video" and video_filename:
                if os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.warning("Video file not found. It may have been deleted during processing.")
            else:
                st.audio(os.path.join("temp", audio_filename))

            with st.spinner("Transcribing audio..."):
                transcription, word_timestamps = assemblyai_stt(audio_filename)

            if transcription:
                st.success("Transcription completed. You can now ask questions.")
                st.text_area("📄 Transcription Preview", transcription, height=300)
                with open("docs/transcription.txt", "r") as f:
                    st.download_button("⬇ Download Transcript", f, file_name="transcription.txt")

                qa_chain, word_timestamps = setup_qa_chain()
                if st.button("🧠 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcription)
                        st.subheader("📝 Summary")
                        st.write(summary)

    with col2:
        st.info("💬 Ask questions about the transcription")
        query = st.chat_input("Ask your question...")
        if query:
            if qa_chain:
                with st.spinner("Thinking..."):
                    result = qa_chain({"query": query})
                    answer = result['result']
                    st.chat_message("user").write(query)
                    st.chat_message("assistant").write(answer)

                    relevant_timestamps = find_relevant_timestamps(answer, word_timestamps)
                    if relevant_timestamps:
                        st.subheader("⏱ Relevant Timestamps")
                        for timestamp in relevant_timestamps[:5]:
                            st.markdown(f"**{timestamp // 60}:{timestamp % 60:02d}**")
            else:
                st.error("⚠ QA system is not ready. Please make sure the transcription is completed.")

def cleanup_temp_files():
    for file in os.listdir('temp'):
        os.remove(os.path.join('temp', file))
