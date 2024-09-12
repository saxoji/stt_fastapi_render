import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
import uuid
from pytube import YouTube  # pytube로 변경
from moviepy.editor import AudioFileClip
from pathlib import Path
from starlette.responses import JSONResponse
import aiofiles
import shutil
from typing import List
import time

app = FastAPI()

# 인증키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# Directory to save the audio files
AUDIO_DIR = "audio"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# 모델 정의
class YouTubeAudioRequest(BaseModel):
    api_key: str
    auth_key: str
    youtube_url: str
    interval_minute: int

# 유튜브에서 오디오를 다운로드하여 지정된 간격으로 나누기
def download_and_split_audio(youtube_url: str, interval_minute: int) -> List[str]:
    # YouTube 객체 생성 및 오디오 스트림 다운로드
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp4")  # mp4로 다운로드

    # 오디오 다운로드
    audio_stream.download(output_path=AUDIO_DIR, filename=os.path.basename(audio_file))

    # 오디오 파일 로드
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration
    interval_seconds = interval_minute * 60
    chunk_files = []

    # 오디오를 청크 단위로 자르기
    for start_time in range(0, int(duration), interval_seconds):
        chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip.subclip(start_time, min(start_time + interval_seconds, duration)).write_audiofile(chunk_file)
        chunk_files.append(chunk_file)

    # 오디오 파일 삭제
    audio_clip.close()
    os.remove(audio_file)
    return chunk_files

# 타임코드 추가한 요약 텍스트 생성
async def summarize_text(api_key: str, text_chunks: List[str], chunk_times: List[str]) -> str:
    # OpenAI API 키 설정
    openai.api_key = api_key

    summarized_text = ""

    for i, chunk in enumerate(text_chunks):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Summarize the following text"},
                {"role": "user", "content": chunk}
            ]
        )
        summary = response.choices[0].message["content"]
        summarized_text += f"{chunk_times[i]}: {summary}\n"

    return summarized_text

@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    # 인증키 확인
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")
    
    # OpenAI API 키 설정
    openai.api_key = request.api_key

    # 유튜브 음성을 다운로드하고 나누기
    try:
        audio_chunks = download_and_split_audio(request.youtube_url, request.interval_minute)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or splitting audio: {str(e)}")

    # 각 청크에 대해 STT 수행
    transcribed_texts = []
    chunk_times = []
    for i, chunk_file in enumerate(audio_chunks):
        start_time = i * request.interval_minute
        chunk_times.append(f"{start_time}m - {start_time + request.interval_minute}m")

        with open(chunk_file, "rb") as audio_file:
            try:
                transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
                transcribed_texts.append(transcript_response['text'])
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
        
        # 추출된 음성 파일 삭제
        os.remove(chunk_file)

    # 텍스트 요약
    summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)

    # 클라이언트에 요약 텍스트 전송
    return JSONResponse(content={"summary_text": summary_text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
