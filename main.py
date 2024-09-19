import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import uuid
import yt_dlp
from moviepy.editor import AudioFileClip
from starlette.responses import JSONResponse
from typing import List
import time
import random

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

# User-Agent 목록
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

# 유튜브에서 오디오를 다운로드하고 지정된 간격으로 나누기
def download_and_split_audio(youtube_url: str, interval_minute: int) -> List[str]:
    ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': os.path.join(AUDIO_DIR, '%(title)s.%(ext)s'),
    'user_agent': random.choice(USER_AGENTS),
    'no_check_certificate': True,
    'ignoreerrors': False,
    'quiet': True,
    'no_warnings': True,
    'extract_flat': 'in_playlist',
    'force_generic_extractor': True,
    'source_address': '13.228.225.19',
    'geo_bypass': True,  # 지역 제한을 우회하는 옵션
    'verbose': True,
    'allow_unplayable_formats': True,  # 재생 불가 형식 허용
    'noplaylist': True  # 플레이리스트가 아닌 개별 영상만 다운로드
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                if info is None:
                    raise Exception("Failed to extract video information")
                audio_file = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")
            time.sleep(5)  # 재시도 전 5초 대기

    # 다운로드한 오디오 파일을 지정된 간격으로 나누기
    audio_clip = AudioFileClip(audio_file)
    duration = audio_clip.duration
    interval_seconds = interval_minute * 60
    chunk_files = []

    # 오디오를 청크 단위로 자르기
    for start_time in range(0, int(duration), interval_seconds):
        chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip.subclip(start_time, min(start_time + interval_seconds, duration)).write_audiofile(chunk_file, verbose=False, logger=None)
        chunk_files.append(chunk_file)

    audio_clip.close()
    os.remove(audio_file)
    return chunk_files

# 타임코드 추가한 요약 텍스트 생성
async def summarize_text(api_key: str, text_chunks: List[str], chunk_times: List[str]) -> str:
    openai.api_key = api_key
    summarized_text = ""

    for i, chunk in enumerate(text_chunks):
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 모델 설정
            messages=[
                {"role": "system", "content": "Summarize the following text."},
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

    return JSONResponse(content={"summary_text": summary_text})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
