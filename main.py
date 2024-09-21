import os
import uuid
import requests
import json
from moviepy.editor import AudioFileClip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import openai

SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI STT API ENGINE",
    "version": "100.100.100",
    "description": "## 영상 내용 음성 추출 및 텍스트 변환 엔진 \n - API Swagger \n - Multilingual VIDEO STT \n - MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n - YOUTUBE",
     "favicon_url": "https://www.linkbricks.com/wp-content/uploads/2022/03/cropped-favicon-512-192x192.png",
    "contact": {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    },
}

app = FastAPI(**SWAGGER_HEADERS)

# 인증키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# Directory to save the files
AUDIO_DIR = "audio"
VIDEO_DIR = "video"
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# 모델 정의
class YouTubeAudioRequest(BaseModel):
    api_key: str
    auth_key: str
    video_url: str  # youtube_url에서 video_url로 수정
    interval_minute: int
    downloader_api_key: str
    summary_flag: int

# 유튜브 영상인지 확인하는 함수
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

# 유튜브 API를 이용해 가장 작은 해상도 MP4 파일을 다운로드하고 지정된 간격으로 오디오를 추출해 나누기
async def download_video_and_split_audio(video_url: str, interval_minute: int, downloader_api_key: str) -> List[str]:
    if is_youtube_url(video_url):
        # 유튜브 영상 처리
        api_url = "https://zylalabs.com/api/3219/youtube+mp4+video+downloader+api/5880/get+mp4"
        api_headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }

        response = requests.get(f"{api_url}?id={video_url.split('v=')[-1]}", headers=api_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve video information from API")

        data = json.loads(response.text)

        smallest_resolution = None
        smallest_mp4_url = None

        for format in data.get('formats', []):
            if format.get('mimeType', '').startswith('video/mp4'):
                width = format.get('width')
                height = format.get('height')
                if width and height:
                    if smallest_resolution is None or (width * height) < (smallest_resolution[0] * smallest_resolution[1]):
                        smallest_resolution = (width, height)
                        smallest_mp4_url = format.get('url')

        if smallest_mp4_url:
            video_response = requests.get(smallest_mp4_url, stream=True)
            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
        else:
            raise HTTPException(status_code=500, detail="Failed to find a suitable MP4 file")

    else:
        # 일반적인 웹의 동영상 파일을 처리 (확장자를 제한하지 않음)
        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download video file from the provided URL")
        
        # 파일 확장자를 유지하여 저장
        video_file_extension = video_url.split('.')[-1]
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.{video_file_extension}")
        
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    # 영상에서 오디오 추출
    audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
    audio_clip = AudioFileClip(video_file)
    audio_clip.write_audiofile(audio_file)

    duration = audio_clip.duration
    interval_seconds = interval_minute * 60
    chunk_files = []

    for start_time in range(0, int(duration), interval_seconds):
        chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip.subclip(start_time, min(start_time + interval_seconds, duration)).write_audiofile(chunk_file, verbose=False, logger=None)
        chunk_files.append(chunk_file)

    audio_clip.close()
    os.remove(video_file)

    return chunk_files

# hh:mm:ss 포맷으로 변환
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 비동기로 텍스트 요약 처리
async def summarize_text(api_key: str, text_chunks: List[str], chunk_times: List[str]) -> str:
    summarized_text = ""

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk in enumerate(text_chunks):
            task = asyncio.create_task(
                session.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {"role": "system", "content": "Summarize the following youtube video transcription text without any your comments."},
                            {"role": "user", "content": chunk}
                        ]
                    },
                    headers={"Authorization": f"Bearer {api_key}"}
                )
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            result = await response.json()
            summary = result['choices'][0]['message']['content']
            summarized_text += f"{chunk_times[i]}: {summary}\n"

    return summarized_text

# 비동기로 오디오 파일을 처리하고 STT 수행
async def transcribe_audio_chunks(api_key: str, audio_chunks, interval_minute):
    transcribed_texts = []
    chunk_times = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk_file in enumerate(audio_chunks):
            start_time_seconds = i * interval_minute * 60
            chunk_times.append(seconds_to_timecode(start_time_seconds))

            task = asyncio.create_task(
                session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    data={
                        "model": "whisper-1",
                        "file": open(chunk_file, "rb"),
                        "response_format": "json"
                    }
                )
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            result = await response.json()
            transcribed_texts.append(result['text'])
            os.remove(audio_chunks[i])

    return transcribed_texts, chunk_times

@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")

    try:
        audio_chunks = await download_video_and_split_audio(request.video_url, request.interval_minute, request.downloader_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or splitting audio: {str(e)}")

    try:
        transcribed_texts, chunk_times = await transcribe_audio_chunks(request.api_key, audio_chunks, request.interval_minute)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

    if request.summary_flag == 1:
        summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)
        return {"summary": summary_text}
    else:
        full_transcription = "\n".join([f"{chunk_times[i]}: {transcribed_texts[i]}" for i in range(len(transcribed_texts))])
        return {"transcription": full_transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
