import os
import uuid
import time
import requests
import json
from moviepy.editor import AudioFileClip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

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
    youtube_url: str
    interval_minute: int
    downloader_api_key: str  # downloader_api_key로 수정

# 유튜브 API를 이용해 가장 작은 해상도 MP4 파일을 다운로드하고 지정된 간격으로 오디오를 추출해 나누기
async def download_video_and_split_audio(youtube_url: str, interval_minute: int, downloader_api_key: str) -> List[str]:
    # 유튜브 영상 정보를 가져오기 위한 API URL과 인증 헤더
    api_url = "https://zylalabs.com/api/3219/youtube+mp4+video+downloader+api/5880/get+mp4"
    api_headers = {
        'Authorization': f'Bearer {downloader_api_key}'  # 요청에서 받은 downloader_api_key 값을 사용
    }

    # API 요청
    response = requests.get(f"{api_url}?id={youtube_url.split('v=')[-1]}", headers=api_headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to retrieve video information from API")

    # JSON 응답 파싱
    data = json.loads(response.text)

    # 가장 작은 해상도의 MP4 파일 찾기
    smallest_resolution = None
    smallest_mp4_url = None

    for format in data.get('formats', []):
        if format.get('mimeType', '').startswith('video/mp4'):  # mp4 파일 필터링
            width = format.get('width')
            height = format.get('height')
            if width and height:
                if smallest_resolution is None or (width * height) < (smallest_resolution[0] * smallest_resolution[1]):
                    smallest_resolution = (width, height)
                    smallest_mp4_url = format.get('url')

    # 가장 작은 해상도의 mp4 파일 다운로드
    if smallest_mp4_url:
        print(f"가장 작은 해상도 MP4 파일 다운로드 링크: {smallest_mp4_url} (해상도: {smallest_resolution[0]}x{smallest_resolution[1]})")
        
        # MP4 파일 다운로드
        video_response = requests.get(smallest_mp4_url, stream=True)
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        raise HTTPException(status_code=500, detail="Failed to find a suitable MP4 file")

    # 영상에서 오디오 추출
    audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
    audio_clip = AudioFileClip(video_file)
    audio_clip.write_audiofile(audio_file)

    # 오디오 파일을 지정된 간격으로 나누기
    duration = audio_clip.duration
    interval_seconds = interval_minute * 60
    chunk_files = []

    for start_time in range(0, int(duration), interval_seconds):
        chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip.subclip(start_time, min(start_time + interval_seconds, duration)).write_audiofile(chunk_file, verbose=False, logger=None)
        chunk_files.append(chunk_file)

    audio_clip.close()
    os.remove(video_file)  # 원본 영상 파일 삭제

    return chunk_files

# hh:mm:ss 포맷으로 변환
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 비동기로 텍스트 요약 처리
async def summarize_text(api_key: str, text_chunks: List[str], chunk_times: List[str]) -> str:
    client = OpenAI(api_key=api_key)
    summarized_text = ""

    tasks = []
    for i, chunk in enumerate(text_chunks):
        task = asyncio.create_task(
            client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Summarize the following text."},
                    {"role": "user", "content": chunk}
                ]
            )
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        summary = response.choices[0].message.content
        summarized_text += f"{chunk_times[i]}: {summary}\n"

    return summarized_text

# 비동기로 오디오 파일을 처리하고 STT 수행
async def transcribe_audio_chunks(client, audio_chunks, interval_minute):
    transcribed_texts = []
    chunk_times = []

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        for i, chunk_file in enumerate(audio_chunks):
            start_time_seconds = i * interval_minute * 60  # 초 단위로 청크 시작 시간 계산
            chunk_times.append(seconds_to_timecode(start_time_seconds))  # hh:mm:ss 형식으로 변환

            # 비동기로 파일 처리 및 전사 요청
            transcript_response = await loop.run_in_executor(
                pool, lambda: client.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(chunk_file, "rb"),
                    response_format="text"
                )
            )
            transcribed_texts.append(transcript_response)

            # 추출된 음성 파일 삭제
            os.remove(chunk_file)

    return transcribed_texts, chunk_times

@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    # 인증키 확인
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")

    # 유튜브 영상을 다운로드하고 오디오 추출 및 나누기
    try:
        audio_chunks = await download_video_and_split_audio(request.youtube_url, request.interval_minute, request.downloader_api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or splitting audio: {str(e)}")

    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=request.api_key)

    # 비동기적으로 STT 처리
    try:
        transcribed_texts, chunk_times = await transcribe_audio_chunks(client, audio_chunks, request.interval_minute)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

    # 텍스트 요약
    summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)

    return summary_text

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
