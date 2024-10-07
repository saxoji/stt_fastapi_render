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
import aiohttp
from pydub import AudioSegment, silence

SWAGGER_HEADERS = {
    "title": "LINKBRICKS HORIZON-AI STT API ENGINE",
    "version": "100.100.100",
    "description": "## 영상 내용 음성 추출 및 텍스트 변환 엔진 \n - API Swagger \n - Multilingual VIDEO STT \n - MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n - YOUTUBE, TIKTOK, INSTAGRAM",
    "contact": {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com",
        "license_info": {
            "name": "GNU GPL 3.0",
            "url": "https://www.gnu.org/licenses/gpl-3.0.html",
        },
    },
}

app = FastAPI(**SWAGGER_HEADERS)

# 인증키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# 파일 저장 디렉토리 설정
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
    video_url: str  # video_url로 수정
    interval_seconds: int  # 초 단위
    downloader_api_key: str
    summary_flag: int
    chunking_method: str  # "interval" 또는 "silence"

# URL 확인 함수들
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def is_tiktok_url(url: str) -> bool:
    return "tiktok.com" in url

def is_instagram_url(url: str) -> bool:
    return "instagram.com/reel/" in url or "instagram.com/p/" in url

# 유튜브 URL 표준화 함수
def normalize_youtube_url(video_url: str) -> str:
    # youtu.be 형식 처리
    if "youtu.be" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    # 기타 형식 처리
    # ... 나머지 코드 동일 ...

# 인스타그램 URL 표준화 함수
def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

# 영상 다운로드 및 오디오 추출 함수
async def download_video_and_split_audio(video_url: str, interval_seconds: int, downloader_api_key: str, chunking_method: str):
    # ... 기존 코드 ...

    if is_youtube_url(video_url):
        # 유튜브 처리
        # ... 기존 코드 ...
        if data.get('status') == 'fail' and 'description' in data:
            raise Exception(data['description'] + "\n[해당 동영상은 저작권자의 요청으로 내용만 출력합니다]")
        # ... 나머지 코드 동일 ...

    elif is_instagram_url(video_url):
        # 인스타그램 처리
        # ... 기존 코드 ...

    else:
        raise Exception("지원되지 않는 비디오 플랫폼입니다")

    # ... 오디오 추출 및 청크 생성 코드 동일 ...

    if not chunk_files:
        raise Exception("오디오 청크가 생성되지 않았습니다")

    return chunk_files, caption

# 시간 형식 변환 함수
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 텍스트 요약 함수
async def summarize_text(api_key: str, text_chunks: List[str], chunk_times: List[str]) -> str:
    # ... 기존 코드 ...

# 오디오 청크 전사 함수
async def transcribe_audio_chunks(api_key: str, audio_chunks, interval_seconds):
    # ... 기존 코드 ...

    # 파일 삭제 시 정확한 파일 참조
    for i, response in enumerate(responses):
        result = await response.json()
        transcribed_texts.append(result.get('text', ""))
        os.remove(audio_chunks[i])  # 정확한 파일 참조

    return transcribed_texts, chunk_times

# API 엔드포인트
@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")

    try:
        # URL 표준화
        if is_youtube_url(request.video_url):
            normalized_video_url = normalize_youtube_url(request.video_url)
        elif is_instagram_url(request.video_url):
            normalized_video_url = normalize_instagram_url(request.video_url)
        else:
            normalized_video_url = request.video_url

        audio_chunks, caption = await download_video_and_split_audio(
            normalized_video_url, request.interval_seconds, request.downloader_api_key, request.chunking_method
        )

    except Exception as e:
        # 인스타그램의 경우 캡션이 있으면 반환
        if is_instagram_url(request.video_url) and caption:
            return {"summary": f"[caption]: {caption}"}
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    try:
        transcribed_texts, chunk_times = await transcribe_audio_chunks(
            request.api_key, audio_chunks, request.interval_seconds
        )
    except Exception as e:
        if is_instagram_url(request.video_url) and caption:
            return {"summary": f"[caption]: {caption}"}
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

    if request.summary_flag == 1:
        summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)
        if caption:
            summary_text = f"[caption]: {caption}\n" + summary_text
        return {"summary": summary_text}
    else:
        full_transcription = "\n".join(
            [f"{chunk_times[i]}: {transcribed_texts[i]}" for i in range(len(transcribed_texts))]
        )
        return {"transcription": full_transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
