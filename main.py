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
from pydub import AudioSegment, silence

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="LINKBRICKS HORIZON-AI STT API ENGINE",
        version="100.100.100",
        description="## 영상 내용 음성 추출 및 텍스트 변환 엔진 \n - API Swagger \n - Multilingual VIDEO STT \n - MP4, MOV, AVI, MKV, WMV, FLV, OGG, WebM \n - YOUTUBE, TIKTOK",
        routes=app.routes,
    )
    openapi_schema["info"]["contact"] = {
        "name": "Linkbricks Horizon AI",
        "url": "https://www.linkbricks.com",
        "email": "contact@linkbricks.com"
    }
    openapi_schema["info"]["license"] = {
        "name": "GNU GPL 3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

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
    interval_seconds: int  # 초 단위로 변경
    downloader_api_key: str
    summary_flag: int
    chunking_method: str  # "interval" 또는 "silence"

# 유튜브 영상인지 확인하는 함수
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

# 틱톡 영상인지 확인하는 함수
def is_tiktok_url(url: str) -> bool:
    return "tiktok.com" in url

# 유튜브 URL을 표준 형식으로 변환하는 함수
def normalize_youtube_url(video_url: str) -> str:
    # youtu.be 형식
    if "youtu.be" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # youtube.com/embed 형식
    if "youtube.com/embed" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"

    # youtube.com/shorts 형식 처리
    if "youtube.com/shorts" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    
    # youtube.com/watch 형식 (이미 표준화된 URL)
    if "youtube.com/watch" in video_url:
        return video_url.split('&')[0]  # `&`로 이어지는 추가 쿼리 매개변수 제거
    
    # 예상치 못한 형식은 예외 처리
    raise ValueError("Invalid YouTube URL format")

# 유튜브 또는 틱톡 API를 이용해 가장 작은 해상도 MP4 파일을 다운로드하고 지정된 간격으로 오디오를 추출해 나누기
async def download_video_and_split_audio(video_url: str, interval_seconds: int, downloader_api_key: str, chunking_method: str) -> List[str]:
    video_file_extension = None

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

        # 특정 지역에서 제한된 영상 처리
        if data.get('status') == 'fail' and 'description' in data:
            return [data['description'] + "\n[해당 동영상은 저작권자의 요청으로 저작권자가 입력한 내용만 출력합니다]"]

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
            video_file_extension = "mp4"
        else:
            raise HTTPException(status_code=500, detail="Failed to find a suitable MP4 file")

    elif is_tiktok_url(video_url):
        # 틱톡 영상 처리
        api_url = "https://zylalabs.com/api/4481/tiktok+video+retriever+api/5499/video+download"
        api_headers = {
            'Authorization': f'Bearer {downloader_api_key}'
        }

        response = requests.get(f"{api_url}?url={video_url}", headers=api_headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve TikTok video information from API")

        data = json.loads(response.text)
        smallest_mp4_url = data.get('data', {}).get('wmplay')

        if smallest_mp4_url:
            video_response = requests.get(smallest_mp4_url, stream=True)
            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            video_file_extension = "mp4"
        else:
            raise HTTPException(status_code=500, detail="Failed to find a suitable MP4 file for TikTok video")

    else:
        # 일반적인 웹의 동영상 파일 또는 mp3 처리
        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to download video or audio file from the provided URL")
        
        # 파일 확장자를 유지하여 저장
        video_file_extension = video_url.split('.')[-1]
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.{video_file_extension}")
        
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    # 영상에서 오디오 추출 (영상 파일인 경우만)
    if video_file_extension in ["mp4", "mov", "avi", "mkv", "wmv", "flv", "ogg", "webm"]:
        audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip = AudioFileClip(video_file)
        audio_clip.write_audiofile(audio_file)
        audio_clip.close()
        os.remove(video_file)
    else:
        # MP3 파일인 경우 이미 오디오 파일이므로 그대로 사용
        audio_file = video_file

    # 오디오 파일을 PyDub로 불러오기
    audio = AudioSegment.from_file(audio_file)
    chunk_files = []

    if chunking_method == "silence":
        # 무음 구간을 기준으로 오디오 청킹
        chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=250)
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)
    elif chunking_method == "interval":
        # 지정된 간격으로 오디오 청킹
        duration = len(audio) / 1000  # PyDub에서 길이는 밀리초 단위
        for start_time in range(0, int(duration), interval_seconds):
            chunk = audio[start_time * 1000:min((start_time + interval_seconds) * 1000, len(audio))]
            chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)

    os.remove(audio_file)
    
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
async def transcribe_audio_chunks(api_key: str, audio_chunks, interval_seconds):
    transcribed_texts = []
    chunk_times = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk_file in enumerate(audio_chunks):
            start_time_seconds = i * interval_seconds
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
        # 유튜브 또는 틱톡 URL 표준화 처리
        if is_youtube_url(request.video_url):
            normalized_video_url = normalize_youtube_url(request.video_url)
        else:
            normalized_video_url = request.video_url
        
        audio_chunks = await download_video_and_split_audio(normalized_video_url, request.interval_seconds, request.downloader_api_key, request.chunking_method)
        
        # 특정 지역 제한된 영상의 경우 description으로 바로 반환
        if len(audio_chunks) == 1 and isinstance(audio_chunks[0], str):
            return {"summary": audio_chunks[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or splitting audio: {str(e)}")

    try:
        transcribed_texts, chunk_times = await transcribe_audio_chunks(request.api_key, audio_chunks, request.interval_seconds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")

    if request.summary_flag == 1:
        summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)
        return {"summary": summary_text}
    else:
        if len(audio_chunks) == 1 and isinstance(audio_chunks[0], str):
            return {"transcription": audio_chunks[0]}
        full_transcription = "\n".join([f"{chunk_times[i]}: {transcribed_texts[i]}" for i in range(len(transcribed_texts))])
        return {"transcription": full_transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
