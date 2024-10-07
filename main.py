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

# 인스타그램 영상인지 확인하는 함수
def is_instagram_url(url: str) -> bool:
    return "instagram.com/reel/" in url or "instagram.com/p/" in url

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

# 인스타그램 URL을 표준 형식으로 변환하는 함수
def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

# 유튜브, 틱톡, 인스타그램 영상 다운로드 및 오디오 추출 함수
async def download_video_and_split_audio(video_url: str, interval_seconds: int, downloader_api_key: str, chunking_method: str):
    video_file_extension = None
    caption = None  # caption 초기화

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
            return [data['description'] + "\n[해당 동영상은 저작권자의 요청으로 저작권자가 입력한 내용만 출력합니다]"], None

        smallest_mp4_url = next((fmt.get('url') for fmt in data.get('formats', []) if fmt.get('mimeType', '').startswith('video/mp4')), None)

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

    elif is_instagram_url(video_url):
        # 인스타그램 영상 처리
        normalized_url = normalize_instagram_url(video_url)
        api_url = f"https://zylalabs.com/api/1943/instagram+reels+downloader+api/2944/reel+downloader?url={normalized_url}"
        headers = {'Authorization': f'Bearer {downloader_api_key}'}
        response = requests.get(api_url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to retrieve Instagram video information from API")
        data = response.json()
        video_download_url = data.get("video")
        caption = data.get("caption", "")

        if not video_download_url:
            raise HTTPException(status_code=500, detail="Failed to find a suitable MP4 file for Instagram video")

        video_response = requests.get(video_download_url, stream=True)
        video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
        with open(video_file, 'wb') as file:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        video_file_extension = "mp4"

    else:
        raise HTTPException(status_code=400, detail="Unsupported video platform")

    # 영상에서 오디오 추출 (영상 파일인 경우만)
    if video_file_extension and video_file_extension.lower() in ["mp4", "mov", "avi", "mkv", "wmv", "flv", "ogg", "webm"]:
        audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        audio_clip = AudioFileClip(video_file)
        audio_clip.write_audiofile(audio_file)
        audio_clip.close()
        os.remove(video_file)
    else:
        raise HTTPException(status_code=500, detail="Unsupported video format for audio extraction")

    # 오디오 파일을 PyDub로 불러오기
    audio = AudioSegment.from_file(audio_file)
    chunk_files = []

    if chunking_method == "silence":
        chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=audio.dBFS-14, keep_silence=250)
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)
    elif chunking_method == "interval":
        duration = len(audio) / 1000  # PyDub에서 길이는 밀리초 단위
        for start_time in range(0, int(duration), interval_seconds):
            chunk = audio[start_time * 1000:min((start_time + interval_seconds) * 1000, len(audio))]
            chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)

    os.remove(audio_file)

    if not chunk_files:
        raise HTTPException(status_code=500, detail="No audio chunks were created")
    
    return chunk_files, caption

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
                            {"role": "system", "content": "Summarize the following transcription text without any of your own comments."},
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

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, chunk_file in enumerate(audio_chunks):
            start_time_seconds = i * interval_seconds
            chunk_times.append(seconds_to_timecode(start_time_seconds))

            # Read the audio file content
            with open(chunk_file, 'rb') as f:
                audio_data = f.read()

            form = aiohttp.FormData()
            form.add_field('file', audio_data, filename=os.path.basename(chunk_file), content_type='audio/mpeg')
            form.add_field('model', 'whisper-1')
            form.add_field('response_format', 'json')

            task = asyncio.create_task(
                session.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    data=form
                )
            )
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            result = await response.json()
            transcribed_texts.append(result.get('text', ""))
            os.remove(audio_chunks[i])  # Remove the processed audio chunk

    return transcribed_texts, chunk_times

@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")

    try:
        if is_youtube_url(request.video_url):
            normalized_video_url = normalize_youtube_url(request.video_url)
        elif is_instagram_url(request.video_url):
            normalized_video_url = normalize_instagram_url(request.video_url)
        else:
            normalized_video_url = request.video_url

        audio_chunks, caption = await download_video_and_split_audio(normalized_video_url, request.interval_seconds, request.downloader_api_key, request.chunking_method)

        # 특정 지역 제한된 영상의 경우 description으로 바로 반환
        if len(audio_chunks) == 1 and isinstance(audio_chunks[0], str):
            return {"summary": audio_chunks[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or splitting audio: {str(e)}")

    try:
        transcribed_texts, chunk_times = await transcribe_audio_chunks(request.api_key, audio_chunks, request.interval_seconds)
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
        full_transcription = "\n".join([f"{chunk_times[i]}: {transcribed_texts[i]}" for i in range(len(transcribed_texts))])
        return {"transcription": full_transcription}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
