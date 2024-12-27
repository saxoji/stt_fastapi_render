import os
import uuid
import requests
import json
from moviepy.editor import AudioFileClip, VideoFileClip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import datetime
import asyncio
import aiohttp
from pydub import AudioSegment, silence
import math
import subprocess
import yt_dlp

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
    video_url: str
    interval_seconds: int
    downloader_api_key: str
    summary_flag: int
    chunking_method: str
    cobalt_url: str  # cobalt API URL

# URL 확인 함수들
def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url

def is_tiktok_url(url: str) -> bool:
    return "tiktok.com" in url

def is_instagram_url(url: str) -> bool:
    return "instagram.com/reel/" in url or "instagram.com/p/" in url

def is_video_url(url: str) -> bool:
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.ogg', '.webm']
    return any(url.lower().endswith(ext) for ext in video_extensions)

# 유튜브 URL 표준화 함수
def normalize_youtube_url(video_url: str) -> str:
    # youtu.be 형식 처리
    if "youtu.be" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    # youtube.com/embed 형식
    if "youtube.com/embed" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    # youtube.com/shorts 형식
    if "youtube.com/shorts" in video_url:
        video_id = video_url.split('/')[-1].split('?')[0]
        return f"https://www.youtube.com/watch?v={video_id}"
    # youtube.com/watch 형식
    if "youtube.com/watch" in video_url:
        return video_url.split('&')[0]  # 추가 쿼리 매개변수 제거
    # 예상치 못한 형식은 예외 처리
    raise ValueError("Invalid YouTube URL format")

# 인스타그램 URL 표준화 함수
def normalize_instagram_url(video_url: str) -> str:
    if "/reel/" in video_url:
        video_id = video_url.split("/reel/")[-1].split("/")[0]
        return f"https://www.instagram.com/p/{video_id}/"
    return video_url

    

# 영상 다운로드 및 오디오 추출 함수
async def download_video_and_split_audio(video_url: str, interval_seconds: int, downloader_api_key: str, chunking_method: str):
    video_file_extension = None
    caption = None
    video_file = None
    audio_file = None
    chunk_files = []

    try:
        if is_youtube_url(video_url):
            # 유튜브 동영상 처리
            max_retries = 5  # 최대 재시도 횟수
            retry_count = 0
            video_file = None
        
            while retry_count < max_retries:
                try:
                    # API 서버에 POST 요청
                    api_url = cobalt_url
                    headers = {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                    payload = {"url": video_url}
                    
                    response = requests.post(api_url, headers=headers, json=payload)
                    if response.status_code != 200:
                        raise Exception(f"유튜브 API 서버 오류: {response.text}")
                    
                    # API 응답 파싱
                    data = response.json()
                    if data.get('status') == 'error':
                        raise Exception("유튜브 동영상 정보를 가져오는데 실패했습니다.")
                    
                    # 다운로드 URL 가져오기
                    download_url = data.get('url')
                    if not download_url:
                        raise Exception("다운로드 URL을 찾을 수 없습니다.")
                    
                    # 동영상 파일 다운로드
                    video_response = requests.get(download_url, stream=True)
                    if video_response.status_code != 200:
                        raise Exception("동영상 다운로드에 실패했습니다.")
                    
                    # 파일 저장
                    video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
                    with open(video_file, 'wb') as file:
                        for chunk in video_response.iter_content(chunk_size=1024):
                            if chunk:
                                file.write(chunk)
                    
                    # 파일 사이즈 체크
                    file_size = os.path.getsize(video_file)
                    if file_size == 0:
                        print(f"다운로드된 파일 사이즈가 0입니다. 재시도 {retry_count + 1}/{max_retries}")
                        if os.path.exists(video_file):
                            os.remove(video_file)  # 빈 파일 삭제
                        retry_count += 1
                        time.sleep(2)  # 재시도 전 잠시 대기
                        continue
                    
                    print(f"유튜브 동영상 다운로드 완료: {video_file} (크기: {file_size} bytes)")
                    video_file_extension = "mp4"
                    break  # 성공적으로 다운로드 완료
        
                except Exception as e:
                    print(f"유튜브 다운로드 중 에러 발생 (시도 {retry_count + 1}/{max_retries}): {e}")
                    if video_file and os.path.exists(video_file):
                        os.remove(video_file)  # 에러 발생 시 파일 삭제
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(2)  # 재시도 전 잠시 대기
                        continue
                    raise Exception(f"유튜브 동영상을 다운로드하는 중 오류 발생: {e}")
        
            # 최대 재시도 횟수를 초과한 경우
            if retry_count >= max_retries:
                raise Exception(f"최대 재시도 횟수({max_retries})를 초과했습니다. 다운로드에 실패했습니다.")

        elif is_tiktok_url(video_url):
            # 틱톡 영상 처리
            api_url = "https://zylalabs.com/api/4640/tiktok+download+connector+api/5719/download+video"
            api_headers = {'Authorization': f'Bearer {downloader_api_key}'}
            
            response = requests.get(f"{api_url}?url={video_url}", headers=api_headers)
            if response.status_code != 200:
                raise Exception("Failed to retrieve TikTok video information from API")

            data = response.json()
            download_url = data.get('download_url')
            if not download_url:
                raise Exception("Failed to find a download URL for TikTok video")

            video_response = requests.get(download_url, stream=True)
            if video_response.status_code != 200:
                raise Exception("Failed to download TikTok video")

            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            video_file_extension = "mp4"

        elif is_instagram_url(video_url):
            # 인스타그램 영상 처리
            normalized_url = normalize_instagram_url(video_url)
            api_url = f"https://zylalabs.com/api/1943/instagram+reels+downloader+api/2944/reel+downloader?url={normalized_url}"
            headers = {'Authorization': f'Bearer {downloader_api_key}'}
            
            response = requests.get(api_url, headers=headers)
            if response.status_code != 200:
                raise Exception("Failed to retrieve Instagram video information from API")

            data = response.json()
            video_download_url = data.get("video")
            caption = data.get("caption", "")

            if not video_download_url:
                raise Exception("Failed to find a suitable video file for Instagram video")

            video_response = requests.get(video_download_url, stream=True)
            if video_response.status_code != 200:
                raise Exception("Failed to download Instagram video")

            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.mp4")
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            video_file_extension = "mp4"

        elif is_video_url(video_url):
            # 일반 웹 동영상 파일 처리
            video_response = requests.get(video_url, stream=True)
            if video_response.status_code != 200:
                raise HTTPException(status_code=500, detail="제공된 URL에서 동영상 파일을 다운로드하는 데 실패했습니다.")
            
            # 원본 확장자로 파일 저장
            video_file_extension = video_url.split('.')[-1]
            video_file = os.path.join(VIDEO_DIR, f"{uuid.uuid4()}.{video_file_extension}")
            
            with open(video_file, 'wb') as file:
                for chunk in video_response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)

        else:
            raise Exception("지원되지 않는 비디오 URL 형식입니다")

        # 영상에서 오디오 추출
        if video_file_extension and video_file_extension.lower() in ["mp4", "mov", "avi", "mkv", "wmv", "flv", "ogg", "webm"]:
            try:
                audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
                print(f"Processing video file: {video_file}")
                print(f"Creating audio file: {audio_file}")
                
                # AudioFileClip 생성 전 가비지 컬렉션
                import gc
                gc.collect()
                
                print("Starting audio extraction...")
                # 새로운 방식으로 AudioFileClip 생성 및 저장
                video = VideoFileClip(video_file)
                audio = video.audio
                audio.write_audiofile(audio_file)
                
                # 즉시 리소스 정리
                audio.close()
                video.close()
                del audio
                del video
                gc.collect()
                
                print("Audio extraction completed")
                
                # 비디오 파일 정리
                if os.path.exists(video_file):
                    os.remove(video_file)
                    video_file = None
                
                # 오디오 파일 확인
                if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                    raise Exception("Audio extraction failed: Output file is empty or not created")
                
                print("Starting audio processing with PyDub")
                # PyDub으로 오디오 처리
                audio = AudioSegment.from_mp3(audio_file)
                
                if chunking_method == "silence":
                    chunks = silence.split_on_silence(
                        audio,
                        min_silence_len=500,
                        silence_thresh=audio.dBFS-14,
                        keep_silence=250
                    )
                    
                    if not chunks:
                        raise Exception("No audio chunks were created using silence detection")
                    
                    for chunk in chunks:
                        chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
                        chunk.export(chunk_file, format="mp3", parameters=["-ac", "1"])
                        chunk_files.append(chunk_file)

                elif chunking_method == "interval":
                    duration = len(audio)
                    interval_ms = interval_seconds * 1000
                    
                    for start in range(0, duration, interval_ms):
                        end = min(start + interval_ms, duration)
                        chunk = audio[start:end]
                        
                        if len(chunk) > 0:  # 빈 청크 방지
                            chunk_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
                            chunk.export(chunk_file, format="mp3", parameters=["-ac", "1"])
                            chunk_files.append(chunk_file)
                else:
                    raise Exception("Invalid chunking method")

                if not chunk_files:
                    raise Exception("No audio chunks were created")

                print(f"Successfully created {len(chunk_files)} audio chunks")

            except Exception as e:
                print(f"Error during audio processing: {str(e)}")
                raise Exception(f"Failed to extract audio: {str(e)}")

    except Exception as e:
        print(f"Error in main process: {str(e)}")
        # 에러 발생 시 모든 임시 파일 정리
        if video_file and os.path.exists(video_file):
            os.remove(video_file)
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        raise Exception(str(e))

    finally:
        # 메인 오디오 파일 정리
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

    return chunk_files, caption

# 시간 형식 변환 함수
def seconds_to_timecode(seconds: int) -> str:
    return str(datetime.timedelta(seconds=seconds))

# 텍스트 요약 함수
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

# 오디오 청크 전사 함수
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
            os.remove(audio_chunks[i])  # 사용한 오디오 파일 삭제

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
        elif is_video_url(request.video_url):
            normalized_video_url = request.video_url  # 일반 비디오 URL은 그대로 사용
        else:
            normalized_video_url = request.video_url

        audio_chunks, caption = await download_video_and_split_audio(
            normalized_video_url, 
            request.interval_seconds, 
            request.downloader_api_key, 
            request.chunking_method
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

    try:
        if request.summary_flag == 1:
            summary_text = await summarize_text(request.api_key, transcribed_texts, chunk_times)
            if caption:
                summary_text = f"[caption]: {caption}\n" + summary_text
            return {"summary": summary_text}
        else:
            full_transcription = "\n".join(
                [f"{chunk_times[i]}: {transcribed_texts[i]}" for i in range(len(transcribed_texts))]
            )
            if caption:
                full_transcription = f"[caption]: {caption}\n" + full_transcription
            return {"transcription": full_transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing results: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
