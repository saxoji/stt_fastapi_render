import os
import uuid
from yt_dlp import YoutubeDL
from moviepy.editor import AudioFileClip
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import JSONResponse
from typing import List

app = FastAPI()

# 인증키
REQUIRED_AUTH_KEY = "linkbricks-saxoji-benedict-ji-01034726435!@#$%231%$#@%"

# Directory to save the files
VIDEO_DIR = "video"
AUDIO_DIR = "audio"

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# 모델 정의
class YouTubeAudioRequest(BaseModel):
    api_key: str
    auth_key: str
    youtube_url: str
    interval_minute: int

# 영상 다운로드 후 오디오 추출
def download_video_and_extract_audio(youtube_url: str) -> str:
    ydl_opts = {
        'outtmpl': os.path.join(VIDEO_DIR, '%(title)s.%(ext)s'),  # 비디오 저장 경로 및 이름 지정
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_file = ydl.prepare_filename(info)  # 다운로드한 영상 파일 경로
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading video: {str(e)}")

    # 영상에서 오디오 파일 추출
    try:
        audio_file = os.path.join(AUDIO_DIR, f"{uuid.uuid4()}.mp3")
        video_clip = AudioFileClip(video_file)
        video_clip.write_audiofile(audio_file)
        video_clip.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting audio: {str(e)}")

    # 영상 파일 삭제 (필요하다면)
    os.remove(video_file)

    return audio_file

# POST 요청 처리
@app.post("/process_youtube_audio/")
async def process_youtube_audio(request: YouTubeAudioRequest):
    # 인증키 확인
    if request.auth_key != REQUIRED_AUTH_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication key")

    # 유튜브 영상 다운로드 후 오디오 추출
    try:
        audio_file = download_video_and_extract_audio(request.youtube_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video or audio: {str(e)}")

    return JSONResponse(content={"audio_file_path": audio_file})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
