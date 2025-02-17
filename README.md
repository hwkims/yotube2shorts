알겠습니다! Hugging Face Transformers를 사용하여 요약하고, YouTube 트렌드에 맞는 화려한 자막 스타일(색상, 폰트)을 적용하고, CUDA를 활용하여 GPU 가속을 사용하는 전체 코드를 제공하겠습니다.  API 키는 사용하지 않습니다.

**주요 변경 사항:**

*   **Hugging Face Transformers 요약:** `transformers` 라이브러리의 `pipeline`을 사용하여 요약을 수행합니다.  `kykim/bertshared-kor-base` 모델을 한국어 요약에 사용합니다. (다른 모델로 변경 가능)
*   **화려한 자막 스타일:**
    *   `moviepy`의 `TextClip` 설정을 조정하여 YouTube 쇼츠에서 자주 보이는 자막 스타일을 구현합니다.
    *   랜덤한 색상 조합 (빨강, 노랑, 초록, 흰색, 검정 등)을 사용합니다.
    *   굵고 눈에 띄는 폰트 (예: `Impact`, `Arial-Black` 등)를 사용합니다.  (시스템에 설치된 폰트 사용)
    *   검정색 테두리(outline)를 추가하여 가독성을 높입니다.
*   **CUDA (GPU) 사용:**
    *   `torch.cuda.is_available()`을 확인하여 CUDA 사용 가능 여부를 확인합니다.
    *   Transformers 모델을 GPU로 이동시킵니다 (`.to('cuda')`).
*   **임시 파일 관리 개선:**  더욱 철저하게 임시 파일을 관리하고, 에러 발생 시에도 삭제하도록 합니다.
*   **`extract_audio` 함수 추가:**  동영상에서 오디오를 추출하는 별도 함수를 만듭니다.
*   **Whisper 사용**: 오디오를 텍스트로 변환하는데 사용

```python
import moviepy.editor as mp
from pytube import YouTube
import subprocess
import os
import re
from moviepy.config import change_settings
from youtube_comment_downloader import YoutubeCommentDownloader
from gtts import gTTS
import random
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import speech_recognition as sr
from tqdm import tqdm


# FFmpeg 경로 설정
change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})

# CUDA 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hugging Face Transformers 요약 모델 및 토크나이저 초기화
try:
    summarizer = pipeline("summarization", model="kykim/bertshared-kor-base", tokenizer="kykim/bertshared-kor-base", device=0 if device == "cuda" else -1) # device=0은 첫번째 gpu를 의미

except Exception as e:
    print(f"Error loading summarization model: {e}.  Check model name and internet connection.")
    exit()


def download_youtube_video(youtube_url, output_path="temp"):
    """YouTube 동영상을 다운로드합니다."""
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        filepath = stream.download(output_path)
        return filepath
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def extract_audio(video_path, output_audio_path):
    """동영상에서 오디오를 추출합니다."""
    try:
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le') # 고품질 오디오
        audio_clip.close()
        video_clip.close()
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def get_youtube_comments(youtube_url, max_comments=5):
    """YouTube 동영상의 댓글을 가져옵니다."""
    try:
        downloader = YoutubeCommentDownloader()
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
        if not match:
            print("Error: Invalid YouTube URL")
            return []
        video_id = match.group(1)

        comments = list(downloader.get_comments_from_url(youtube_url, sort_by=1))
        return [comment['text'] for comment in comments[:max_comments]]

    except Exception as e:
        print(f"Error getting comments: {e}")
        return []



def create_text_clip(text, fontsize=48, color='white', bg_color=None,
                    duration=3, font='Impact', stroke_color='black', stroke_width=2):
    """화려한 스타일의 텍스트 클립을 생성합니다."""
    # YouTube 트렌드 색상
    colors = ['red', 'yellow', 'lime', 'white', 'cyan']  # 더 다양한 색상 추가
    text_color = random.choice(colors)

    return mp.TextClip(text, fontsize=fontsize, color=text_color,
                       bg_color=bg_color, size=(1080 * 0.9, 1920 * 0.3),
                       method='caption', align='center', interline=-5,
                       font=font, stroke_color=stroke_color, stroke_width=stroke_width)




def create_speech_from_text(text, output_path):
    """gTTS를 사용하여 텍스트를 음성으로 변환합니다."""
    try:
        tts = gTTS(text=text, lang='ko')
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error creating speech: {e}")
        return None


def transcribe_audio(audio_path):
    """Whisper를 사용하여 오디오를 텍스트로 변환합니다."""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
        text = r.recognize_whisper(audio, language="korean") #한국어
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


def summarize_with_huggingface(text, max_length=150, min_length=30):
    """Hugging Face Transformers를 사용하여 텍스트를 요약합니다."""
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error summarizing with Hugging Face: {e}")
        return ""




def create_shorts(youtube_url, output_filename="shorts.mp4"):
    """YouTube 동영상을 다운로드하고, 요약하고, 댓글을 가져와서 쇼츠를 생성합니다."""

    temp_dir = "temp"
    video_path = download_youtube_video(youtube_url, temp_dir)
    if not video_path:
        return

    audio_path = os.path.join(temp_dir, "temp_audio.wav")
    extract_audio(video_path, audio_path)

    try:
        # 1. 오디오를 텍스트로 변환 (Whisper 사용)
        full_text = transcribe_audio(audio_path)
        if not full_text:
            print("Error: Could not transcribe audio.")
            return

        # 2. 텍스트 요약 (Hugging Face Transformers)
        summary = summarize_with_huggingface(full_text)
        if not summary:
            print("Error: Could not summarize text.")
            return
        print(f"Summary: {summary}")

        # 3. 댓글 가져오기
        comments = get_youtube_comments(youtube_url)
        if not comments:
            print("No comments found or error getting comments.")


        # 4. 요약 TTS 생성
        summary_audio_path = os.path.join(temp_dir, "summary_audio.mp3")
        create_speech_from_text(summary, summary_audio_path)
        summary_audio = mp.AudioFileClip(summary_audio_path)


        # 5. 쇼츠 클립 생성 및 배치

        clips = []

        # 요약 클립 추가 (자막 + 오디오)
        summary_text_clip = create_text_clip(summary, duration=summary_audio.duration)
        summary_clip = summary_text_clip.set_audio(summary_audio)
        clips.append(summary_clip)


        # 댓글 클립 추가
        for i, comment in enumerate(comments):
            comment_audio_path = os.path.join(temp_dir, f"comment_audio_{i}.mp3")
            create_speech_from_text(comment, comment_audio_path)
            comment_audio = mp.AudioFileClip(comment_audio_path)
            comment_text_clip = create_text_clip(comment, duration=comment_audio.duration)
            comment_clip = comment_text_clip.set_audio(comment_audio)
            clips.append(comment_clip)


        # 6. 비디오 클립에서 짧은 구간 추출 (선택적)
        #   전체 비디오를 사용하면 쇼츠 길이가 너무 길어질 수 있으므로,
        #   일부 구간만 사용하는 것이 좋습니다.
        video_clip = mp.VideoFileClip(video_path).subclip(5, 10)  # 예시: 5초~10초 구간
        clips.insert(1, video_clip) # 요약과 첫번째 댓글 사이에 삽입


        # 7. 클립들을 합쳐서 최종 쇼츠 생성
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        final_clip = final_clip.resize((1080, 1920))  # 9:16 비율로 조정
        final_clip.write_videofile(output_filename, fps=24, codec="libx264", audio_codec="aac")


    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 임시 파일 삭제
        print("Cleaning up temporary files...")
        try:
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                os.remove(file_path)
            os.rmdir(temp_dir)  # 빈 temp 디렉토리 삭제
            print("Cleanup complete.")

        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")




if __name__ == "__main__":
    # youtube_url = "https://www.youtube.com/watch?v=ojUVHLBbYVA"  # 예시 URL
    youtube_url = input("Enter the YouTube video URL: ")
    create_shorts(youtube_url)
```

**코드 사용 설명:**

1.  **필요한 라이브러리 설치:**
    ```bash
    pip install moviepy pytube youtube-comment-downloader gTTS transformers speechrecognition tqdm
    ```
2.  **FFmpeg 설치:**  시스템에 FFmpeg가 설치되어 있어야 합니다. (설치 방법은 운영체제에 따라 다릅니다.)
3.  **코드 실행:**  `create_shorts` 함수에 YouTube URL을 입력하여 실행합니다.
4.  **CUDA 확인:** 코드가 CUDA를 올바르게 사용하는지 확인합니다. (GPU 사용량 모니터링)
5.  **폰트:**  `create_text_clip` 함수에서 `font` 매개변수를 시스템에 설치된 다른 굵은 폰트로 변경할 수 있습니다.  (예: `Arial-Black`, `Helvetica-Bold`, `Impact` 등)
6. **모델 변경:** `kykim/bertshared-kor-base`대신 다른 한국어 요약 모델 사용가능.

**실행 흐름:**

1.  YouTube 동영상 다운로드 (`download_youtube_video`)
2.  동영상에서 오디오 추출 (`extract_audio`)
3.  Whisper를 사용하여 오디오를 텍스트로 변환 (`transcribe_audio`)
4.  Hugging Face Transformers를 사용하여 텍스트 요약 (`summarize_with_huggingface`)
5.  YouTube 댓글 가져오기 (`get_youtube_comments`)
6.  요약과 각 댓글에 대한 TTS 음성 생성 (`create_speech_from_text`)
7.  화려한 스타일의 자막 클립 생성 (`create_text_clip`)
8.  비디오 클립에서 짧은 구간 추출 (선택 사항)
9.  모든 클립(요약, 비디오, 댓글)을 합쳐서 최종 쇼츠 비디오 생성
10. 임시 파일 삭제

이 코드는 YouTube 쇼츠 제작을 위한 강력하고 유연한 프레임워크를 제공합니다.  필요에 따라 각 부분을 수정하고 확장하여 더욱 개성 있는 쇼츠를 만들 수 있습니다.
