from moviepy.editor import VideoFileClip
from  text_to_translation import text_translate
import tempfile
translator = text_translate
def video_translate(video_path, target_language,model,pipe):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_audio_file:
        video = VideoFileClip(video_path)
        audio = video.audio
        temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path)
        audio.close()
        video.close()
        audio_clip = temp_audio_path
        segments, _ = model.transcribe(audio_clip)
        translated_text = ""
        for segment in segments:
            translated_text += translator(segment.text,target_language,pipe) + " "
    return translated_text.strip()