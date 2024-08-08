import whisper_timestamped as whisper
model = whisper.load_model("tiny")
def timestamp(input_audio):
    audio = whisper.load_audio(input_audio)
    result = whisper.transcribe(model, audio, language="en")
    return result