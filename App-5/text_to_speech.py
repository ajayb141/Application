def text_to_speech(text, speaker_audio_path,tts):
    output_file_path = "output.wav"
    language = "en"
    speaker_wav = speaker_audio_path
    tts.tts_to_file(text=text, file_path=output_file_path, speaker_wav=speaker_wav, language=language)
    return output_file_path