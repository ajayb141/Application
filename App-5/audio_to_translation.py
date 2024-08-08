from text_to_translation import text_translate

translator = text_translate
def audio_to_translate(input, target_language,model,pipe):
    seg = " "
    segments, info = model.transcribe(input)
    for segment in segments:
        seg += "%s " % (segment.text)
    translated_seg = translator(seg,target_language,pipe)
    return translated_seg