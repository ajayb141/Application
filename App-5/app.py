import gradio as gr
from audio_to_translation import audio_to_translate
from video_to_translation import video_translate
from text_to_translation import text_translate
from microphone_to_translation import microphone_to_translation
from live_streaming import audio_streaming
from text_to_speech import text_to_speech
from timestamping import timestamp

from faster_whisper import WhisperModel
from transformers import pipeline
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
model = WhisperModel("large-v2")
pipe = pipeline(task='text2text-generation', model='facebook/m2m100_418M')

languages = {
    "Afrikaans": "af", "Amharic": "am", "Arabic": "ar", "Asturian": "ast", "Azerbaijani": "az",
    "Bashkir": "ba", "Belarusian": "be", "Bulgarian": "bg", "Bengali": "bn", "Breton": "br",
    "Bosnian": "bs", "Catalan": "ca", "Cebuano": "ceb", "Czech": "cs", "Welsh": "cy",
    "Danish": "da", "German": "de", "Greeek": "el", "English": "en", "Spanish": "es", "Estonian": "et",
    "Persian": "fa", "Fulah": "ff", "Finnish": "fi", "French": "fr", "Western Frisian": "fy", "Irish": "ga",
    "Gaelic": "gd", "Galician": "gl", "Gujarati": "gu", "Hausa": "ha", "Hebrew": "he", "Hindi": "hi",
    "Croatian": "hr", "Haitian": "ht", "Hungarian": "hu", "Armenian": "hy", "Indonesian": "id", "Igbo": "ig",
    "Iloko": "ilo", "Icelandic": "is", "Italian": "it", "Japanese": "ja", "Javanese": "jv", "Georgian": "ka",
    "Kazakh": "kk", "Central Khmer": "km", "Kannada": "kn", "Korean": "ko", "Luxembourgish": "lb", "Ganda": "lg",
    "Lingala": "ln", "Lao": "lo", "Lithuanian": "lt", "Latvian": "lv", "Malagasy": "mg", "Macedonian": "mk",
    "Malayalam": "ml", "Mongolian": "mn", "Marathi": "mr", "Malay": "ms", "Burmese": "my", "Nepali": "ne",
    "Dutch": "nl", "Norwegian": "no", "Northern Sotho": "ns", "Occitan": "oc", "Oriya": "or",
    "Panjabi": "pa", "Polish": "pl", "Pushto": "ps", "Portuguese": "pt", "Romanian": "ro",
    "Russian": "ru", "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk", "Slovenian": "sl", "Somali": "so",
    "Albanian": "sq", "Serbian": "sr", "Swati": "ss", "Sundanese": "su", "Swedish": "sv", "Swahili": "sw",
    "Tamil": "ta", "Thai": "th", "Tagalog": "tl", "Tswana": "tn", "Turkish": "tr", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Wolof": "wo", "Xhosa": "xh", "Yiddish": "yi",
    "Yoruba": "yo", "Chinese": "zh", "Zulu": "zu"
}



def audio_text(audio_file, target_language):
    target_language_name = list(languages.values())[list(languages.keys()).index(target_language)]
    return audio_to_translate(audio_file, target_language_name,model,pipe)
def video_translation(videofile,target_lang):
    target_language_name = list(languages.values())[list(languages.keys()).index(target_lang)]
    return video_translate(videofile,target_language_name,model,pipe)
def text_translation(text,targetlang):
    target_language_name = list(languages.values())[list(languages.keys()).index(targetlang)]
    return text_translate(text,target_language_name,pipe)
def microphone_text(audio,targetlang):
    target_language_name = list(languages.values())[list(languages.keys()).index(targetlang)]
    return microphone_to_translation(audio,target_language_name,model,pipe)
def microphone_live(audio):
    return audio_streaming(audio,model)
def text_speech(text,audio):
   return text_to_speech(text,audio,tts)
def whisper_timestamp(input):
    return timestamp(input)

with gr.Blocks() as trail:
   with gr.Tab("audio to translation"):
        audio_input = [gr.Audio(label="Speak Here",type="filepath"),
               gr.Dropdown(label="Target Language", choices=list(languages.keys()))]
        text_output = gr.Textbox(label="Transcription")
        gr.Interface(
            fn=audio_text,
            inputs=audio_input,
            outputs=text_output,
            examples=[["examples/sample1.mp3"],
                      ["examples/sample2.mp3"]],
            title="Audio to text Translator",
            allow_flagging=False
        )
   with gr.Tab("video to translation"):
    video_input = [gr.Video(label="Put your video here"),
               gr.Dropdown(label="Target Language", choices=list(languages.keys()))]
    text_output = gr.Textbox(label="Transcription")
    gr.Interface(
                fn=video_translation,
                inputs=video_input,
                outputs=text_output,
                examples=[["examples/sample1.mp4"],
                          ["examples/sample2.mp4"],
                          ["examples/sample3.mp4"]],
                title="video-to-Text Translator",
                allow_flagging=False
            )
   with gr.Tab("text to translation"):
    audio_input = [gr.Text(label="Enter your text here"),
               gr.Dropdown(label="Target Language", choices=list(languages.keys()))]
    text_output = gr.Textbox(label="Transcription")
    gr.Interface(
                fn=text_translation,
                inputs=audio_input,
                outputs=text_output,
                examples=[["The definition of technology is the application of scientific knowledge for practical purposes or applications. Technology uses scientific principles, and applies them to change the environment in which humans live. Technology can also use scientific principles to advance industry or other human constructions."],
                          ["आर्टिफिशियल इंटेलिजेंस (AI) क्या है और कैसे काम करता है? AI यानी Artificial Intelligence को हिंदी में कृत्रिम बुद्धिमत्ता कहते हैं, जिसका मतलब है बनावटी यानी कृत्रिम तरीके से विकसित की गई बौद्धिक क्षमता. आर्टिफिशियल इंटेलिजेंस कम्प्यूटर साइंस की एक एडवांस्ड शाखा है."],
                          ["每个人都有自己的梦想，长大后要实现。有些孩子想变得有钱，以便可以买到任何东西，有些孩子想成为医生，律师或工程师。但是只有您知道，要实现这些目标，您必须努力工作并保持专注。在这篇关于我的梦想的文章中，我们将讨论有助于实现我的梦想的基本事物。"]],
                title="Text to Translation",
                description="Enter the text to Translate.",
                allow_flagging=False
            )
   with gr.Tab("microphone to translation"):
    inputs=[gr.Audio(sources="microphone",type="filepath"),
                gr.Dropdown(label="Target Language", choices=list(languages.keys()))]
    text_output = gr.Textbox(label="Transcription")
    gr.Interface(
                fn=microphone_text,
                inputs=inputs,
                outputs=text_output,
                title="Speech-to-Text Converter",
                description="Speak into the microphone and get text Translation.",
                allow_flagging=False
        )
   with gr.Tab("Microphone to Streaming Transcription"):
    audio_input=[gr.Audio(streaming=True,type="filepath")]
    text_output = gr.Textbox(label="Transcription")
    gr.Interface(
                fn=microphone_live,
                inputs=audio_input,
                outputs=text_output,
                title="Speech-to-Text Converter",
                description="Speak into the microphone and get text Transcription.",
                allow_flagging=False,
                live="true"
        )
   with gr.Tab("Text to speech converter"):
    input=[gr.Textbox(label="Enter Text"),gr.Audio(type="filepath",label="Select Audio")]
    audio_output=gr.Audio(type="filepath", label="Generated Speech")
    gr.Interface(
                fn=text_speech,
                inputs=input,
                outputs=audio_output,
                examples=[["सुप्रभातम आपका दिन शुभ हो"],["The soft whispers of nature create a calming melody, inviting a moment of reflection in the quiet canvas of dusk."]],
                title="Text-to-Speech",
                description="upload an audio file for the speaker to generated text.",
                allow_flagging=False
        )
   with gr.Tab("Whisper Timestamp"):
    aud_input=gr.Audio(type="filepath",label="upload audio")
    text_output=gr.TextArea(type="text")
    gr.Interface(
            fn=whisper_timestamp,
            inputs=aud_input,
            outputs=text_output,
            title="Text to Audio converter",
            examples=[["examples/sample1.mp3"],["examples/sample2.mp3"]],
            description="Type yours and get Timestamp.",
            allow_flagging=False,
        )
trail.launch()