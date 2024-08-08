def text_translate(text, target_lang,pipe):
    translated_text =pipe(text, forced_bos_token_id=pipe.tokenizer.get_lang_id(lang=target_lang))
    generated_text = translated_text[0]['generated_text']
    return generated_text