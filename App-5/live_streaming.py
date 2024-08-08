final=[]
def  audio_streaming(input,model):
   seg=" "
   segments, info = model.transcribe((input))
   for segment in segments:
     seg +="%s" % (segment.text)
   final.append(seg)
   output_string = ' '.join(final)
   return output_string