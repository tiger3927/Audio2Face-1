import wave

with wave.open(r'tts_audio/20230224154812/20230224154812_tts.wav', 'r') as wav_file:
    frames = wav_file.getnframes()
    rate = wav_file.getframerate()
    duration = frames / float(rate)
    print(f"音频文件的时长为：{duration:.2f} 秒")
