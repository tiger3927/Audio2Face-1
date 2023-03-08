import os
import numpy as np
import scipy.io.wavfile as wavfile
import librosa

def wav2npy(wav_path, b_frames = -1, fps = -1):
    """Convert wav to numpy file in a dictionary
    Args:
        wav_path: str, wav file path
        b_frames: int, if is training data, reverse the frames to calculate fps
        fps: int, if is inference data, set the fps
    """
    # read wav file
    rate, signal = wavfile.read(wav_path)

    # check if inference or training data
    frames_per_second = b_frames / (float(len(signal)) / rate) if fps < 0 else fps
    audio_frameNum = int(len(signal) / rate * frames_per_second) if b_frames < 0 else b_frames
    print('rate:', rate, 'audio_frameNum: ', audio_frameNum, 'fps', frames_per_second)

    # add padding audio
    chunks_length = 260
    a = np.zeros(chunks_length * rate // 1000, dtype=np.int16)
    signal = np.hstack((a, signal, a))

    frames_step = 1000.0 / frames_per_second
    rate_kHz = int(rate / 1000)

    # cut the frames
    audio_frames = [signal[int(i * frames_step * rate_kHz): 
                            int((i * frames_step + chunks_length * 2) * rate_kHz)]
                                for i in range(audio_frameNum)]
    
    inputData_array = np.zeros(shape=(1, 32, 64))
    for i in range(len(audio_frames)):
        audio_frame = audio_frames[i]

        overlap_frames_apart = 0.008 * chunks_length / 260
        overlap = int(rate * overlap_frames_apart)
        frameSize = int(rate * overlap_frames_apart * 2)
        numberOfFrames = 64

        # initiate a 2D array with numberOfFrames rows and frame size columns
        frames = np.ndarray((numberOfFrames, frameSize))
        for k in range(0, numberOfFrames):
            for i in range(0, frameSize):
                if ((k * overlap + i) < len(audio_frame)):
                    frames[k][i] = audio_frame[k * overlap + i]
                else:
                    frames[k][i] = 0

        frames *= np.hanning(frameSize)
        frames_lpc_features = []

        # linear predictive coding
        for k in range(0, numberOfFrames):
            a = frames[k]
            b = librosa.lpc(a, order=31)
            frames_lpc_features.append(b)

        # (64, 32)
        image_temp1 = np.array(frames_lpc_features)
        image_temp2 = image_temp1.transpose()
        image_temp3 = np.expand_dims(image_temp2, axis=0)
        inputData_array = np.concatenate((inputData_array, image_temp3), axis=0)

    # expand dims to (,32,64,1)
    return np.expand_dims(inputData_array[1:], axis=3)

def wavs2npys(input_wav_dir, input_label_dir, output_dir):
    """Convert wav to numpy file in a dictionary
    Args:
        input_wav_dir: str, data dir with wav file
        input_label_dir: str, data dir with bs label file
        output_dir: str, ouput dir
    """
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(os.path.join(output_dir))

    # list all txt file
    name_list = [fn.split('.')[0] for fn in os.listdir(input_wav_dir) if fn.split('.')[1] == 'txt']
    wav_path = [os.path.join(input_wav_dir, file + '.wav') for file in name_list]
    label_path = [os.path.join(input_label_dir, 'BS_' + file + '.npy') for file in name_list]
    save_path = [os.path.join(output_dir, file + '.npy') for file in name_list]

    for i in range(len(name_list)):
        # get frames num from label
        frames_num = len(np.load(label_path[i]))

        inputData_array = wav2npy(wav_path[i], frames_num)

        np.save(save_path[i], inputData_array)
        print(inputData_array.shape, 'npy array saved to {}'.format(save_path[i]))

if __name__ == '__main__':
    input_wav_dir = r'./Participant Data'
    input_label_dir = r'./bs_value'
    output_dir  = r'./lpc'
    print('input wav dir', input_wav_dir)
    print('input_label_dir', input_wav_dir)
    wavs2npys(input_wav_dir, input_label_dir, output_dir)