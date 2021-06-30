# pip install pyTelegramBotAPI
import telebot
import librosa
import numpy as np
import torch
import torchaudio

from models import Wav2VecClassifierModelMean6 as Wav2VecClassifierModel
from processors import CustomWav2Vec2Processor


class SpeechClassification:

    def __init__(self, path=None):
        self.model = Wav2VecClassifierModel.from_pretrained(path).to("cuda")
        self.processor = CustomWav2Vec2Processor.from_pretrained(path)

    def classify(self, wav_file):
        return self.predict(self.load_file_to_data(wav_file),
                            self.model, self.processor)

    def load_file_to_data(self, file, srate=16_000):
        batch = {}
        speech_array, sampling_rate = torchaudio.load(file)
        speech_array = speech_array[0].numpy()[:10 * sampling_rate]
        batch["speech"] = librosa.resample(np.asarray(speech_array), sampling_rate, srate)
        batch["sampling_rate"] = srate
        return batch

    def predict(self, data, model, processor):
        max_length = 160000
        features = processor(data["speech"][:max_length],
                             sampling_rate=data["sampling_rate"],
                             max_length=max_length,
                             pad_to_multiple_of=max_length,
                             padding=True, return_tensors="pt")

        input_values = features.input_values.to("cuda")
        attention_mask = features.attention_mask.to("cuda")
        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)

        dialects = ['teens', 'twenties', 'thirties', 'fourties', 'fifties', 'sixties-nineties']

        softmax = torch.nn.Softmax(dim=-1)
        probs = softmax(outputs['logits'])
        top_prob, top_lbls = torch.topk(probs[0], 2)
        return {"x": dialects[top_lbls[0]], dialects[top_lbls[0]]: format(float(top_prob[0]), '.2f')}


if __name__ == '__main__':
    classifier = SpeechClassification(path="/cluster/home/reisedom/data_german/model-saves/age/max-samples-correct/3/4000/")
    bot = telebot.TeleBot("1853091605:AAEvSc0m1OA0o1hbajwIBdg1wrESeVxUyLE")
    print("bot started ...")
    path = "/cluster/home/reisedom/data_german/bot_age_inputs/"

    @bot.message_handler(commands=['start', 'help'])
    def send_welcome(message):
        bot.reply_to(message, "Hi I am the great Age Identifier, please send me your voice so I can determine your age.")


    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, "Please send me your voice so I can identify your age.")


    @bot.message_handler(content_types=['voice'])
    def voice_processing(message):
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        file_name = message.chat.first_name + (str(message.id) if message.id else "_unknown")
        file_ending = ".mp3"

        full_path = path + file_name + file_ending
        print("received file: " + full_path)

        with open(full_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        prediction = classifier.classify(full_path)
        print("identified file " + full_path + " as: " + prediction['x'])

        bot.reply_to(message, "Thanks, based on your voice, I think you are in your " + prediction["x"] + ".")


    bot.polling()
    print("bot stopped ...")
