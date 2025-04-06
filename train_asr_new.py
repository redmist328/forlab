import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import torch.nn.functional as F
from pypinyin import pinyin, lazy_pinyin
import re
import os
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
import glob
from librosa.filters import mel as librosa_mel_fn
from torchaudio.transforms import MelSpectrogram, Resample
def load_wav(full_path, sample_rate):
    data, _ = librosa.load(full_path, sr=sample_rate, mono=True)
    return data

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):

    mel = librosa_mel_fn(sr=sampling_rate,n_fft= n_fft,n_mels= num_mels,fmin= fmin,fmax= fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis, spec)
    spec = spectral_normalize_torch(spec)

    return spec #[batch_size,n_fft/2+1,frames]
class ThchsData(Dataset):
    def __init__(self, wav_dir, sample_rate=16000):
        """
        Args:
            wav_dir (str): 音频文件夹路径
            transcription_dir (str): 转录文本文件夹路径
            sample_rate (int): 重采样的目标采样率
            transform (callable, optional): 音频和文本的预处理操作
        """
        self.wav_dir = wav_dir
        self.sample_rate = sample_rate
        self.transcriptions={}
        # 获取所有音频文件
        
        self.texts = []
        self.paths = []
        text_paths = glob.glob(self.wav_dir)
        for path in text_paths:
            with open(path, 'r', encoding='utf8') as fr:
                lines = fr.readlines()
                line = lines[0].strip('\n').replace(' ', '').replace('../data','./data_thchs30/data')
                with open(line, 'r', encoding='utf8') as fr2:
                    lines2 = fr2.readlines()
                    self.texts.append(lines2[0].strip('\n').replace(' ', ''))
                    self.paths.append(path.rstrip('.trn'))
        chars = {}
        for text in self.texts:
            for c in text:
                chars[c] = chars.get(c, 0) + 1

        chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
        chars = [char[0] for char in chars]
        
        # print(len(chars), chars[:100])
        self.char2id =torch.load('cha2id.pth')
        self.id2char =torch.load('id2char.pth')
        new_char2id = {char: id + 1 for char, id in self.char2id.items()}
        self.char2id = new_char2id

        # 更新 self.id2char 字典
        new_id2char = {id + 1: char for id, char in self.id2char.items()}
        self.id2char = new_id2char
        self.blank_index=0
        
        self.char2id['<blank>'] = self.blank_index
        self.id2char[self.blank_index] = '<blank>'
        self.eos_index=len(self.char2id)
        self.char2id['<eos>'] = self.eos_index
        self.id2char[self.eos_index]='<eos>'
        # print(len(self.char2id))
        # exit()
        # self.char2id = {c: i for i, c in enumerate(chars)}
        # self.id2char = {i: c for i, c in enumerate(chars)}
        # torch.save(self.char2id,'cha2id.pth')
        # torch.save(self.id2char,'id2char.pth')
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        wav_path = self.paths[idx]
        # wav_path = os.path.join(self.wav_dir, wav_file)
        # 加载音频
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 如果采样率不符合要求，进行重采样
        if sample_rate != self.sample_rate:
            resampler = Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        # print(transcription)
        # 应用预处理操作（如Mel Spectrogram）
        path1=wav_path.replace('train/','mfcc/').replace('wav','npy')
        fea=torch.FloatTensor(np.load(path1))
        # phone=torch.LongTensor([self.char2id[c] for c in self.texts[idx]])
        phone_list = []  # 用于存储处理后的字符表示

        for i in range(len(self.texts[idx])):
            char = self.texts[idx][i]
            phone_list.append(self.char2id[char])  # 将字符映射为对应的id
            
            # 如果是最后一个字符，则跳出循环
            if i < len(self.texts[idx]) - 1 and self.texts[idx][i] == self.texts[idx][i + 1]:
                phone_list.append(0)  # 如果当前字符和下一个字符相同，则在中间插入 0

        # 将处理后的 phone_list 转换为 LongTensor
        phone = torch.LongTensor(phone_list)
        # new_tensor = torch.zeros(2 * len(phone), dtype=phone.dtype)
        # new_tensor[::2] = phone
        # new_tensor[1::2] = self.blank_index
        # print(new_tensor)
        transcription=self.texts[idx]
        return fea, phone,waveform,transcription
def collate_fn(batch):
    # 获取音频和文本数据
    feas, phones ,waveform,transcription= zip(*batch)
    # print(transcriptions.shape)
    # print(transcriptions.shape)
    input_lengths=[]
    label_lengths=[]
    # 找到最大长度的音频（填充时用到）
    max_fea_len = max([fea.size(0) for fea in feas])
    for fea in feas:
        input_lengths.append(fea.size(0))
    # 填充音频
    padded_feas = []
    for fea in feas:
        fea=fea.transpose(0,1)
        # print(fea.shape)
        padding = max_fea_len - fea.size(1)
        padded_feas.append(torch.nn.functional.pad(fea, (0, padding)))
    
    # 找到最长文本的长度
    max_phone_len = max([len(phone) for phone in phones])
    for phone in phones:
        label_lengths.append(len(phone)+1) 
    # 填充文本
    padded_phones = []
    
    max_phone_len = max([phone.size(0) for phone in phones])
    for phone in phones:
        # 假设文本是字符级的，可以直接填充空格
        padding = max_phone_len - phone.size(0)+1
        padded_phones.append(torch.nn.functional.pad(phone, (0, padding), value=2666))
    
    # 转换为张量
    padded_fea = torch.stack(padded_feas, dim=0)
    # print(padded_fea.shape)
    padded_phones = torch.stack(padded_phones, dim=0)
    
    return padded_fea, padded_phones,input_lengths,label_lengths



class SimplifiedASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_classes, hidden_size=128, num_layers=2):
        super(SimplifiedASRModel, self).__init__()
        # 添加 GRU 层
        self.gru = nn.GRU(mfcc_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 调整维度以适应 GRU 输入 [batch_size, seq_len, input_size]
        x = x.permute(0, 2, 1)

        # 通过 GRU 层
        output, _ = self.gru(x)

        # 通过全连接层
        y_pred = self.fc(output)

        # 调整维度以适应后续 CTCLoss 计算 [seq_len, batch_size, num_classes]
        y_pred = y_pred.permute(1, 0, 2)

        return y_pred
class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 1, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)

    def forward(self, x):
        hf = torch.tanh(self.bn1(self.conv1(x)))
        hg = torch.sigmoid(self.bn1(self.conv1(x)))
        h0 = hf * hg
        ha = torch.tanh(self.bn2(self.conv2(h0)))
        hs = torch.tanh(self.bn2(self.conv2(h0)))
        return ha + x, hs

class ResBlock(nn.Module):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(filters, filters, kernel_size, stride=1, padding='same', dilation=dilation_rate)
        self.bn1 = nn.BatchNorm1d(filters)
        self.conv2 = nn.Conv1d(filters, filters, 1, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        hf = self.leaky_relu(self.bn1(self.conv1(x)))
        hg = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = hf * hg
        ha = self.leaky_relu(self.bn2(self.conv2(h0)))
        hs = self.leaky_relu(self.bn2(self.conv2(h0)))
        return ha + x, hs

class ASRModel(nn.Module):
    def __init__(self, mfcc_dim, num_blocks, filters, num_classes, hidden_size=128, num_layers=2):
        super(ASRModel, self).__init__()
        self.conv1 = nn.Conv1d(mfcc_dim, filters, 3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(filters)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.blocks = []
        for i in range(num_blocks):
            for r in [1, 2, 4]:
                self.blocks.append(ResBlock(filters, 7, r))
        self.blocks = nn.ModuleList(self.blocks)
        self.conv2 = nn.Conv1d(filters, filters, 3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(filters)
        self.dropout2 = nn.Dropout(0.2)

        # 添加GRU层
        self.gru = nn.GRU(filters, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = self.leaky_relu(self.bn1(self.conv1(x)))
        h0 = self.dropout1(h0)
        shortcut = []
        for block in self.blocks:
            h0, s = block(h0)
            shortcut.append(s)
        h1 = self.leaky_relu(sum(shortcut))
        h1 = self.leaky_relu(self.bn2(self.conv2(h1)))
        h1 = self.dropout2(h1)

        # 调整维度以适应GRU输入 [batch_size, seq_len, input_size]
        h1 = h1.permute(0, 2, 1)

        # 通过GRU层
        output, _ = self.gru(h1)

        # 通过全连接层
        y_pred = self.fc(output)

        # 调整维度以适应后续CTCLoss计算 [seq_len, batch_size, num_classes]
        y_pred = y_pred.permute(1, 0, 2)

        return y_pred

epochs = 5000
num_blocks = 3
filters = 128
char2id=torch.load('cha2id.pth')
id2char=torch.load('id2char.pth')
new_char2id = {char: id + 1 for char, id in char2id.items()}
char2id = new_char2id

        # 更新 self.id2char 字典
new_id2char = {id + 1: char for id, char in id2char.items()}
id2char = new_id2char
id2char[0]='<blank>'
index=len(id2char)
id2char[index]='<eos>'
char2id['<blank>']=0
char2id['<eos>']=index
num_classes = len(char2id)
mfcc_dim = 13
model = ASRModel(mfcc_dim, num_blocks, filters, num_classes).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CTCLoss()

train_losses = []
valid_losses = []
batch_size = 32
train_path='./data_thchs30/train/*.trn'
dataset = ThchsData(train_path, sample_rate=16000)
# print(len((dataset)))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# print(len((dataloader)))
i2=0
for epoch in range(epochs):
    model.train()
    train_loss = 0

    model.train()
    for batch in dataloader:
        
        fea, labels, input_lengths, label_lengths = batch
        optimizer.zero_grad()
        Y_pred = model(fea.cuda())
        Y_pred = Y_pred.log_softmax(2)
        loss = criterion(Y_pred, labels.cuda(), input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        i2+=1
        train_loss += loss.item()
        print(f'steps {i2}, Train Loss: {loss:.4f}')
        # 去除空白标签（假设0为空白标签）
        Y_pred=Y_pred.permute(1, 0, 2)
        Y_pred=torch.argmax(Y_pred, dim=2).squeeze(0)
        # print(Y_pred)
        non_blank_sequence = [num for num in Y_pred[0]]
        
        # 去重合并
        final_sequence = []
        for i, num in enumerate(non_blank_sequence):
            if i == 0 or num != non_blank_sequence[i - 1]:
                final_sequence.append(num)

        # 将数字转换为字符
        final_text = ''.join([id2char[num.item()] for num in final_sequence])
        print(final_text)
        true_label=''.join([id2char[num.item()] for num in labels[0]])
        print(true_label)
    train_loss /= len(dataloader) // batch_size
    train_losses.append(train_loss)

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')
    
torch.save(model,'final.pth')