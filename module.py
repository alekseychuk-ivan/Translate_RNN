import unicodedata
import re
import torch.nn as nn
import torch
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import torch.optim as optim
import torch.nn.functional as F
plt.switch_backend('agg')


SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5


class LanguageVocabulary(object):
    def __init__(self, name):
        # название языка
        self.name = name
        # словарик word2index который хранит соответственно кодировку слова в целочисленный индекс словаря
        self.word2index = {}
        # обычный словарик который хранит распределение слов, сколько слов мы использовали и сколько обнаружили
        self.word2count = {}
        # Обратный словарик словарю word2index где хранятся уже индексы и замаппенные слова к каждому индексу, нужен будет для расшифровки последовательности
        self.index2word = {0: "SOS", 1: "EOS"}
        # Count SOS and EOS, храним просто общее количество слов в нашем словаре, то есть количество токенов в сформированном словарике нашего языка
        self.n_words = 2

    def add_sentence(self, sentence):
        """
        Метод класса, для добавления предложения в словарь.
        Каждое предложение поступающее к нам, будет разбираться на
        примитивные токены и добавляться в словарь при помощи метода класса addword()
        """
        for word in sentence.split(' '):
            self.add_word(word)


    def add_word(self, word):
        # проверяем не входит ли наше слово в словарь word2index
        if word not in self.word2index:
            # добавляем в качестве ключа слово а в качестве значения последнее n_words
            self.word2index[word] = self.n_words
            # меняем на единичку
            self.word2count[word] = 1
            # и соответственно меняем и index2word словарик добавляя уже слово для декодирования
            self.index2word[self.n_words] = word
            # инкрементируем n_words
            self.n_words += 1
        else:
            # Если такое уже слово есть просто добавляем 1 что добавилось одно слово
            self.word2count[word] += 1


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s, lang='eng'):
    # Декодируем из юникода в ascii
    s = unicode_to_ascii(s.lower().strip())
    # Что означает данное регулярное выражение - точку, !, ? меняем на пробел чтобы этот символ стоял отдельно от всех
    # https://docs.python.org/3/library/re.html - стандартная (родная) библиотка Python которая нужна для работы с регулярными выражениями
    s = re.sub(r"([.!?])", r" \1", s)
    # оставляем только наборы символов указанных в паттерне регулярного выражения остальное заменим на пробел
    if lang in ['eng', 'afr', 'ita', 'nld', ]:
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    elif lang in ['rus', ]:
        s = re.sub(r"[^а-яА-ЯёЁ.!?]+", r" ", s)
    elif lang in ['ukr', 'bel']:
        s = re.sub(r"[^а-яА-ЯёЁієїґiI\'.!?]+", r" ", s)
    elif lang in ['cmn']:
        s = re.sub(r'[^\u4e00-\u9fff.!?]+', r' ', s)
    elif lang in ['pol']:
        s = re.sub(r'[^A-Za-zżźćńółęąśŻŹĆĄŚĘŁÓŃ.!?]+', r' ', s)
    elif lang in ['deu']:
        s = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß]+', r' ', s)
    return s


def read_file(file_path, file_name, reverse=False):
    print("Reading lines...")
    # Берем документ корпуса, лежащий в директории ./data/___.txt подставляя значения указанных языков в нашем случае eng-fra, он читается бьется на предложения
    lines = open(f'{file_path}/{file_name}', encoding='utf-8').read().strip().split('\n')
    lang1, lang2 = 'eng', file_name[:file_name.find('.txt')]
    lang = [lang1, lang2]
    # Разбиваем построчно и нормализуем строку:
    # pairs = list()
    # for l in lines:
    #     pairs.append([normalize_string(l.split('\t')[i]) for i in range(2)])
    pairs = [[normalize_string(l.split('\t')[i], lang[i]) for i in range(2)] for l in lines]
    # Можем создавать и проходить как с целевого языка на исходный так и наоборот:
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = LanguageVocabulary(lang2)
        output_lang = LanguageVocabulary(lang1)
    else:
        input_lang = LanguageVocabulary(lang1)
        output_lang = LanguageVocabulary(lang2)
    return input_lang, output_lang, pairs


def prepare_data(file_path, file_name, reverse=False):
    input_lang, output_lang, pairs = read_file(file_path, file_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    # pairs = filter_pairs(pairs)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device=torch.device('cpu')):
        super(EncoderRNN, self).__init__()
        # Как помним hidden_size - размер скрытого состояния
        self.hidden_size = hidden_size
        # Слой Эмбеддингов, который из входного вектора последовательности (либо батча) отдаст представление последовательности для скрытого состояния
        # FYI: в качестве Input_size у нас размер словаря
        self.embedding = nn.Embedding(input_size, hidden_size)
        # И соответственно рекуррентная ячейка GRU которая принимает MxM (hidden на hidden)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        # Приводим эмбеддинг к формату одного предлоежния 1х1 и любая размерность
        embedded = self.embedding(input).view(1, 1, -1)
        # Нужно для следующего шага пока не запутываемся :) просто присвоили наш эмбеддинг
        output = embedded
        # и соответственно подаем все в ГРЮ ячейку (эмбеддинг и скрытые состояния)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # Дополнительно сделаем инициализацию скрытого представления (просто заполним нулями)
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class DecoderRNN(nn.Module):
    # Будьте внимательны, теперь на вход мы получаем размер скрытого представления
    def __init__(self, hidden_size, output_size, device=torch.device('cpu')):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # Слой эмбеддингов - рамер словаря, размер скрытого представления
        self.embedding = nn.Embedding(output_size, hidden_size)
        # GRU скрытое состояние на скрытое
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Переводим hidden size в распределение для этого передаем в линейный слов скрытое состояни и размер словаря
        self.out = nn.Linear(hidden_size, output_size)
        # Получаем распределение верояностей
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0])) # берем output по нулевому индексу (одно токену)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


# Токены кодируем в целочисленное представление
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


# Берем предложение с указанным языком, делаем из него индексы и вставляем метку конца предложения, превращаем в тензор:
def tensorFromSentence(lang, sentence, device=torch.device('cpu')):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


# Для создания тензора из пар:
def tensorsFromPair(input_lang, output_lang, pair, device=torch.device('cpu')):
    input_tensor = tensorFromSentence(input_lang, pair[0], device=device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device=device)
    return (input_tensor, target_tensor)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- eta: %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, pairs, input_lang, output_lang, print_every=1000, plot_every=100,
               learning_rate=0.01, device=torch.device('cpu')):
    print(f'Model trains. Please wait...')
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # Делаем выборку наших пар функцией которую создали до
    training_pairs = [tensorsFromPair(input_lang=input_lang, output_lang=output_lang, pair=random.choice(pairs),
                                      device=device) for i in range(n_iters)]
    # FYI! Используем Negative Log-Likelihood Loss потому что log softmax уже присутствует в модели
    criterion = nn.NLLLoss()

    for epoch in range(1, n_iters + 1):
        training_pair = training_pairs[epoch - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # Используем функцию для тренировки на отдельных токенах, которую написали выше
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, device=device)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_iters),
                                         epoch, epoch / n_iters * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    # showPlot(plot_losses)
    return encoder, decoder


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(encoder, decoder, sentence, input_lang, output_lang, device, max_length=50,):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = [] # Наши деокдированные слова

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words


def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device, n=10):
    acc = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang=input_lang, output_lang=output_lang,
                                device=device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    for pair in pairs:
        output_words = evaluate(encoder, decoder, pair[0], input_lang=input_lang, output_lang=output_lang,
                                device=device)
        if pair[1] == ' '.join(output_words[:-1]):
            acc += 1
    print(f'{acc / len(pairs):.3f}')


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=50, device=torch.device('cpu')):
    # Просто инициализируем скрытое представление для энкодера
    encoder_hidden = encoder.initHidden()
    # Скиыдваем градиенты для алгоритма градиентного спуска как и у энкодера так и у дэкодера
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Получаем размер в словаря (токенов) для входящего и выходящего тензора так как мы пробегаемся по каждому предложению по кусочкам
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # Создаем переменную где будем хранить наши выходы из энкодера (в данной реализации пока не юзаем, далее будет еще один вариант)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    loss = 0
    # пробегаем по длине входящего тензора и в экодер передаем последовательно каждый из токенов:
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # Сохраняем все выходы из энкодера для одного слова
        encoder_outputs[ei] = encoder_output[0, 0]


    # Закончили с энкодером пошли к декодеру, как было сказано декодер начинается с SOS
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # FYI здесь мы скрытое представление из энкодера передаем в скрытое представление в декодер, то есть после знака =
    # у нас будут ходить градиенты из декодера в энкодер, то есть когда мы будем считать градиенты, они сначала пробегут по декодеру
    # дойдут до знака = перескочат в энкодер и будут дальше считаться по энкодеру и эти градиенты сохранятся в соответствующих тензорах
    # и когда будут отрабатывать разные оптимайзеры (у нас их 2) у них будут соответствующие правильные градиенты которые смогут правильно отработать
    decoder_hidden = encoder_hidden

    # Будем использовать Teacher Forcing в части случае (подставляя правильную последовательность)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Подаем decoder_input = torch.tensor([[SOS_token]], device=device) то есть по одному слову и скрытое представление
        for di in range(target_length):
            # Переведенное предложение и скрытое представление
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # Считаем ошибку
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length