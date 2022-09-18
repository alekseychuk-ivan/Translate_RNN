import torch
import random
from module import *
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--filepath', '-fp', help='Path to file', default="./input", type=str)
parse.add_argument('--file', '-f', help='File name', default="afr.txt", type=str)
parse.add_argument('--hidden', '-hi', help='Number of hidden layer', default=256, type=int)
parse.add_argument('--reverse', '-r', help='Number of hidden layer', default=False, type=bool)
parse.add_argument('--iters', '-i', help='Number of iter', default=70000, type=int)

args = parse.parse_args()

args = parse.parse_args()
FILE_NAME = args.file
FILE_PATH = args.filepath
hidden_size = args.hidden
reverse = args.reverse
iters = args.iters

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
input_lang, output_lang, pairs = prepare_data(file_path=FILE_PATH, file_name=FILE_NAME, reverse=reverse)


encoder = EncoderRNN(input_lang.n_words, hidden_size, device=device).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, device=device).to(device)
encoder, decoder = trainIters(encoder, decoder,  n_iters=iters, print_every=5000, pairs=pairs, input_lang=input_lang,
                              output_lang=output_lang, device=device)

evaluateRandomly(encoder, decoder, pairs, n=5, input_lang=input_lang, output_lang=output_lang, device=device)
