"""
Implementation of Bi-directional LSTM-CNNs-CRF model for NER.
"""

import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse

import numpy as np
import torch
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from neuronlp2.io import get_logger, conll03_data, CoNLL03Writer, iterate_data
from neuronlp2.models import BiRecurrentConv, BiVarRecurrentConv, BiRecurrentConvCRF, BiVarRecurrentConvCRF
from neuronlp2.optim import ExponentialScheduler
from neuronlp2 import utils


def evaluate(output_file, scorefile):
    script = os.path.join(current_path, 'eval/conll03eval.v2')
    os.system("perl %s < %s > %s" % (script, output_file, scorefile))
    with open(scorefile, 'r') as fin:
        fin.readline()
        line = fin.readline()
        fields = line.split(";")
        print("fileds", fields)
        acc = float(fields[0].split(":")[1].strip()[:-1])
        precision = float(fields[1].split(":")[1].strip()[:-1])
        recall = float(fields[2].split(":")[1].strip()[:-1])
        f1 = float(fields[3].split(":")[1].strip())
    return acc, precision, recall, f1


def get_optimizer(parameters, optim, learning_rate, lr_decay, amsgrad, weight_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=amsgrad, weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler


def eval(data, network, writer, outfile, scorefile, device):
    network.eval()
    writer.start(outfile)
    for data in iterate_data(data, 256):
        words = data['WORD'].to(device)
        chars = data['CHAR'].to(device)
        labels = data['NER'].numpy()
        masks = data['MASK'].to(device)
        postags = data['POS'].numpy()
        chunks = data['CHUNK'].numpy()
        lengths = data['LENGTH'].numpy()
        preds = network.decode(words, chars, mask=masks, leading_symbolic=conll03_data.NUM_SYMBOLIC_TAGS)
        writer.write(words.cpu().numpy(), postags, chunks, preds.cpu().numpy(), labels, lengths)
    writer.close()
    acc, precision, recall, f1 = evaluate(outfile, scorefile)
    return acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='NER with bi-directional RNN-CNN')
    parser.add_argument('--config', type=str, help='config file', required=True)
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence', help='loss type (default: sentence)')
    parser.add_argument('--optim', choices=['sgd', 'adam'], help='type of optimizer', required=True)
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    parser.add_argument('--grad_clip', type=float, default=0, help='max norm for gradient clip (default 0: no clip')
    parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train', help='path for training file.', required=True)
    parser.add_argument('--dev', help='path for dev file.', required=True)
    parser.add_argument('--test', help='path for test file.', required=True)
    parser.add_argument('--model_path', help='path for saving model file.', required=True)

    args = parser.parse_args()

    logger = get_logger("NER")

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    optim = args.optim
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    amsgrad = args.amsgrad
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip

    loss_ty_token = args.loss_type == 'token'
    unk_replace = args.unk_replace

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    embedding = args.embedding
    embedding_path = args.embedding_dict

    print(args)

    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet = conll03_data.create_alphabets(alphabet_path, train_path,
                                                                                                             data_paths=[dev_path, test_path],
                                                                                                             embedd_dict=embedd_dict, max_vocabulary_size=50000)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Chunk Alphabet Size: %d" % chunk_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())

    logger.info("Reading Data")

    data_train = conll03_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    num_data = sum(data_train[1])
    num_labels = ner_alphabet.size()

    data_dev = conll03_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)
    data_test = conll03_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

    writer = CoNLL03Writer(word_alphabet, char_alphabet, pos_alphabet, chunk_alphabet, ner_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conll03_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()

    logger.info("constructing network...")

    hyps = json.load(open(args.config, 'r'))
    json.dump(hyps, open(os.path.join(model_path, 'config.json'), 'w'), indent=2)
    dropout = hyps['dropout']
    crf = hyps['crf']
    bigram = hyps['bigram']
    print("embed dim", embedd_dim)
    print("embed dim", hyps['embedd_dim'])
    assert embedd_dim == hyps['embedd_dim']
    char_dim = hyps['char_dim']
    mode = hyps['rnn_mode']
    hidden_size = hyps['hidden_size']
    out_features = hyps['out_features']
    num_layers = hyps['num_layers']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    p_rnn = hyps['p_rnn']
    activation = hyps['activation']

    if dropout == 'std':
        if crf:
            network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), mode, hidden_size, out_features, num_layers,
                                         num_labels, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, activation=activation)
        else:
            network = BiRecurrentConv(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), mode, hidden_size, out_features, num_layers,
                                      num_labels, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, activation=activation)
    elif dropout == 'variational':
        if crf:
            network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), mode, hidden_size, out_features, num_layers,
                                            num_labels, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, bigram=bigram, activation=activation)
        else:
            network = BiVarRecurrentConv(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), mode, hidden_size, out_features, num_layers,
                                         num_labels, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, activation=activation)
    else:
        raise ValueError('Unkown dropout type: {}'.format(dropout))

    network = network.to(device)

    optimizer, scheduler = get_optimizer(network.parameters(), optim, learning_rate, lr_decay, amsgrad, weight_decay, warmup_steps)
    model = "{}-CNN{}".format(mode, "-CRF" if crf else "")
    logger.info("Network: %s, num_layer=%d, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (weight_decay, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): %s(%.2f, %.2f, %s)" % (dropout, p_in, p_out, p_rnn))
    print('# of Parameters: %d' % (sum([param.numel() for param in network.parameters()])))

    best_f1 = 0.0
    best_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    test_f1 = 0.0
    test_acc = 0.0
    test_precision = 0.0
    test_recall = 0.0
    best_epoch = 0
    patient = 0
    num_batches = num_data // batch_size + 1
    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = 0.
        num_insts = 0
        num_words = 0
        num_back = 0
        network.train()
        lr = 0.01 #scheduler.get_lr()[0]
        print('Epoch %d (%s, lr=%.6f, lr decay=%.6f, amsgrad=%s, l2=%.1e): ' % (epoch, optim, lr, lr_decay, amsgrad, weight_decay))
        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()
        for step, data in enumerate(iterate_data(data_train, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
            optimizer.zero_grad()
            words = data['WORD'].to(device)
            chars = data['CHAR'].to(device)
            labels = data['NER'].to(device)
            masks = data['MASK'].to(device)

            nbatch = words.size(0)
            nwords = masks.sum().item()

            loss_total = network.loss(words, chars, labels, mask=masks).sum()
            if loss_ty_token:
                loss = loss_total.div(nwords)
            else:
                loss = loss_total.div(nbatch)
            loss.backward()
            if grad_clip > 0:
                clip_grad_norm_(network.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                num_insts += nbatch
                num_words += nwords
                train_loss += loss_total.item()

            # update log
            if step % 100 == 0:
                torch.cuda.empty_cache()
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                curr_lr = scheduler.get_lr()[0]
                log_info = '[%d/%d (%.0f%%) lr=%.6f] loss: %.4f (%.4f)' % (step, num_batches, 100. * step / num_batches,
                                                                           curr_lr, train_loss / num_insts, train_loss / num_words)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('total: %d (%d), loss: %.4f (%.4f), time: %.2fs' % (num_insts, num_words, train_loss / num_insts,
                                                                  train_loss / num_words, time.time() - start_time))
        print('-' * 100)

        # evaluate performance on dev data
        with torch.no_grad():
            outfile = os.path.join(result_path, 'pred_dev%d' % epoch)
            scorefile = os.path.join(result_path, "score_dev%d" % epoch)
            acc, precision, recall, f1 = eval(data_dev, network, writer, outfile, scorefile, device)
            print('Dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision, recall, f1))
            if best_f1 < f1:
                torch.save(network.state_dict(), model_name)
                best_f1 = f1
                best_acc = acc
                best_precision = precision
                best_recall = recall
                best_epoch = epoch

                # evaluate on test data when better performance detected
                outfile = os.path.join(result_path, 'pred_test%d' % epoch)
                scorefile = os.path.join(result_path, "score_test%d" % epoch)
                test_acc, test_precision, test_recall, test_f1 = eval(data_test, network, writer, outfile, scorefile, device)
                print('test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (test_acc, test_precision, test_recall, test_f1))
                patient = 0
            else:
                patient += 1
            print('-' * 100)

            print("Best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d (%d))" % (best_acc, best_precision, best_recall, best_f1, best_epoch, patient))
            print("Best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d (%d))" % (test_acc, test_precision, test_recall, test_f1, best_epoch, patient))
            print('=' * 100)

        if patient > 4:
            logger.info('reset optimizer momentums')
            scheduler.reset_state()
            patient = 0


if __name__ == '__main__':
    main()
