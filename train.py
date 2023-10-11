import torch
from torch import optim
import numpy as np
import random
import argparse
import os
from dataset.FewRelFS import get_loader_fewrel_fs
from dataset.FewRelPairFS import get_loader_fewrel_pair_fs
from dataset.GenerateNOTAVectors import generate_NOTA_vector
from encoder.BertSentenceEncoder import BERTSentenceEncoder
from encoder.BertPairSentenceEncoder import BERTPAIRSentenceEncoder
from model.Pair import Pair
from model.DProto import DProto
from model.OProto import OProto
from model.MNAV import MNAV
from toolkit.framework import FewShotREFramework


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='fewrel',
                        help='Training and testing dataset.')
    parser.add_argument('--N', default=5, type=int,
                        help='N way (training)')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--Q', default=3, type=int,
                        help='Q represents the number of instances in the query set.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size in training and testing.')
    parser.add_argument('--seed', default=5, type=int,
                        help='seed')
    parser.add_argument('--train_iter', default=30000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='dproto',
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='sentence encoder')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--lr', default=-1, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--trainNA', default=0.5, type=float,
                        help='NA rate in training')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='sgd',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=230, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--only_test', action='store_true',
                        help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')

    # only for Density-Proto
    parser.add_argument('--gamma', default=1e-5, type=float,
                        help='parameter gamma')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='threshold.')

    # only for MNAV
    parser.add_argument('--vector_num', default=20, type=int,
                        help='the number of NOTA vectors')

    # experiment
    parser.add_argument('--mask_entity', action='store_true',
                        help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true',
                        help='use SGD instead of AdamW for BERT.')

    opt = parser.parse_args()
    N = opt.N
    K = opt.K
    Q = opt.Q
    trainNA = opt.trainNA
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    seed = opt.seed

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("dataset: {}".format(opt.dataset))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    # select a dataset
    if opt.dataset == "fewrel":
        trainName = "train_wiki"
        valName = "val_wiki"
        testName = "test_wiki"
    else:
        raise NotImplementedError

    # set seed
    set_seed(seed)

    # select a sentence encoder
    if encoder_name == "bert":
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.pair:
            sentence_encoder = BERTPAIRSentenceEncoder(
                pretrain_ckpt,
                max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(
                pretrain_ckpt,
                max_length,
                cat_entity_rep=opt.cat_entity_rep,
                mask_entity=opt.mask_entity)
    else:
        raise NotImplementedError

    if model_name == "pair":
        if opt.dataset == "fewrel":
            train_data_loader = get_loader_fewrel_pair_fs(trainName, sentence_encoder, N=N, K=K, Q=Q,
                                                          na_rate=trainNA, batch_size=batch_size)
            val_data_loader = get_loader_fewrel_pair_fs(valName, sentence_encoder, N=N, K=K, Q=Q, na_rate=trainNA,
                                                        batch_size=batch_size)
            test15_data_loader = get_loader_fewrel_pair_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.15,
                                                           batch_size=batch_size)
            test30_data_loader = get_loader_fewrel_pair_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.3,
                                                           batch_size=batch_size)
            test50_data_loader = get_loader_fewrel_pair_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.5,
                                                           batch_size=batch_size)
        else:
            raise NotImplementedError
    else:
        if opt.dataset == "fewrel":
            train_data_loader = get_loader_fewrel_fs(trainName, sentence_encoder, N=N, K=K, Q=Q, na_rate=trainNA,
                                                     batch_size=batch_size, root='./data/fewrel')
            val_data_loader = get_loader_fewrel_fs(valName, sentence_encoder, N=N, K=K, Q=Q, na_rate=trainNA,
                                                   batch_size=batch_size, root='./data/fewrel')
            test15_data_loader = get_loader_fewrel_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.15,
                                                      batch_size=batch_size, root='./data/fewrel')
            test30_data_loader = get_loader_fewrel_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.3,
                                                      batch_size=batch_size, root='./data/fewrel')
            test50_data_loader = get_loader_fewrel_fs(testName, sentence_encoder, N=N, K=K, Q=Q, na_rate=0.5,
                                                      batch_size=batch_size, root='./data/fewrel')
        else:
            raise NotImplementedError

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    framework = FewShotREFramework(train_data_loader, val_data_loader)
    prefix = '-'.join([model_name, encoder_name, trainName, valName, testName, str(N), str(K), str(seed)])
    if trainNA != 0:
        prefix += '-na{}'.format(trainNA)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if opt.threshold:
        prefix += '-' + str(opt.threshold)
    if opt.gamma:
        prefix += '-' + str(opt.gamma)
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    # model
    model = None
    if model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
    elif model_name == "dproto":
        model = DProto(sentence_encoder=sentence_encoder, threshold=opt.threshold, gamma=opt.gamma)
    elif model_name == "oproto":
        model = OProto(sentence_encoder=sentence_encoder, dot=opt.dot)
    elif model_name == "mnav":
        NOTA_vectors = None
        if opt.dataset == "fewrel":
            NOTA_vectors = generate_NOTA_vector(encoder=sentence_encoder, root="./data/fewrel", name=trainName,
                                                num=opt.vector_num)
        model = MNAV(sentence_encoder=sentence_encoder, NOTA_vectors=NOTA_vectors, dot=opt.dot)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model.cuda()

    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if not opt.only_test:
        if encoder_name in ['bert']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, model_name, opt.dataset, batch_size, N, K, Q, pytorch_optim=pytorch_optim,
                        load_ckpt=opt.load_ckpt, save_ckpt=ckpt, val_step=opt.val_step, pair=opt.pair,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim,
                        learning_rate=opt.lr,
                        use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    if opt.dataset == "fewrel":
        acc15 = framework.eval(model, model_name, opt.dataset, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt,
                               pair=opt.pair, test_data_loader=test15_data_loader)
        print("")
        acc30 = framework.eval(model, model_name, opt.dataset, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt,
                               pair=opt.pair, test_data_loader=test30_data_loader)
        print("")
        acc50 = framework.eval(model, model_name, opt.dataset, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt,
                               pair=opt.pair, test_data_loader=test50_data_loader)
        print("0.15 NOTA rate, test RESULT: %.2f" % (acc15 * 100))
        print("0.30 NOTA rate, test RESULT: %.2f" % (acc30 * 100))
        print("0.50 NOTA rate, test RESULT: %.2f" % (acc50 * 100))
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
