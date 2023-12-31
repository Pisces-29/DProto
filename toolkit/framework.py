import os
import sys
import torch
from torch import autograd, optim, nn
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        """
        sentence_encoder: Sentence encoder
        You need to set self.cost as your own loss function.
        """
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        """
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        """
        raise NotImplementedError

    def loss(self, logits, label):
        """
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        """
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        """
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        """
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader=None):
        """
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        """
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        """
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        """
        if os.path.isfile(ckpt):
            if not torch.cuda.is_available():
                checkpoint = torch.load(ckpt, map_location='cpu')
            else:
                checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        """
        PyTorch before and after 0.4
        """
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              dataset,
              B,
              N_for_train,
              K,
              Q,
              learning_rate=2e-5,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=1000,
              test_iter=10000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup_step=300,
              grad_iter=1,
              pair=False,
              use_sgd_for_bert=False):
        """
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N_for_train: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        na_rate: NOTA rate in training
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        """
        print("Start training...")
        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                        num_training_steps=train_iter)
        else:
            optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        model.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_right = 0.0
        iter_f1_macro = 0.0
        iter_sample = 0.0
        early_stopping_step = 0

        for it in range(start_iter, start_iter + train_iter):
            if pair:
                batch, query_label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    query_label = query_label.cuda()
            else:
                support, query, query_label, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    query_label = query_label.cuda()
                    label = label.cuda()
                label = label.view(B, N_for_train * K + Q)
            query_label = query_label.view(B, Q)

            if model_name == 'dproto':
                logits, cat_embedding, pred = model(support, query, N_for_train, K, Q)
                loss = model.loss(logits, cat_embedding, label, N_for_train, K, Q) / float(grad_iter)
            elif model_name == 'pair':
                logits, pred = model(batch, N_for_train, K, Q)
                loss = model.loss(logits, query_label) / float(grad_iter)
            elif model_name == 'mnav':
                logits, pred = model(support, query, N_for_train, K, Q)
                loss = model.loss(logits, query_label) / float(grad_iter)
            elif model_name == 'oproto':
                cos_similarity, pred = model(support, query, N_for_train, K, Q)
                loss = model.loss(cos_similarity, query_label, N_for_train, K, Q)
            else:
                raise NotImplementedError

            right = model.accuracy(pred, query_label)
            f1_macro = f1_score(y_true=query_label.view(-1).cpu().numpy(), y_pred=pred.cpu().view(-1).numpy(),
                                average='macro')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_f1_macro += f1_macro
            iter_sample += 1

            sys.stdout.write(
                'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, macro_f1: {3:3.2f}%'.format(
                    it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample,
                    100 * iter_f1_macro / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, model_name, dataset, B, N_for_train, K, Q, val_iter, pair=pair)
                model.train()
                if acc >= best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                    early_stopping_step = 0
                else:
                    early_stopping_step += 1
                    if early_stopping_step >= 6:
                        print("early stopping...")
                        break

                iter_loss = 0.0
                iter_right = 0.0
                iter_f1_macro = 0.0
                iter_sample = 0.0

        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
             model,
             model_name,
             dataset,
             B, N, K, Q,
             eval_iter,
             pair=False,
             ckpt=None,
             test_data_loader=None):
        print("")
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = test_data_loader

        iter_right = 0.0
        iter_f1_macro = 0.0
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                if pair:
                    batch, query_label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in batch:
                            batch[k] = batch[k].cuda()
                        query_label = query_label.cuda()
                else:
                    support, query, query_label, label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        query_label = query_label.cuda()

                query_label = query_label.view(B, Q)

                if model_name == 'dproto':
                    logits, cat_embedding, pred = model(support, query, N, K, Q)
                elif model_name == 'pair':
                    logits, pred = model(batch, N, K, Q)
                elif model_name == 'mnav':
                    logits, pred = model(support, query, N, K, Q)
                elif model_name == 'oproto':
                    cos_similarity, pred = model(support, query, N, K, Q)
                else:
                    raise NotImplementedError

                right = model.accuracy(pred, query_label)
                f1_macro = f1_score(y_true=query_label.view(-1).cpu().numpy(), y_pred=pred.cpu().view(-1).numpy(),
                                    average='macro')

                iter_right += self.item(right.data)
                iter_f1_macro += f1_macro
                iter_sample += 1

                sys.stdout.write(
                    '[EVAL] step: {0:4} | accuracy: {1:3.2f}%, macro_f1: {2:3.2f}%'.format(
                        it + 1, 100 * iter_right / iter_sample, 100 * iter_f1_macro / iter_sample) + '\r')
                sys.stdout.flush()
            print("")
        return iter_right / iter_sample
