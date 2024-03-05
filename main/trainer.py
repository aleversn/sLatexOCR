import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from main.dataloader.pix2tex_loader import Im2LatexDataset

from main.models import get_model
from main.analysis import Analysis
from main.utils import *
from Levenshtein import distance
from torchtext.data import metrics


class Trainer():

    def __init__(self, model_config, encoder_structure='hybrid', batchsize=10, testbatchsize=20, pad=False, keep_smaller_batches=True, max_seq_len=512, vocab_file=None, train_data_path=[], eval_data_path=[], resume_path=None, task_name='Sim'):
        self.model_config = model_config
        self.encoder_structure = encoder_structure
        self.batchsize = batchsize
        self.testbatchsize = testbatchsize
        self.pad = pad
        self.keep_smaller_batches = keep_smaller_batches
        self.max_seq_len = max_seq_len
        self.vocab_file = vocab_file
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.task_name = task_name
        self.analysis = Analysis()
        self.dataloader_init()
        self.model_init(resume_path=resume_path)

    def model_init(self, resume_path=None):
        # print('AutoModel Choose Model: {}\n'.format(self.model_from_pretrained))
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_config.device = self.device
        self.model = get_model(self.encoder_structure,
                               self.device, self.model_config)
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            self.model.load_state_dict(torch.load(
                resume_path, map_location=self.device))
        self.model.to(self.device)

    def dataloader_init(self):
        assert len(self.train_data_path) > 0, 'Train Data Path is Empty'
        assert len(self.eval_data_path) > 0, 'Eval Data Path is Empty'
        sorted(self.train_data_path)
        sorted(self.eval_data_path)
        train_cache_path = os.path.join(
            './tmp/train', self.get_hash(''.join(self.train_data_path)))
        eval_cache_path = os.path.join(
            './tmp/eval', self.get_hash(''.join(self.eval_data_path)))
        self.train_loader = None
        self.eval_loader = None
        if not os.path.exists(train_cache_path):
            os.makedirs(train_cache_path, exist_ok=True)
            for path in self.train_data_path:
                if self.train_loader is None:
                    self.train_loader = Im2LatexDataset(tokenizer=self.vocab_file, images=os.path.join(
                        path, 'train'), equations=os.path.join(path, 'labels.txt'))
                else:
                    self.train_loader.combine(Im2LatexDataset(tokenizer=self.vocab_file, images=os.path.join(
                        path, 'train'), equations=os.path.join(path, 'labels.txt')))
            self.train_loader.update(batchsize=1, keep_smaller_batches=True)
            self.train_loader.save(os.path.join(
                train_cache_path, 'dataset.pkl'))
        else:
            self.train_loader = Im2LatexDataset().load(os.path.join(
                train_cache_path, 'dataset.pkl'))
        args = {
            'batchsize': self.batchsize,
            'shuffle': True,
            'pad': self.pad,
            'keep_smaller_batches': self.keep_smaller_batches,
            'max_seq_len': self.max_seq_len,
            'max_dimensions': [self.model_config.max_width, self.model_config.max_height],
            'min_dimensions': [self.model_config.min_width, self.model_config.min_height]
        }
        self.train_loader.update(**args, test=False)

        if not os.path.exists(eval_cache_path):
            os.makedirs(eval_cache_path, exist_ok=True)
            for path in self.eval_data_path:
                if self.eval_loader is None:
                    self.eval_loader = Im2LatexDataset(tokenizer=self.vocab_file, images=os.path.join(
                        path, 'val'), equations=os.path.join(path, 'labels.txt'))
                else:
                    self.eval_loader.combine(Im2LatexDataset(tokenizer=self.vocab_file, images=os.path.join(
                        path, 'val'), equations=os.path.join(path, 'labels.txt')))
            self.eval_loader.update(batchsize=1, keep_smaller_batches=True)
            self.eval_loader.save(os.path.join(eval_cache_path, 'dataset.pkl'))
        else:
            self.eval_loader = Im2LatexDataset().load(
                os.path.join(eval_cache_path, 'dataset.pkl'))
        valargs = args.copy()
        valargs.update(batchsize=self.testbatchsize,
                       keep_smaller_batches=True, test=True)
        self.eval_loader.update(**valargs)
        self.tokenizer = self.eval_loader.tokenizer

    def get_hash(self, text):
        hash_object = hashlib.sha256()
        hash_object.update(text.encode('utf-8'))
        hashed_text = hash_object.hexdigest()
        return hashed_text

    def __call__(self, resume_step=None, num_epochs=30, lr=1e-4, gpu=[0], eval_call_epoch=None):
        return self.train(resume_step=resume_step,
                          num_epochs=num_epochs, lr=lr, gpu=gpu, eval_call_epoch=eval_call_epoch)

    def train(self, resume_step=None, num_epochs=30, lr=1e-4, gpu=[0], eval_call_epoch=None):
        if torch.cuda.is_available():
            self.model_config.gpu_devices = gpu
            gpu_memory_check(self.model, self.model_config)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.)
        scheduler = get_linear_schedule_with_warmup(optimizer, 190, 80000)

        current_uid = str(uuid.uuid1()).split('-')[0]

        train_step = resume_step if resume_step is not None else 0
        for epoch in range(num_epochs):
            train_count = 0
            train_loss = 0

            train_iter = tqdm(self.train_loader)
            self.model.train()

            for (seq, im) in train_iter:

                tgt_seq, tgt_mask = seq['input_ids'].to(
                    self.device), seq['attention_mask'].bool().to(self.device)
                im = im.to(self.device)
                loss = self.model.data_parallel(
                    im, device_ids=gpu, tgt_seq=tgt_seq, mask=tgt_mask)
                loss.backward()  # data parallism loss is a vector
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                train_loss += loss.data.item()
                train_count += 1
                train_step += 1

                train_iter.set_description(
                    'Train: {}/{}'.format(epoch + 1, num_epochs))
                train_iter.set_postfix(
                    train_loss=train_loss / train_count)

            self.analysis.append_train_record({
                'epoch': epoch + 1,
                'train_loss': train_loss / train_count,
            })

            model_uid = self.save_model(train_step)
            if eval_call_epoch is None or eval_call_epoch(epoch):
                self.eval(epoch)

            self.analysis.save_all_records(
                uid=current_uid if self.task_name is None else self.task_name)
            yield (epoch, self.analysis.train_record, self.analysis.eval_record, self.analysis.model_record, model_uid)

    def save_model(self, current_step=0):
        if self.task_name is None:
            dir = 'undefined'
        else:
            dir = self.task_name
        save_path = f'./save_model/{dir}/Checkpoint_{current_step}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(),
                   os.path.join(save_path, 'model.pth'))
        self.analysis.append_model_record(current_step)
        return current_step

    def detokenize(self, tokens, tokenizer):
        toks = [tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in (['[BOS]', '[EOS]', '[PAD]']):
                    del toks[b][i]
        return toks

    def eval(self, epoch, temperature=0.2, gpu=[0]):
        with torch.no_grad():
            eval_count = 0

            eval_iter = tqdm(self.eval_loader)
            self.model.eval()

            bleus, edit_dists, token_acc = [], [], []
            bleu_score, edit_distance, token_accuracy = 0, 1, 0
            for (seq, im) in eval_iter:
                if seq is None or im is None:
                    continue
                # loss = decoder(tgt_seq, mask=tgt_mask, context=encoded)
                dec = self.model.generate(
                    im.to(self.device), temperature=temperature)
                pred = self.detokenize(dec, self.tokenizer)
                truth = self.detokenize(seq['input_ids'], self.tokenizer)
                bleus.append(metrics.bleu_score(
                    pred, [alternatives(x) for x in truth]))
                for predi, truthi in zip(token2str(dec, self.tokenizer), token2str(seq['input_ids'], self.tokenizer)):
                    ts = post_process(truthi)
                    if len(ts) > 0:
                        edit_dists.append(
                            distance(post_process(predi), ts)/len(ts))
                dec = dec.cpu()
                tgt_seq = seq['input_ids'][:, 1:]
                shape_diff = dec.shape[1]-tgt_seq.shape[1]
                if shape_diff < 0:
                    dec = torch.nn.functional.pad(
                        dec, (0, -shape_diff), "constant", self.pad)
                elif shape_diff > 0:
                    tgt_seq = torch.nn.functional.pad(
                        tgt_seq, (0, shape_diff), "constant", self.pad)
                mask = torch.logical_or(tgt_seq != self.pad, dec != self.pad)
                tok_acc = (dec == tgt_seq)[mask].float().mean().item()
                token_acc.append(tok_acc)
                eval_iter.set_description('BLEU: %.3f, ED: %.2e, ACC: %.3f' % (
                    np.mean(bleus), np.mean(edit_dists), np.mean(token_acc)))
            bleu_score = np.mean(bleus) if len(bleus) > 0 else 0

            edit_distance = np.mean(edit_dists) if len(edit_dists) > 0 else 0

            token_accuracy = np.mean(token_acc) if len(token_acc) > 0 else 0

            print('\n%s\n%s' % (truth, pred))
            print('BLEU: %.2f' % bleu_score)

            self.analysis.append_eval_record({
                'epoch': epoch + 1,
                'bleu_score': bleu_score,
                'edit_distance': edit_distance,
                'token_accuracy': token_accuracy
            })

    def cuda(self, inputX):
        if type(inputX) == tuple:
            if torch.cuda.is_available():
                result = []
                for item in inputX:
                    result.append(item.cuda())
                return result
            return inputX
        else:
            if torch.cuda.is_available():
                return inputX.cuda()
            return inputX
