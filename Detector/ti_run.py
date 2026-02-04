from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import recall_score,precision_score,f1_score, roc_auc_score,confusion_matrix, classification_report,roc_curve, auc,accuracy_score
import csv
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
import matplotlib.pyplot as plt
import seaborn as sns
from ti_model import Model
from scipy.special import softmax

logger = logging.getLogger(__name__)
cpu_cont = 16

def get_example(item):
    comment,code,label,tokenizer,args = item
    code_tokens = tokenizer.tokenize(code)
    ls_tokens = tokenizer.tokenize(comment)
    return convert_examples_to_features(ls_tokens,code_tokens, label, tokenizer, args, {})


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        
        
def convert_examples_to_features(code1_tokens,code2_tokens,label,tokenizer,args,cache):
    """convert examples to token ids"""
    code1_tokens = code1_tokens[:args.ls_size-4]
    code1_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code1_tokens+[tokenizer.sep_token]
    code2_tokens = code2_tokens[:args.block_size-4]
    code2_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code2_tokens+[tokenizer.sep_token]  
    
    code1_ids = tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.ls_size - len(code1_ids)
    code1_ids += [tokenizer.pad_token_id]*padding_length
    
    code2_ids = tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids += [tokenizer.pad_token_id]*padding_length
    
    source_tokens = code1_tokens+code2_tokens
    source_ids = code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, pool=None):
        postfix = file_path.split('/')[-1].split('.jsonl')[0]
        self.examples = []
        logger.info("Creating features from json  file at %s ", file_path)

        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            for json_obj in json_data:
                comment = json_obj['old_comment_raw']

                code = json_obj["new_code_raw"]
                if json_obj['label'] == 0:
                    label = 0
                elif json_obj['label'] == 1:
                    label = 1
                data.append((comment,code,label, tokenizer, args))


        self.examples = pool.map(get_example,tqdm(data,total=len(data)))
        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer,pool):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    
    args.max_steps = args.num_train_epochs * len( train_dataloader)
    args.save_steps = args.max_steps // args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu )
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    losses, best_f1 = [], 0
    model.zero_grad()
 
    for idx in range(args.num_train_epochs): 
        for step, batch in enumerate(train_dataloader):
            inputs = batch[0].to(args.device)        
            labels = batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)
            
            if args.n_gpu > 1:
                loss = loss.mean()  

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            losses.append(loss.item())
            
            if (step+1)% 100==0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(np.mean(losses[-100:]),4)))

                
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()  
            
            if len(losses) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer,args.eval_data_file,pool)                 
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,4))    
                    
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best f1:%s",round(best_f1,4))
                    logger.info("  "+"*"*20)                          

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)                        
                    model_to_save = model.module if hasattr(model,'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format(f'model_{args.lamda}.bin')) 
                    torch.save(model_to_save.state_dict(), output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                        



def evaluate(args, model, tokenizer, data_file, pool, dotest=False):
    """ Evaluate the model """
    eval_output_dir = args.output_dir
    eval_dataset = TextDataset(tokenizer, args, data_file,pool)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []  
    y_trues = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,cos_sim = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(cos_sim.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = np.argmax(logits, axis=1)

    recall=recall_score(y_trues, y_preds,average="macro")
    precision=precision_score(y_trues, y_preds,average="macro")   
    f1=f1_score(y_trues, y_preds,average="macro")
    acc = accuracy_score(y_trues,y_preds)
    result = {
        "eval_loss": eval_loss / nb_eval_steps,
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "acc": float(acc)
    }
    if dotest:

        conf_matrix = confusion_matrix(y_trues, y_preds)
        class_report = classification_report(y_trues, y_preds,digits=4)


        logger.info("***** TEST results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        logger.info("***** Classification Report *****")
        logger.info("\n%s", class_report)

        class_precisions = precision_score(y_trues, y_preds, average=None)
        class_recalls = recall_score(y_trues, y_preds, average=None)
        class_f1s = f1_score(y_trues, y_preds, average=None)

        for i, class_label in enumerate(np.unique(y_trues)):
            logger.info("Class %d: Precision = %.4f, Recall = %.4f, F1 Score = %.4f",
                        class_label, class_precisions[i], class_recalls[i], class_f1s[i])


    return result

def predict(args, model, tokenizer, data_file, output_file, pool):
    """ Make predictions with the model """
    predict_dataset = TextDataset(tokenizer, args, data_file, pool)
    predict_sampler = SequentialSampler(predict_dataset)
    predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.eval_batch_size, num_workers=4)

    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(predict_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    logits = []
    y_trues = []
    for batch in predict_dataloader:
        inputs = batch[0].to(args.device)
        labels = batch[1].to(args.device)
        with torch.no_grad():
            _, cos_sim = model(inputs, labels)
            logits.append(cos_sim.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

    logits = np.concatenate(logits, 0)
    y_preds = np.argmax(logits, axis=1)
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  

    with open(data_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        csv_reader = csv.reader(f_in)
        csv_writer = csv.writer(f_out)
        header = next(csv_reader)
        csv_writer.writerow(header + ['predict_label', 'ratio'])
        for i, row in enumerate(csv_reader):
            csv_writer.writerow(row + [y_preds[i], probabilities[i][y_preds[i]]])

    logger.info("***** Prediction results saved to %s *****", output_file)
                                                
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")    
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--pred_data_file", default=None, type=str,
                        help="An optional input pred data file.")
    parser.add_argument("--model_name_or_path", default="microsoft/unixcoder-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--ls_size", default=512, type=int,
                        help="The length of the logging statement.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_pred", action='store_true')  
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--lamda', type=float, default=0.1)
    pool = multiprocessing.Pool(cpu_cont)
    

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    new_tokens = ["@return", "@param", "<KEEP_END>", "<KEEP>", "<DELETE>", "<DELETE_END>", "<REPLACE_NEW>","<REPLACE_END>", "<REPLACE_OLD>", "<INSERT>","<INSERT_END>"]
    num_added_tokens = tokenizer.add_tokens(new_tokens)  
    print(f"Added {num_added_tokens} new tokens to the tokenizer.")

    tokenizer.save_pretrained("./updated_tokenizer")
    print("Tokenizer with new tokens saved to './updated_tokenizer'")

    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model vocab size to {len(tokenizer)}")

    model = Model(model, config, tokenizer, args)

    logger.info(f"Training/evaluation parameters: {args}")

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)      
    
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, args.train_data_file, pool=pool)
        train(args, train_dataset, model, tokenizer, pool)
            
    results = {}
    if args.do_eval:
        checkpoint_prefix = f'checkpoint-best-f1/model_{args.lamda}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))      
        result = evaluate(args, model, tokenizer, args.eval_data_file, pool = pool)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],2)))
            
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/model_{args.lamda}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))       
        result = evaluate(args, model, tokenizer, args.test_data_file, pool = pool,dotest=True)
        logger.info("***** Test results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key]*100 if "map" in key else result[key],2)))

    if args.do_pred:
        checkpoint_prefix = f'checkpoint-best-f1/model_{args.lamda}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model_to_load = model.module if hasattr(model, 'module') else model  
        model_to_load.load_state_dict(torch.load(output_dir))
        output_name = args.pred_data_file.split('/')[-1].split('.csv')[0]       
        predict(args, model, tokenizer, args.pred_data_file, f'{output_name}_prediction_result_{args.lamda}.csv', pool=pool)

if __name__ == "__main__":
    main()
