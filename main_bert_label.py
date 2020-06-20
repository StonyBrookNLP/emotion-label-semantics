import torch
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses import BCEWithLogLoss
from pybert.train.trainer_old import Trainer
from torch.utils.data import DataLoader
from pybert.io.bert_processor_label2 import BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config_label_finetune import config
from pybert.model.nn.bert_for_multi_label import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics import AUC, AccuracyThresh, MultiLabelReport
from transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pickle
from sklearn import metrics

warnings.filterwarnings("ignore")

# def plot_sim_matrix(label_similarity_matrix, label_list):

#   import matplotlib.pyplot as plt
   
#   labels = label_list[:]

#   # for hood in hood_menu_data:
#   #   labels.append(hood["properties"]['NAME'])
   
#   fig, ax = plt.subplots(figsize=(20,20))
#   cax = ax.matshow(label_similarity_matrix, interpolation='nearest')
#   ax.grid(True)
#   plt.title('Label Embeddings Similarity matrix')
#   plt.xticks(range(33), labels, rotation=90);
#   plt.yticks(range(33), labels);

#   fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,1])

#   # plt.show()
#   plt.savefig(config['checkpoint_dir'] / f"label_similarity_bert_vis.png")

# def get_label_similarity_matrix(label_embeddings_dict, label_list):
  
#   import pandas as pd

#   label2id = {label: i for i, label in enumerate(label_list)}

#   embeddings_matrix = np.zeros((len(label_list), 768))
#   print ("Labels:")
#   print (label_embeddings_dict.keys())

#   for word in label_list:
#     embeddings_matrix[label2id[word]] = label_embeddings_dict[word]

#   print ("embeddings_matrix")
#   print (embeddings_matrix)
#   print ("                    ")
#   cosine_sim_matrix = metrics.pairwise.cosine_similarity(embeddings_matrix)

#   print (cosine_sim_matrix)

#   cosine_sim_df = pd.DataFrame(data=cosine_sim_matrix, index=label_list, columns=label_list)
  
#   print ("Similarity matrix:")
#   print (cosine_sim_df)
  
#   pickle.dump(cosine_sim_df, open(config['checkpoint_dir'] / f"label_similarity_matrix_bert.p", "wb"))

#   plot_sim_matrix(cosine_sim_matrix, label_list)

#   return cosine_sim_matrix

def run_train(args):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)

    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_data = processor.get_train(config['data_dir'] / f"{args.data_name}.label_train.pkl")

    print ("Train data is:")
    print (train_data)

    train_examples = processor.create_examples(lines=train_data,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_cache'] / f"cached_train_label_examples_finetune{args.arch}")

    # print ("Training examples are:")
    # print (train_examples)
    train_features = processor.create_features(examples=train_examples,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_cache'] / "cached_train_label_features_finetune{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))

    train_dataset = processor.create_dataset(train_features, is_sorted=args.sorted)

    if args.sorted:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = processor.get_dev(config['data_dir'] / f"{args.data_name}.label_valid.pkl")
    
    valid_examples = processor.create_examples(lines=valid_data,
                                               example_type='valid',
                                               cached_examples_file=config[
                                                                        'data_cache'] / f"cached_valid_examples_label_finetune{args.arch}")

    valid_features = processor.create_features(examples=valid_examples,
                                               max_seq_len=args.eval_max_seq_len,
                                               cached_features_file=config[
                                                                        'data_cache'] / "cached_valid_features_label_finetune{}_{}".format(
                                                   args.eval_max_seq_len, args.arch
                                               ))


    valid_dataset = processor.create_dataset(valid_features)
    valid_sampler = SequentialSampler(valid_dataset)

    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    # ------- model
    logger.info("initializing model")

    if args.resume_path:
        args.resume_path = Path(args.resume_path)
        model = BertForMultiLable.from_pretrained(args.resume_path, num_labels=len(label_list))
    
    else:
        print ("Labels are:")
        print (label_list)
        # model = BertForMultiLable.from_pretrained(config['bert_model_dir'], num_labels=len(label_list))
        model = BertForMultiLable.from_pretrained("bert-base-uncased", num_labels=len(label_list))
        
    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # ---- callbacks
    logger.info("initializing callbacks")
    train_monitor = TrainingMonitor(file_dir=config['figure_dir'], arch=args.arch)

    model_checkpoint = ModelCheckpoint(checkpoint_dir=config['checkpoint_dir'],mode=args.mode,
                                       monitor=args.monitor,arch=args.arch,
                                       save_best_only=args.save_best)

    # **************************** training model ***********************
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    trainer = Trainer(n_gpu=args.n_gpu,
                      model=model,
                      epochs=args.epochs,
                      logger=logger,
                      criterion=BCEWithLogLoss(),
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      early_stopping=None,
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      epoch_metrics=[AUC(average='micro', task_type='binary'),
                                     MultiLabelReport(id2label=id2label)])

    # embeddings_dict = pickle.load(open("/home/rgaonkar/context_home/rgaonkar/label_embeddings/code/Bert_Masked_LM/label_embeddings_dict.p", "rb"))

    # label_similarity_matrix = get_label_similarity_matrix(embeddings_dict, label_list)

    trainer.train(train_data=train_dataloader, valid_data=valid_dataloader, seed=args.seed)


def run_test(args):
    from pybert.io.task_data_label import TaskData
    from pybert.test.predictor import Predictor
    import pickle
    import os
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)

    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    test_data = processor.get_train(config['data_dir'] / f"{args.data_name}.label_test.pkl")
    print ("Test data is:")
    print (test_data)

    print ("Label list is:")
    print (label_list)
    print ("----------------------------------------")
    # test_data = processor.get_test(lines=lines)

    test_examples = processor.create_examples(lines=test_data,
                                              example_type='test',
                                              cached_examples_file=config[
                                                                       'data_cache'] / f"cached_test_examples_label_finetune{args.arch}")
    test_features = processor.create_features(examples=test_examples,
                                              max_seq_len=args.eval_max_seq_len,
                                              cached_features_file=config[
                                                                       'data_cache'] / "cached_test_features_label_finetune{}_{}".format(
                                                  args.eval_max_seq_len, args.arch
                                              ))
    test_dataset = processor.create_dataset(test_features)
    test_sampler = SequentialSampler(test_dataset)

    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.train_batch_size)
    
    model = BertForMultiLable.from_pretrained(config['checkpoint_dir'], num_labels=len(label_list))

    # ----------- predicting
    logger.info('model predicting....')
    predictor = Predictor(model=model,
                          logger=logger,
                          n_gpu=args.n_gpu,
                          batch_metrics=[AccuracyThresh(thresh=0.5)],
                          epoch_metrics=[AUC(average='micro', task_type='binary'),
                                     MultiLabelReport(id2label=id2label)])

    result, test_predicted, test_true = predictor.predict(data=test_dataloader)

    pickle.dump(test_true, open(os.path.join(config["test/checkpoint_dir"], "test_true.p"), "wb"))

    pickle.dump(test_predicted, open(os.path.join(config["test/checkpoint_dir"], "test_predicted.p"), "wb"))

    pickle.dump(id2label, open(os.path.join(config["test/checkpoint_dir"], "id2label.p"), "wb"))


    print ("Predictor results:")
    print(result)
    print ("-----------------------------------------------")

def main():
    parser = ArgumentParser()
    parser.add_argument("--arch", default='bert', type=str)
    parser.add_argument("--do_data", action='store_true')
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--save_best", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument('--data_name', default='kaggle', type=str)
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--resume_path", default='', type=str)
    parser.add_argument("--mode", default='min', type=str)
    parser.add_argument("--monitor", default='valid_loss', type=str)
    parser.add_argument("--valid_size", default=0.2, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--sorted", default=1, type=int, help='1 : True  0:False ')
    parser.add_argument("--n_gpu", type=str, default='0', help='"0,1,.." or "0" or "" ')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument("--train_max_seq_len", default=256, type=int)
    parser.add_argument("--eval_max_seq_len", default=256, type=int)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, )
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1')

    args = parser.parse_args()
    config['checkpoint_dir'] = config['checkpoint_dir'] / args.arch
    config['checkpoint_dir'].mkdir(exist_ok=True)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, config['checkpoint_dir'] / 'training_args.bin')
    seed_everything(args.seed)
    init_logger(log_file=config['log_dir'] / f"{args.arch}.log")

    logger.info("Training/evaluation parameters %s", args)

    if args.do_data:
        from pybert.io.task_data_label import TaskData
        data = TaskData()
        print ("Train data path:")
        print (config['raw_data_path'])
        targets, sentences_char = data.read_data(raw_data_path=config['raw_data_path'],
                                            preprocessor=EnglishPreProcessor(),
                                            is_train=True)

        print ("Target:")
        print (targets)
        print ("                          ")
        print ("Sentence:")
        print (sentences_char)
        print ("                          ")
        data.train_val_split(X=sentences_char, y=targets,
                             valid_size=args.valid_size, data_dir=config['data_dir'],
                             data_name=args.data_name)
    

        ##Get the test data
        targets_test, sentences_char_test = data.read_data(raw_data_path=config['test_path'], preprocessor=EnglishPreProcessor(), is_train=True)

        print (targets_test)

        data.save_test_data(X=sentences_char_test, y=targets_test,
                             data_dir=config['data_dir'],
                             data_name=args.data_name)

    if args.do_train:
        run_train(args)

    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
