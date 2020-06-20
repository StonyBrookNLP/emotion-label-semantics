import torch
import warnings
from pathlib import Path
from argparse import ArgumentParser
from pybert.train.losses_soft_sig import ContinuousBCEWithLogLoss
from pybert.train.trainer_old_soft_joint_corr1 import Trainer
from torch.utils.data import DataLoader
from pybert.io.bert_processor_label2 import BertProcessor as BertProcessor
#from pybert.io.bert_processor_label2_semi_emotion import BertProcessor as BertProcessor
from pybert.common.tools import init_logger, logger
from pybert.common.tools import seed_everything
from pybert.configs.basic_config_label_finetune_soft_joint_corr_emotion import config
from pybert.model.nn.bert_for_multi_label_soft import BertForMultiLable
from pybert.preprocessing.preprocessor import EnglishPreProcessor
from pybert.callback.modelcheckpoint import ModelCheckpoint
from pybert.callback.trainingmonitor import TrainingMonitor
from pybert.train.metrics_soft import AUC, AccuracyThresh, MultiLabelReport
from transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import RandomSampler, SequentialSampler
import numpy as np
import pickle
from sklearn import metrics

import pickle
from sklearn import metrics
import pandas as pd

warnings.filterwarnings("ignore")

def get_label_corr(true_labels_matrix):

    label_columns = ["joy","trust","fear","surprise","sad","disgust","anger","anticipation"]

    print (true_labels_matrix)
    print (type(true_labels_matrix))
    # print (true_labels_matrix.shape)

    true_labels_df = pd.DataFrame(data = true_labels_matrix, columns=label_columns, dtype=np.float64)

    print (true_labels_df)

    correlation = true_labels_df.corr(method='pearson')

    correlation_matrix = correlation.to_numpy()

    print (correlation)
    print (type(correlation))
    print (correlation_matrix)

    return correlation_matrix


def plot_sim_matrix(label_similarity_matrix, label_list):

  import matplotlib.pyplot as plt

  labels = label_list[:]

  # for hood in hood_menu_data:
  #   labels.append(hood["properties"]['NAME'])

  fig, ax = plt.subplots(figsize=(20,20))
  cax = ax.matshow(label_similarity_matrix, interpolation='nearest')
  ax.grid(True)
  plt.title('Label Embeddings Similarity matrix')
  plt.xticks(range(33), labels, rotation=90);
  plt.yticks(range(33), labels);

  fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,1])

  # plt.show()
  plt.savefig(config['checkpoint_dir'] / f"label_similarity_bert_vis.png")

def get_label_similarity_matrix(label_embeddings_dict, label_list):

  import pandas as pd

  label2id = {label: i for i, label in enumerate(label_list)}

  embeddings_matrix = np.zeros((len(label_list), 768))
  print ("Labels:")
  print (label_embeddings_dict.keys())

  for word in label_list:
    embeddings_matrix[label2id[word]] = label_embeddings_dict[word]

  print ("embeddings_matrix")
  print (embeddings_matrix)
  print ("                    ")
  cosine_sim_matrix = metrics.pairwise.cosine_similarity(embeddings_matrix)

  print (cosine_sim_matrix)

  cosine_sim_df = pd.DataFrame(data=cosine_sim_matrix, index=label_list, columns=label_list)

  print ("Similarity matrix:")
  print (cosine_sim_df)

  pickle.dump(cosine_sim_df, open(config['checkpoint_dir'] / f"label_similarity_matrix_bert.p", "wb"))

  plot_sim_matrix(cosine_sim_matrix, label_list)

  return cosine_sim_matrix

def run_train(args):
    # --------- data
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)

    label_list = processor.get_labels()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    ##Get the data for the soft training task
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

    ###########################################################################
    ##Get data for the semi-supervised task

    # processor_semi = BertProcessor_semi(vocab_path=config['bert_vocab_path'], do_lower_case=args.do_lower_case)

    # label_list = processor_semi.get_labels()
    # label2id = {label: i for i, label in enumerate(label_list)}
    # id2label = {i: label for i, label in enumerate(label_list)}


    train_data_semi = processor.get_train_semi(config['unlabel_data_path'])

    print ("Train data is:")
    print (train_data)

    train_examples_semi = processor.create_examples_semi(lines=train_data_semi,
                                               example_type='train',
                                               cached_examples_file=config[
                                                    'data_cache'] / f"cached_train_unlabel_examples_finetune{args.arch}")

    # print ("Training examples are:")
    # print (train_examples)
    train_features_semi = processor.create_features_semi(examples=train_examples_semi,
                                               max_seq_len=args.train_max_seq_len,
                                               cached_features_file=config[
                                                    'data_cache'] / "cached_train_unlabel_features_finetune{}_{}".format(
                                                   args.train_max_seq_len, args.arch
                                               ))

    train_dataset_semi = processor.create_dataset_semi(train_features_semi, is_sorted=args.sorted)


    if args.sorted:
        train_sampler_semi = SequentialSampler(train_dataset_semi)
    else:
        train_sampler_semi = RandomSampler(train_dataset_semi)

    train_dataloader_semi = DataLoader(train_dataset_semi, sampler=train_sampler_semi, batch_size=args.train_batch_size)


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

        #model = BertForMultiLable.from_pretrained("pybert/output/checkpoints_label_finetune_soft_joint_corr_emotion/bert", num_labels=len(label_list))
        model = BertForMultiLable.from_pretrained("bert-base-uncased")

        # model = BertForMultiLable.from_pretrained("bert-base-uncased", num_labels=len(label_list))

    t_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)

    param_optimizer = list(model.named_parameters())
    # param_optimizer = list(filter(lambda named_param: named_param[1].requires_grad, model.named_parameters()))

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    ##For semi-supervision
    t_total_semi = int(len(train_dataloader_semi) / args.gradient_accumulation_steps * args.epochs)

    ##params for this model only contains the params other than the label graph
    # param_optimizer_semi = [(name, param) for (name, param) in list(model.named_parameters()) if "label_graph" not in name]

    param_optimizer_semi = [(name, param) for name, param in model.named_parameters() if name == 'label_graph.weight']
    # param_optimizer = list(filter(lambda named_param: named_param[1].requires_grad, model.named_parameters()))

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters_semi = [
        {'params': [p for n, p in param_optimizer_semi if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer_semi if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps_semi = int(t_total_semi * args.warmup_proportion)

    optimizer_semi = AdamW(optimizer_grouped_parameters_semi, lr=args.learning_rate, eps=args.adam_epsilon)

    lr_scheduler_semi = WarmupLinearSchedule(optimizer_semi, warmup_steps=warmup_steps_semi, t_total=t_total)


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
                      # criterion_hard=BCEWithLogLoss(),
                      criterion= ContinuousBCEWithLogLoss(),
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      optimizer_semi=optimizer_semi,
                      lr_scheduler_semi=lr_scheduler_semi,
                      early_stopping=None,
                      training_monitor=train_monitor,
                      fp16=args.fp16,
                      resume_path=args.resume_path,
                      grad_clip=args.grad_clip,
                      model_checkpoint=model_checkpoint,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      batch_metrics=[AccuracyThresh(thresh=0.5)],
                      ##Only look at the f1 score
                      epoch_metrics=[MultiLabelReport(id2label=id2label)])

    # embeddings_dict = pickle.load(open("/home/rgaonkar/context_home/rgaonkar/label_embeddings/code/Bert_Masked_LM/label_embeddings_dict.p", "rb"))

    # label_similarity_matrix = get_label_similarity_matrix(embeddings_dict, label_list)

    # ------- model
    logger.info("initializing model")

    true_labels_matrix = [sample[-1].tolist() for sample in train_dataset]
    print ("True train labels:")
    print (true_labels_matrix)

    train_label_corr = get_label_corr(true_labels_matrix)

    print ("True train label correlations:")
    print (train_label_corr)

    #Save the correlation matrix of the true labels in the data cache folder
    pickle.dump(train_label_corr, open(config['data_cache'] / "train_label_corr.p", "wb"))

    trainer.train(train_data=train_dataloader, train_data_semi=train_dataloader_semi, valid_data=valid_dataloader, seed=args.seed, prob_thresh=args.prob_thresh, true_label_corr=train_label_corr, tokenizer=processor.tokenizer, args=args)


def run_test(args):
    import os
    import pickle
    from pybert.io.task_data_label import TaskData
    from pybert.test.predictor_soft import Predictor

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

    result, test_predicted, test_true, label_graph = predictor.predict(data=test_dataloader)
    ##Save the true labels from the training set
    pickle.dump(test_true, open(os.path.join(config["test/checkpoint_dir"], "test_true.p"), "wb"))

    pickle.dump(test_predicted, open(os.path.join(config["test/checkpoint_dir"], "test_predicted.p"), "wb"))

    pickle.dump(label_graph, open(os.path.join(config["test/checkpoint_dir"], "learned_label_corr.p"), "wb"))

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
    parser.add_argument("--prob_thresh", default=0.5, type=float)

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
