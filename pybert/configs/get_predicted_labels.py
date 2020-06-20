import pandas as pd
import pickle
#get the predicted labels 
predicted_logits = pickle.load(open("/home/rgaonkar/context_home/rgaonkar/label_embeddings/code/Bert-Multi-Label-Text-Classification/pybert/output/checkpoints_label_finetune_soft_semi/bert/train_predicted.p", "rb"))

print (predicted_logits) 

