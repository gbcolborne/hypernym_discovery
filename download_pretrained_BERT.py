import arparse
import torch
import transformers

""" Download and save a pretrained BERT model and tokenizer. """

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, default="bert-base-cased")
parser.add_argument("cache_dir", type=str, help="Directory where downloaded model and tokenizer will be saved.")
args = parser.parse_args()

pretrained_weights = args.model_name
tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_weights)
tokenizer.save_pretrained(args.cache_dir)
model = transformers.BertModel.from_pretrained(pretrained_weights)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(args.cache_dir)
