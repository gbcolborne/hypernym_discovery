#!/usr/bin/env  python3

import torch
#import transformers

from argparse import ArgumentParser
from pathlib import Path
from transformers import XLMConfig
from transformers import XLMModel
from transformers import XLMTokenizer
from tempfile import NamedTemporaryFile



def convert_vocab(dico):
   """
   Convert a fastBPE vocabulary to a HuggingFace's tokenizers format.
   From a suffix continuation form `token@@ ization`
   To a suffix word termination form `token ization</w>`
   """
   special_tokens = [
     "<s>",
     "</s>",
     "<pad>",
     "<unk>",
     "<special0>",
     "<special1>",
     "<special2>",
     "<special3>",
     "<special4>",
     "<special5>",
     "<special6>",
     "<special7>",
     "<special8>",
     "<special9>",
   ]
   new_dico = {}
   for k, v in dico.items():
      if k not in special_tokens:
          if k.endswith('@@'):
              k = k[:-2]
          else:
              if not k.endswith('</w>'):
                 k += '</w>'
      new_dico[k] = v

   assert len(dico) == len(new_dico)
   return new_dico



def xlm_convert_to_huggingface(args):
   """
   Given a FaceBook's XLM model checkpoint, a BPE merges file, create and save
   a HuggingFace XLMTokenizer and a XLMModel.
   """
   xlm_pth = torch.load(args.checkpoint, map_location=torch.device('cpu'))

   with NamedTemporaryFile() as tfile:
      tfile.write(b'{}')
      tfile.flush()
      tokenizer = XLMTokenizer(
         tfile.name,
         args.merges,
         do_lowercase_and_remove_accent=False)
   tokenizer.encoder = convert_vocab(xlm_pth['dico_word2id'])
   vocab_size = len(tokenizer)
      
   params = xlm_pth['params']
   xlm_config = XLMConfig(
      emb_dim=params['emb_dim'],
      vocab_size=params['n_words'],
      n_layers=params['n_layers'],
      n_heads=params['n_heads'],
      n_langs=params['n_langs'],
      sinusoidal_embeddings=params['sinusoidal_embeddings'],
      use_lang_emb=params['use_lang_emb'],
      is_encoder=params['encoder_only'],
      output_hidden_states=True,
      n_words = params['n_words'],
   )
   
   # Provide both config and state dict to model init
   model = XLMModel.from_pretrained(
      None,
      config=xlm_config,
      state_dict=xlm_pth['model'])

   # Save
   save_directory = Path(args.output_dir)
   if not save_directory.exists():
      save_directory.mkdir(parents=True, exist_ok=True)
   model.save_pretrained(str(save_directory))
   tokenizer.save_pretrained(str(save_directory))
   tokenizer.save_vocabulary(str(save_directory))



def getArgs():
   usage="xlm2HuggingFace.py [options] checkpoint.pth merges.txt"
   help="""
      Given a XLM model trained with FaceBook's XLM and the BPE merges,
      converts them into a HuggingFace's XLMTokenizer and XLMModel to disk.
      """

   # Use the argparse module, not the deprecated optparse module.
   #parser = ArgumentParser(usage=usage, description=help, add_help=True)
   parser = ArgumentParser(usage=usage, description=help)

   parser.add_argument("-o",
         '--output_dir',
         dest="output_dir",
         type=str,
         default="HuggingFace",
         help="a directory name to write the HuggingFace XLMTokenizer & XLMModel [%(default)s]")

   parser.add_argument("checkpoint",
         type=str,
         help="FaceBook's XLM checkpoint filename.")
   parser.add_argument("merges",
         type=str,
         help="BPE merge operation filename.")

   cmd_args = parser.parse_args()

   return cmd_args






if __name__ == '__main__':
   args = getArgs()
   xlm_convert_to_huggingface(args)
