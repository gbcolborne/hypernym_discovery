import argparse, os, json
import torch
from transformers import XLMTokenizer, XLMConfig, XLMModel

""" Given a PyTorch checkpoint for an XLM model, dump a Transformers
pre-trained model.

Note: some properties are hard-coded assuming we are working on the
SemEval-2018 hypernym discovery task, so the languages are EN, ES, and
IT, for instance. Also, I assume that the tokenizer should not lower
case and remove accents, as accents are important in 2 of these
languages.

"""

def convert_vocab(word2id):
    """
    Convert a fastBPE vocabulary to XLMTokenizer format.
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
    word2id_mod = {}
    for k, v in word2id.items():
        if k not in special_tokens:
            if k.endswith('@@'):
                k = k[:-2]
            else:
                k += '</w>'
        word2id_mod[k] = v
    assert len(word2id_mod) == len(word2id)
    return word2id_mod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("bpe_codes")
    parser.add_argument("dir_output")
    args = parser.parse_args()

    if os.path.exists(args.dir_output):
        msg = "ERROR: output_dir already exists"
        raise ValueError(msg)
    os.makedirs(args.dir_output)

    # Load checkpoint
    cp = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    # Load params from checkpoint
    params = cp["params"]
    print("Params loaded from checkpoint: ")
    for p in sorted(params):
        print("  {}: {}".format(p, params[p]))
    
    # Convert vocab to Transformers format
    word2id = cp['dico_word2id']
    word2id = convert_vocab(word2id)
    id2word = {i:w for w,i in word2id.items()}
    print("First items of vocab:")
    for i in range(10):
        print("  {}. {}".format(i, id2word[i]))
    print("Vocab size: {}".format(len(word2id)))        

    # Write vocab in Transformers format
    vocab_path = os.path.join(args.dir_output, "vocab.json")
    with open(vocab_path, 'w', encoding='utf8') as f:
       json.dump(word2id, f, indent=2)

    # Create XLMTokenizer
    lang2id = params["lang2id"]
    id2lang = params["id2lang"]
    tokenizer = XLMTokenizer(vocab_path,
                             args.bpe_codes,
                             unk_token="<unk>",
                             bos_token="<s>",
                             sep_token="</s>",
                             pad_token="<pad>",
                             cls_token="</s>",
                             mask_token="<special1>",
                             additional_special_tokens=["<special0>",
                                                        "<special1>",
                                                        "<special2>",
                                                        "<special3>",
                                                        "<special4>",
                                                        "<special5>",
                                                        "<special6>",
                                                        "<special7>",
                                                        "<special8>",
                                                        "<special9>",
                             ],
                             lang2id=lang2id,
                             id2lang=id2lang,
                             do_lowercase_and_remove_accent=False)
    print("Tokenizer: {}".format(tokenizer))
    print("Vocab size of tokenizer (including added tokens): {}".format(len(tokenizer)))

    # Save tokenizer
    tokenizer.save_pretrained(args.dir_output)
    
    # Create XLMConfig for the pre-trained model
    vocab_size = len(tokenizer)
    config = XLMConfig(vocab_size=vocab_size,
                       emb_dim=params["emb_dim"],
                       n_layers=params["n_layers"],
                       n_heads=params["n_heads"],
                       dropout=params["dropout"],
                       attention_dropout=params["attention_dropout"],
                       gelu_activation=params["gelu_activation"],
                       sinusoidal_embeddings=params["sinusoidal_embeddings"],
                       causal=False,
                       asm=params["asm"],
                       n_langs=params["n_langs"],
                       use_lang_emb=params["use_lang_emb"],
                       max_position_embeddings=params["bptt"],
                       embed_init_std=0.02209708691207961,
                       layer_norm_eps=1e-12, 
                       init_std=0.02,
                       bos_index=params["bos_index"],
                       eos_index=params["eos_index"],
                       pad_index=params["pad_index"],
                       unk_index=params["unk_index"],
                       mask_index=params["mask_index"],
                       is_encoder=True,
                       summary_type='first',
                       summary_use_proj=True,
                       summary_activation=None,
                       summary_proj_to_labels=True,
                       summary_first_dropout=0.1,
                       start_n_top=5,
                       end_n_top=5,
                       mask_token_id=0,
                       lang_id=0,
                       n_words=vocab_size
    )

    # Hack: for some reason, the config init function won't set the
    # n_words correctly no matter how I call it, so I set this
    # attribute manually. Maybe this issue is known, as they created a
    # setter for n_words which sets the vocab_size property (for
    # backwards compatibility, they say).
    config.n_words = vocab_size
    
    # Save config
    config.save_pretrained(args.dir_output)
    
    # Create XLM model
    # model = XLMModel(config)
    
    # Load pre-trained weights
    # TODO

    # Save model
    model.save_pretrained(args.dir_output)
