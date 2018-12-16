import argparse
from collections import defaultdict
import utils

doc=""" Given a (pre-tokenized) corpus and a vocab of terms, convert
words in corpus to lower case and replace multi-word terms in corpus
with a single token (by replacing spaces with underscores), and
optionally replace out-of-vocab tokens with <UNK>. Write resulting
corpus and a TSV file containing the frequency of terms in vocab (path
is like output corpus, but with suffix .vocab).

"""

def get_indices_unmasked_spans(mask):
    """ Given a mask array (list where masked elements are evaluated
    as True and unmasked elements are evaluated as False), return
    spans of unmasked list items."""
    spans = []
    start = 0
    while start < len(mask):
        if mask[start]:
            start += 1
            continue
        end = start
        for i in range(start+1, len(mask)):
            if mask[i]:
                break
            else:
                end = i
        spans.append((start, end))
        start = end + 1
    return spans

def extract_ngrams(tokens, n, ngram_vocab, term_to_freq):
    """ Given a list of tokens and a vocab of n-grams, extract list of
    non-overlapping n-grams found in tokens, using term frequency to
    resolve overlap, in a way that favours low-frequency n-grams.

    Args:
    - Tokens: list of tokens
    - n: size of n-grams
    - ngram_vocab: set of target n-grams
    - term_to_freq: dict that maps terms to their frequency

    Returns: 
    - List of (index, term) tuples, where index is the index of the
      first token of each n-gram, and term is the n-gram (joined with
      spaces).

    """
    ngrams_found = []
    for i in range(len(tokens)-n+1):
        term = " ".join(tokens[i:i+n])
        if term in ngram_vocab:
            ngrams_found.append((i,term))
    if len(ngrams_found) < 2:
        return ngrams_found
    # Eliminate overlap
    ngrams_filtered = ngrams_found[:1]
    for (start, term) in ngrams_found[1:]:
        prev_start, prev_term = ngrams_filtered[-1]
        if start - prev_start < n:
            if term not in term_to_freq or term_to_freq[term] < term_to_freq[prev_term]:
                ngrams_filtered[-1] = (start, term)
        else:
            ngrams_filtered.append((start, term))
    return ngrams_filtered

def get_formatted_sample(strings, max_sampled):
    sub = strings[:max_sampled]
    if len(strings) > max_sampled:
        sub.append("... ({}) more".format(len(strings)-max_sampled))
    return ", ".join(sub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("subtask", choices=["1A", "1B", "1C", "2A", "2B"], help="subtask")
    parser.add_argument("path_corpus", help="path of corpus (text file)")
    msg = ("path of directory containing datasets for SemEval-2018 Task 9, "
           "from which we extract vocab and queries")
    parser.add_argument("dir_datasets", help=msg)
    msg = """path of output file containing corpus (path of vocab will
be created by adding the suffix .vocab)"""
    parser.add_argument("path_output", help=msg)
    parser.add_argument("-r", "--replace_OOV", action="store_true", help="Replace OOV tokens with <UNK>")
    args = parser.parse_args()

    # Load candidates and queries
    print("Loading candidates and queries...")
    candidates, queries = utils.load_vocab(args.dir_datasets, args.subtask, lower_queries=True) 
    print("Nb candidates: {}".format(len(candidates)))
    print("Nb queries: {}".format(len(queries)))
    vocab = candidates.union(queries)
    print("Size of vocab: {}".format(len(vocab)))
    trigrams = set()
    bigrams = set()
    unigrams = set()
    for term in vocab:
        nb_words = len(term.split())
        if nb_words == 3:
            trigrams.add(term)
        elif nb_words == 2:
            bigrams.add(term)
        elif nb_words == 1:
            unigrams.add(term)
        else:
            msg = "Error: '{}' is not unigram, bigram or trigram".format(term)
            raise ValueError(msg)
    print("Nb unigrams: {}".format(len(unigrams)))
    print("Nb bigrams: {}".format(len(bigrams)))
    print("Nb trigrams: {}".format(len(trigrams)))
    
    # Count n-gram frequencies in corpus
    print("Counting lines in corpus...")
    nb_lines = sum(1 for line in open(args.path_corpus))
    print("Counting n-gram frequencies in corpus...")
    term_to_freq_in = defaultdict(int)
    line_count = 0
    with open(args.path_corpus) as f_in:
        for line in f_in:
            line_count += 1
            if line_count % 100000 == 0:
                msg = "{}/{} lines processed.".format(line_count, nb_lines)
                msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
                print(msg)
            line = line.strip().replace("_", "")
            words = [w.lower() for w in line.split()]
            for n in [1,2,3]:
                for i in range(len(words)+n-1):
                    term = " ".join(words[i:i+n])
                    if term in vocab:
                        term_to_freq_in[term] += 1
    msg = "{}/{} lines processed.".format(line_count, nb_lines)
    msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_in), len(vocab))
    print(msg)
    nb_missing_q = sum(1 for w in queries if term_to_freq_in[w] == 0)
    nb_missing_c = sum(1 for w in candidates if term_to_freq_in[w] == 0)
    print("Nb zero-frequency queries: {}".format(nb_missing_q))
    print("Nb zero-frequency candidates: {}".format(nb_missing_c))

    # Replace multi-word terms with single tokens
    print("\nProcessing corpus...")
    term_to_freq_out = defaultdict(int)
    line_count = 0
    with open(args.path_corpus) as f_in, open(args.path_output, "w") as f_out:
        for line in f_in:
            line_count += 1
            if line_count % 100000 == 0:
                msg = "{}/{} lines processed.".format(line_count, nb_lines)
                msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
                print(msg)
            line = line.strip().replace("_", "")
            words = [w.lower() for w in line.split()]
            # Make array indicating the length of the term found at each
            # position
            term_lengths = [0 for _ in range(len(words))]
            # Make array indicating which indices are masked because a
            # term has already been found there
            masked_indices = [0 for _ in range(len(words))]
            # Check for trigrams
            trigrams_found = extract_ngrams(words, 3, trigrams, term_to_freq_in)
            for (i, term) in trigrams_found:
                term_lengths[i] = 3
                term_to_freq_out[term] += 1
                masked_indices[i] = 1
                masked_indices[i+1] = 1
                masked_indices[i+2] = 1
            # Check for bigrams
            for (beg, end) in get_indices_unmasked_spans(masked_indices):    
                bigrams_found = extract_ngrams(words[beg:end+1], 2, bigrams, term_to_freq_in)                
                for (i, term) in bigrams_found:
                    term_lengths[beg+i] = 2
                    term_to_freq_out[term] += 1
                    masked_indices[beg+i] = 1
                    masked_indices[beg+i+1] = 1
            # Check for unigrams
            for (beg, end) in get_indices_unmasked_spans(masked_indices):    
                for i in range(beg,end+1):
                    term = words[i]
                    if term in unigrams:
                        term_to_freq_out[term] += 1
                        term_lengths[i] = 1
            # Write sentence
            norm_terms = []
            i = 0
            while i < len(term_lengths):
                n = term_lengths[i] 
                if n > 1:
                    norm_terms.append("_".join(words[i:i+n]))
                    i += n
                else:
                    if args.replace_OOV and n == 0:
                        norm_term = "<UNK>"
                    else:
                        norm_term = words[i]
                    norm_terms.append(norm_term)
                    i += 1
            sent = " ".join(norm_terms)
            f_out.write(sent+"\n")
    msg = "{}/{} lines processed.".format(line_count, nb_lines)
    msg += " Vocab coverage: {}/{}.".format(len(term_to_freq_out), len(vocab))
    print(msg)
    missing_q = [w for w in queries if term_to_freq_out[w] == 0]
    missing_c = [w for w in candidates if term_to_freq_out[w] == 0]
    print("Nb missing queries in output: {}".format(len(missing_q)))
    max_shown = 200
    if len(missing_q):
        msg = "Examples: {}".format(get_formatted_sample(sorted(missing_q), max_shown))
        print(msg)
    print("Nb missing candidates in output: {}".format(len(missing_c)))
    if len(missing_c):
        msg = "Examples: {}".format(get_formatted_sample(sorted(missing_c), max_shown))
        print(msg)

    msg = "\nWrote corpus --> {}".format(args.path_output)
    print(msg)

    # Write frequencies
    path_output = args.path_output + ".vocab"
    with open(path_output, "w") as f:
        for term, freq in sorted(term_to_freq_out.items(), key=lambda x:x[0]):
            # Normalize term
            term_norm = "_".join(term.split())
            f.write("{}\t{}\n".format(term_norm, freq))
    msg = "Wrote vocab --> {}".format(path_output)
    print(msg)
