
import sys
import os
import pandas as pd
import stanza
import networkx as nx

# Add the repository to python path if not already there
# This assumes the script is run from the root of the repo or textcomplexity folder is adjacent
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.join(current_dir, "textcomplexity") not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(current_dir, "textcomplexity")))

try:
    from textcomplexity.utils.token import Token
    from textcomplexity.utils.text import Text
    from textcomplexity import surface, sentence, pos, dependency
    from textcomplexity.cli import surface_based, sentence_based, dependency_based, pos_based
except ImportError:
    # Fallback if the script is run from a different location
    sys.path.append(os.path.abspath("textcomplexity"))
    from textcomplexity.utils.token import Token
    from textcomplexity.utils.text import Text
    from textcomplexity import surface, sentence, pos, dependency
    from textcomplexity.cli import surface_based, sentence_based, dependency_based, pos_based

def calculate_text_complexity(input_path, output_path, window_size=100, nlp=None):
    """
    Reads text from input_path, calculates text complexity features using Stanza and textcomplexity,
    and saves the results to output_path (CSV).

    Args:
        input_path (str): Path to the input text file.
        output_path (str): Path where the output CSV will be saved.
        window_size (int): Window size for surface-based measures. Default 100.
        nlp (stanza.Pipeline, optional): Pre-initialized Stanza pipeline. If None, one will be initialized.
    """
    
    print(f"Reading {input_path}...")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    if not text_content:
        print("Error: Input file is empty.")
        return

    # Initialize Stanza if not provided
    if nlp is None:
        print("Initializing Stanza Pipeline...")
        # We need tokenize, pos, lemma for sure. depparse for dependency measures.
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', logging_level='WARN')

    print("Processing text with Stanza...")
    doc = nlp(text_content)

    # --- Convert Stanza output to textcomplexity input formats ---
    all_tokens = []
    all_sentences = []
    all_graphs = []

    print("Converting to textcomplexity objects...")
    for sent in doc.sentences:
        sent_tokens = []
        dg = nx.DiGraph()
        stanza_id_to_idx = {}
        
        # 1. Add nodes
        for i, word in enumerate(sent.words):
            tc_token = Token(word.text, word.xpos, word.upos)
            sent_tokens.append(tc_token)
            stanza_id_to_idx[word.id] = i
            dg.add_node(i, word=word.text, lemma=word.lemma, wc=word.upos, pos=word.xpos)
        
        # 2. Add edges
        for i, word in enumerate(sent.words):
            head_id = str(word.head)
            if head_id == "0":
                dg.nodes[i]["root"] = "root"
            else:
                if head_id in stanza_id_to_idx:
                    head_idx = stanza_id_to_idx[head_id]
                    dg.add_edge(head_idx, i, relation=word.deprel)
        
        all_tokens.extend(sent_tokens)
        all_sentences.append(sent_tokens)
        all_graphs.append(dg)

    print(f"Extracted {len(all_tokens)} tokens from {len(all_sentences)} sentences.")

    # --- Run Measures ---
    results = {}

    # 1. Surface Based
    print("Running surface_based measures...")
    # Ensure window_size is valid
    if window_size > len(all_tokens):
        print(f"Warning: window_size ({window_size}) > text length ({len(all_tokens)}). Reducing to text length.")
        actual_window_size = len(all_tokens)
    else:
        actual_window_size = window_size

    try:
        surf_res = surface_based(all_tokens, window_size=actual_window_size, preset="lexical_core")
        for r in surf_res:
            results[f"surface_{r.name}"] = r.value
    except Exception as e:
        print(f"Error calculating surface measures: {e}")

    # 2. POS Based
    print("Running pos_based measures...")
    # Define English tags
    punct_tags = {".", "," ,":", "\"", "``", "(", ")", "-LRB-", "-RRB-"}
    name_tags = {"NNP", "NNPS"}
    open_tags = {"AFX", "JJ", "JJR", "JJS", "NN", "NNS", "RB", "RBR", "RBS", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    
    try:
        pos_res = pos_based(all_tokens, punct_tags, name_tags, open_tags, reference_frequency_list=None, preset="lexical_core")
        for r in pos_res:
            results[f"pos_{r.name}"] = r.value
    except Exception as e:
        print(f"Error calculating POS measures: {e}")

    # 3. Sentence Based
    print("Running sentence_based measures...")
    try:
        sent_res = sentence_based(all_sentences, punct_tags, preset="core")
        for r in sent_res:
            results[f"sentence_{r.name}"] = r.value
    except Exception as e:
        print(f"Error calculating sentence measures: {e}")

    # 4. Dependency Based
    print("Running dependency_based measures...")
    try:
        dep_res = dependency_based(all_graphs, preset="core")
        for r in dep_res:
            results[f"dependency_{r.name}"] = r.value
    except Exception as e:
        print(f"Error calculating dependency measures: {e}")

    # --- Save Output ---
    print(f"Saving results to {output_path}...")
    df = pd.DataFrame([results])
    
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Example usage when run as script
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        calculate_text_complexity(input_file, output_file)
    else:
        # Default for testing
        print("Usage: python calculate_text_complexity.py <input_file> <output_file>")
        print("Using default test paths...")
        calculate_text_complexity("data-inputs/narrative-text-1.txt", "narrative-text-1_calc_output.csv")
