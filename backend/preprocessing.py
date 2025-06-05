from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("data/frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

clip_model = SentenceTransformer("clip-ViT-B-32")

def correct_spelling(query):
    suggestions = sym_spell.lookup_compound(query, max_edit_distance=2)
    return suggestions[0].term if suggestions else query

def score_sentence_with_gpt2(sentence, model, tokenizer, device="cpu"):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
    loss = outputs.loss.item()
    return loss

def process_query(query):
    corrected_query = correct_spelling(query)
    candidates = [query, corrected_query]
    scores = [(c, score_sentence_with_gpt2(c, gpt2_model, gpt2_tokenizer, device)) for c in candidates]
    best_candidate = sorted(scores, key=lambda x: x[1])[0][0]
    print(f"Best correction: {best_candidate}")
    variations = [
        best_candidate,
        f"A picture of {best_candidate}",
        f"Photograph showing {best_candidate}",
        f"Scene with {best_candidate}"
    ]
    vectors = clip_model.encode(variations, normalize_embeddings=True)
    return vectors.mean(axis=0), best_candidate
