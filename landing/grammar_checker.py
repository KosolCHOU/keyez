import torch
import torch.nn as nn
import fasttext
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import threading

# Paths - Use absolute paths for Django context
BASE_DIR = Path(__file__).resolve().parent
FASTTEXT_PATH = '/home/kosol/khmer_text/cc.km.300.bin'  # Absolute path to FastText model
GRU_MODEL_PATH = BASE_DIR / 'model' / 'gru_model.pth'
GRU_FALLBACK_PATH = BASE_DIR / 'model' / 'final_cv_based_gru.pth'
POS_MODEL_NAME = 'seanghay/khmer-pos-roberta'

_lock = threading.Lock()
_predictor = None

class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, embedded_text, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(embedded_text, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden)

class KhmerPOSTagger:
    def __init__(self, model_name=POS_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipe = pipeline('token-classification', model=self.model, tokenizer=self.tokenizer, aggregation_strategy='simple')
    def tag(self, sentence):
        try:
            out = self.pipe(sentence)
            tokens = [t['word'] for t in out]
            tags = [t['entity_group'] for t in out]
            return tokens, tags
        except Exception:
            return [], []

class FeaturePipeline:
    def __init__(self, embedding_model, pos_tagger):
        self.embedding_model = embedding_model
        self.pos_tagger = pos_tagger
        self.vocab = set(embedding_model.get_words())
        self.dim = embedding_model.get_dimension()
    def _vec(self, w):
        try:
            return self.embedding_model.get_word_vector(w)
        except Exception:
            return np.zeros(self.dim)
    def extract(self, sentence):
        tokens, pos = self.pos_tagger.tag(sentence)
        if not tokens:
            return None
        oov = [t for t in tokens if t not in self.vocab]
        oov_ratio = len(oov) / max(len(tokens), 1)
        # Simple grammar score
        has_noun = any(tag.startswith('NN') or tag.startswith('PR') for tag in pos)
        has_verb = any(tag.startswith('VB') or tag == 'AUX' for tag in pos)
        grammar_score = 0.5 + (0.3 if has_noun and has_verb else 0) + (0.2 if sum(tag.startswith('NN') for tag in pos) > 1 else 0)
        grammar_score = min(grammar_score, 1.0)
        # Semantic coherence (adjacent cosine avg)
        if len(tokens) < 2:
            coherence = 0.5
        else:
            sims = []
            for i in range(len(tokens)-1):
                v1 = self._vec(tokens[i]); v2 = self._vec(tokens[i+1])
                sims.append(cosine_similarity([v1],[v2])[0][0])
            coherence = float(np.mean(sims)) if sims else 0.0
        return {
            'tokens': tokens,
            'pos_tags': pos,
            'oov_words': oov,
            'oov_ratio': oov_ratio,
            'grammar_score': grammar_score,
            'semantic_coherence': coherence,
            'sentence_length': len(tokens)
        }

class SentencePredictor:
    def __init__(self, model, embedding_model, feature_pipeline, device):
        self.model = model
        self.embedding_model = embedding_model
        self.fp = feature_pipeline
        self.device = device
        self.model.eval()
    def predict(self, sentence):
        feats = self.fp.extract(sentence)
        if feats is None:
            return None
        embs = []
        for tok in feats['tokens']:
            try:
                embs.append(self.embedding_model.get_word_vector(tok))
            except Exception:
                embs.append(np.zeros(self.embedding_model.get_dimension()))
        emb_t = torch.FloatTensor(np.array(embs)).unsqueeze(0).to(self.device)
        len_t = torch.LongTensor([len(feats['tokens'])]).to(self.device)
        with torch.no_grad():
            out = self.model(emb_t, len_t)
            probs = torch.softmax(out, dim=1)
            conf, pred_cls = torch.max(probs, 1)
        label = 'Correct' if pred_cls.item() == 0 else 'Incorrect'
        return {'prediction': label, 'confidence': float(conf.item()), 'features': feats}


def _build_predictor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding = fasttext.load_model(FASTTEXT_PATH)
    pos_tagger = KhmerPOSTagger()
    EMBEDDING_DIM=300; HIDDEN_DIM=256; OUTPUT_DIM=2; N_LAYERS=2; DROPOUT=0.5
    gru = GRUClassifier(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT)
    if GRU_MODEL_PATH.exists():
        state = torch.load(str(GRU_MODEL_PATH), map_location=device)
        model_type = 'Primary'
    else:
        state = torch.load(str(GRU_FALLBACK_PATH), map_location=device)
        model_type = 'Fallback'
    gru.load_state_dict(state)
    if device.type == 'cpu':
        try:
            gru = torch.quantization.quantize_dynamic(gru, {nn.GRU, nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
    gru.to(device)
    fp = FeaturePipeline(embedding, pos_tagger)
    return SentencePredictor(gru, embedding, fp, device), model_type


def get_predictor():
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                _predictor, _ = _build_predictor()
    return _predictor


def analyze_sentence(sentence: str):
    predictor = get_predictor()
    return predictor.predict(sentence)
