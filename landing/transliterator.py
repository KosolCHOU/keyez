"""
SingKhmer to Khmer transliteration using Seq2Seq model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

# Special tokens
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers, batch_first=True)
    
    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
    
    def forward(self, input_token, hidden):
        # input_token: (batch,)
        input_token = input_token.unsqueeze(1)        # (batch,1)
        embedded = self.embedding(input_token)        # (batch,1,emb_dim)
        output, hidden = self.rnn(embedded, hidden)   # output: (batch,1,hid_dim)
        logits = self.fc_out(output.squeeze(1))       # (batch,vocab)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
    
    def forward(self, src, src_lengths, tgt_in):
        batch_size, tgt_len = tgt_in.shape
        device = src.device
        outputs = torch.zeros(batch_size, tgt_len, self.tgt_vocab_size, device=device)

        hidden = self.encoder(src, src_lengths)
        input_token = tgt_in[:, 0]

        for t in range(tgt_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t, :] = logits

            if t + 1 < tgt_len:
                top1 = logits.argmax(1)
                input_token = top1

        return outputs


class SingKhmerTransliterator:
    """Handles SingKhmer to Khmer transliteration"""
    
    def __init__(self, model_path: Path):
        self.device = torch.device('cpu')  # Use CPU for now
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load vocabularies
        self.src_stoi = checkpoint['src_stoi']
        self.src_itos = checkpoint['src_itos']
        self.tgt_stoi = checkpoint['tgt_stoi']
        self.tgt_itos = checkpoint['tgt_itos']
        
        # Model hyperparameters (matching training architecture from singkhmer_model.py)
        INPUT_DIM = len(self.src_itos)
        OUTPUT_DIM = len(self.tgt_itos)
        EMB_DIM = 128
        HID_DIM = 256
        NUM_LAYERS = 1
        
        # Build model
        enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS)
        dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, NUM_LAYERS)
        self.model = Seq2Seq(enc, dec, OUTPUT_DIM).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        # Special tokens
        self.SOS_token = SOS_IDX
        self.EOS_token = EOS_IDX
        self.PAD_token = PAD_IDX
        self.UNK_token = UNK_IDX
        
    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token indices"""
        tokens = []
        for char in text.lower():
            tokens.append(self.src_stoi.get(char, self.UNK_token))
        return tokens
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Convert token indices to text"""
        chars = []
        for token in tokens:
            if token in [self.SOS_token, self.EOS_token, self.PAD_token]:
                continue
            if 0 <= token < len(self.tgt_itos):
                chars.append(self.tgt_itos[token])
        return ''.join(chars)
    
    def translate_with_beam_search(self, singkhmer: str, beam_width: int = 5, max_length: int = 50) -> List[Tuple[str, float]]:
        """
        Translate SingKhmer to Khmer using beam search
        Returns list of (khmer_text, score) tuples, sorted by score (best first)
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize input: add SOS and EOS
            src_chars = [self.SOS_token] + self._tokenize(singkhmer) + [self.EOS_token]
            src_tensor = torch.LongTensor(src_chars).unsqueeze(0).to(self.device)
            src_lengths = torch.LongTensor([len(src_chars)]).to(self.device)
            
            # Encode
            hidden = self.model.encoder(src_tensor, src_lengths)
            
            # Initialize beam with SOS token
            beams = [([self.SOS_token], 0.0, hidden)]
            completed = []
            
            for _ in range(max_length):
                candidates = []
                
                for tokens, score, h in beams:
                    if tokens[-1] == self.EOS_token:
                        completed.append((tokens, score))
                        continue
                    
                    # Decode one step
                    input_token = torch.LongTensor([tokens[-1]]).to(self.device)
                    output, new_h = self.model.decoder(input_token, h)
                    
                    # Get top k predictions
                    log_probs = F.log_softmax(output, dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                    
                    for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                        new_tokens = tokens + [idx.item()]
                        new_score = score + log_prob.item()
                        candidates.append((new_tokens, new_score, new_h))
                
                # Keep top beam_width candidates
                if not candidates:
                    break
                    
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                
                # Stop if all beams are completed
                if all(tokens[-1] == self.EOS_token for tokens, _, _ in beams):
                    completed.extend([(tokens, score) for tokens, score, _ in beams])
                    break
            
            # Add remaining beams to completed
            for tokens, score, _ in beams:
                if tokens not in [t for t, _ in completed]:
                    completed.append((tokens, score))
            
            # Sort by score and decode
            completed = sorted(completed, key=lambda x: x[1], reverse=True)
            
            results = []
            seen = set()
            for tokens, score in completed[:beam_width]:
                # Skip SOS token when decoding
                text = self._decode_tokens(tokens[1:])
                # Remove duplicates
                if text and text not in seen:
                    seen.add(text)
                    results.append((text, score))
            
            return results[:beam_width]
    
    def translate(self, singkhmer: str, top_k: int = 3) -> List[str]:
        """
        Translate SingKhmer to Khmer, returning top_k candidates
        """
        results = self.translate_with_beam_search(singkhmer, beam_width=top_k * 2)
        return [text for text, _ in results[:top_k]]


# Global transliterator instance
_transliterator = None

def get_transliterator() -> SingKhmerTransliterator:
    """Get or create the global transliterator instance"""
    global _transliterator
    if _transliterator is None:
        model_path = Path(__file__).resolve().parent / "model" / "singkhmer_seq2seq.pt"
        _transliterator = SingKhmerTransliterator(model_path)
    return _transliterator
