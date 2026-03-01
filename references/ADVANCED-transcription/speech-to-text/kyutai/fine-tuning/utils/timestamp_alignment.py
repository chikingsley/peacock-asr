from typing import Dict, Any, List, Tuple, Optional
import bisect


def _hhmmss_to_s(t: Optional[str]) -> float:
    """Convert HH:MM:SS string to seconds."""
    if not t: 
        return 0.0
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)


def build_word_spans(text: str, words: List[str]) -> List[Tuple[int,int]]:
    """Build character spans for each word in the text."""
    spans, cursor = [], 0
    for w in words:
        i = text.find(w, cursor)
        if i < 0:
            raise ValueError(f"Couldn't locate word '{w}' in text starting at {cursor}")
        spans.append((i, i+len(w)))
        cursor = i + len(w)
    return spans


def assign_tokens_to_words(text: str, tokenizer, word_spans: List[Tuple[int,int]]):
    """Assign tokens to words using simple sequential approach."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    toks = tokenizer.convert_ids_to_tokens(ids)

    per_word = [[] for _ in word_spans]
    current_word_idx = 0
    
    for tid, (cs, ce), tok in zip(ids, offs, toks):
        if cs == ce:  # Skip empty tokens
            continue
            
        # Handle SentencePiece space-prefixed tokens that span word boundaries
        while current_word_idx < len(word_spans) - 1:
            next_word_start = word_spans[current_word_idx + 1][0]
            
            # If token starts at or after next word, definitely advance
            if cs >= next_word_start:
                current_word_idx += 1
            # If token starts right before next word and has space prefix, assign to next word
            elif tok.startswith('▁') and cs == next_word_start - 1:
                current_word_idx += 1
                break
            else:
                break
        
        # Assign token to current word
        if current_word_idx < len(per_word):
            per_word[current_word_idx].append((tid, tok))
            
    return per_word


def schedule_tokens_discrete(
    sample: Dict[str, Any],
    tokenizer,
    stt_delay: float = 0.5,          # Kyutai-style text-audio delay
    frame_hop_s: float = 0.08,       # one token per frame (80ms at 24kHz)
    unk_id: int = 0,                 # their "0"
    pad_id: int = 3,                 # their "3"
    spillover: str = "shift",     # "truncate" or "shift"
    pad_to_segment_end: bool = True,
):
    """
    Returns a time-ordered list: {token_id, token, start, end} in seconds.
    
    Args:
        sample: Dict with 'text', 'word_timestamps', 'start_time', 'end_time'
        tokenizer: Tokenizer with convert_ids_to_tokens method
        stt_delay: Delay to apply to word timestamps (seconds)
        frame_hop_s: Time per token frame (seconds)
        unk_id: Token ID for UNK tokens
        pad_id: Token ID for PAD tokens
        spillover: Policy for handling token overflow ("truncate" or "shift")
        pad_to_segment_end: Whether to pad to segment end time
        
    Returns:
        List of dicts with keys: token_id, token, start, end
    """
    text = sample["text"]
    seg_start = _hhmmss_to_s(sample.get("start_time"))
    seg_end = _hhmmss_to_s(sample.get("end_time")) if sample.get("end_time") else None

    # 1) use the *full-string* tokenization so token IDs match production
    words_ts = sample["word_timestamps"]
    words = [w["word"] for w in words_ts]
    spans = build_word_spans(text, words)
    tokens_by_word = assign_tokens_to_words(text, tokenizer, spans)

    # 2) compute word start times (delayed from END of word) and UNK times (one frame before)
    word_start_times = [seg_start + float(w["end"]) + stt_delay for w in words_ts]
    unk_times = [t - frame_hop_s for t in word_start_times]

    out = []
    cur_t = seg_start

    def emit(token_id: int, token_str: str, t: float):
        out.append({"token_id": token_id, "token": token_str, "start": t, "end": t + frame_hop_s})

    # 3) iterate words, placing pads → UNK → word tokens → pads to next word
    skip_next_unk = False
    i = 0

    while i < len(words):
        toks_this = tokens_by_word[i]

        # Only place UNK if not skipped due to previous spillover
        if not skip_next_unk:
            # pads up to this UNK
            while cur_t + 1e-9 < unk_times[i]:
                emit(pad_id, tokenizer.convert_ids_to_tokens([pad_id])[0], cur_t)
                cur_t += frame_hop_s

            # Place UNK token
            cur_t = max(cur_t, unk_times[i])
            emit(unk_id, tokenizer.convert_ids_to_tokens([unk_id])[0], cur_t)
            cur_t += frame_hop_s

        # Reset skip flag
        skip_next_unk = False

        # Check for spillover into next word
        if i + 1 < len(words):
            word_end_time = word_start_times[i] + len(toks_this) * frame_hop_s
            next_word_start = word_start_times[i+1]

            if word_end_time > next_word_start:
                # Spillover detected - skip next UNK
                skip_next_unk = True

                if spillover == "truncate":
                    # Truncate current word to fit
                    max_tokens = max(0, int((next_word_start - word_start_times[i]) / frame_hop_s))
                    toks_this = toks_this[:max_tokens]
                else:  # "shift"
                    # Shift all remaining words
                    shift_amount = word_end_time - next_word_start
                    for k in range(i+1, len(word_start_times)):
                        word_start_times[k] += shift_amount
                        unk_times[k] = word_start_times[k] - frame_hop_s

        # Place word tokens
        cur_t = max(cur_t, word_start_times[i])
        for tid, tok in toks_this:
            emit(tid, tok, cur_t)
            cur_t += frame_hop_s

        i += 1

    # optional tail padding
    if pad_to_segment_end and seg_end is not None:
        while cur_t + 1e-9 < seg_end:
            emit(pad_id, tokenizer.convert_ids_to_tokens([pad_id])[0], cur_t)
            cur_t += frame_hop_s

    return out