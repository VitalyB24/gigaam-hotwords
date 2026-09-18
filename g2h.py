#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""g2h: Russian speech to text with GigaAM v3 (CTC) and an optional term dictionary.

Pipeline:
  1. Voice activity detection of faster-whisper (Silero VAD, pause 500 ms) cuts the recording into speech chunks
     of at most 25 s, merged exactly as faster-whisper's BatchedInferencePipeline merges them; a chunk is its speech
     parts glued together, its timestamps map back to the recording. The detector runs in the main Python that has
     faster-whisper installed (a subprocess), so the chunking is the one a Whisper transcription of the same file uses.
  2. Every chunk goes through the GigaAM v3_e2e_ctc encoder on the CPU.
  3. Decoding. Without a dictionary: greedy CTC, exactly GigaAM's own. With one: a CTC prefix beam search whose
     hypothesis score is its best alignment (the maximum over alignments, not the sum), beam 16; on every frame the
     blank and the tokens within 10 of the frame's best log-probability are tried. A bonus of w is added for every
     token of a dictionary term that starts a word; a hypothesis that leaves the term before its end loses the bonus,
     one that completes it keeps it, and an ending of up to 3 letters may follow ("НДС" -> "НДСы"). With w = 0 the
     search returns the greedy result at any beam width.

Usage (python: the interpreter of the GigaAM virtual environment):
  python g2h.py --audio rec.wav --out rec.txt [--dict terms.txt] [--w 3] [--reserve 4] [--title "..."] [--log run.log]
  python g2h.py --audio rec.wav --out rec.txt --dict terms.txt --keep-logprobs rec.npz    keep the CTC output
  python g2h.py --from-logprobs rec.npz --out rec2.txt --dict terms2.txt                 re-decode, no model, seconds
  python g2h.py --audio rec.wav --out rec.txt --dict terms.txt --w 1 --reserve 0          the defaults before 18.09.2026
  python g2h.py --audio rec.wav --out rec.txt --words rec.words.json                     word times and confidences
  python g2h.py --dict terms.txt --dict-check                                            terms -> tokens, rejections
  python g2h.py --audio rec.wav --out probe.txt --limit-sec 120                          first two minutes only

Input: WAV, 16 kHz, mono, 16-bit (ffmpeg -i in -ac 1 -ar 16000 -vn -acodec pcm_s16le out.wav).
Output: a two-line "#" header, then one line per chunk "[H:MM:SS → H:MM:SS] text"; an .srt next to it (--srt to
place it elsewhere); <out>.partial.txt is written chunk by chunk while running and removed at the end.
Log (stdout, and --log if given): "=== START g2h", "=== DONE g2h", "=== FAILED g2h".
Exit codes: 0 done; 1 no audio (no file, an empty one, or no speech in it); 2 failure.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = 16000
CHUNK_SEC = 25
VAD_PARAMETERS = {'min_silence_duration_ms': 500}
MODEL_NAME = 'v3_e2e_ctc'
TOKENIZER_FILE = MODEL_NAME + '_tokenizer.model'
BEAM, PRUNE, NBEST, ENDING = 16, 10.0, 5, 3     # beam width, frame pruning, token splits per form, ending letters
DEFAULT_W = 3
DEFAULT_RESERVE = 4
DEFAULT_THREADS = 16
EXIT_OK, EXIT_NO_AUDIO, EXIT_FAILED = 0, 1, 2
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
NEG = float('-inf')
LOG_PATH = None


class NoAudio(Exception):
    """Nothing to transcribe: no file, an empty file, or no speech in it (exit code 1)."""


def log(msg):
    line = '[%s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    try:
        print(line, flush=True)
    except (OSError, ValueError):    # a detached process may have no usable console
        pass
    if LOG_PATH:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def fmt(sec):
    return str(timedelta(seconds=int(sec or 0)))


def srt_ts(sec):
    ms = int(round(sec * 1000))
    return '%02d:%02d:%02d,%03d' % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, ms % 1000)


def keep_awake():
    """ES_CONTINUOUS | ES_SYSTEM_REQUIRED: Windows does not go to sleep while the process lives."""
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except (AttributeError, OSError):
            pass


def peak_ram_mb():
    """Peak working set of the process, MB (Windows only; None elsewhere)."""
    if os.name != 'nt':
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t), ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                    ('PeakPagefileUsage', ctypes.c_size_t)]
    try:
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return round(c.PeakWorkingSetSize / 2**20)
    except (AttributeError, OSError):
        pass
    return None


# ---------------------------------------------------------------- chunking (voice activity detection)

def audio_digest(audio):
    return hashlib.sha1(audio.tobytes()).hexdigest()


def make_chunks(audio_path, limit_sec, out_json):
    """Main-Python mode: VAD of faster-whisper -> chunks of at most CHUNK_SEC s, merged as faster-whisper's
    BatchedInferencePipeline merges them. The grouping loop below repeats the merge condition of
    faster_whisper.vad.collect_chunks (faster-whisper, MIT License) to know which speech parts form each chunk, and
    the result is checked against collect_chunks itself."""
    import faster_whisper
    from faster_whisper import decode_audio
    from faster_whisper.vad import SpeechTimestampsMap, VadOptions, collect_chunks, get_speech_timestamps
    t0 = time.time()
    audio = decode_audio(str(audio_path), sampling_rate=SR)
    if limit_sec:
        audio = audio[:int(limit_sec * SR)]
    speech = get_speech_timestamps(audio, VadOptions(**VAD_PARAMETERS, max_speech_duration_s=CHUNK_SEC))
    groups, cur, cur_len = [], [], 0
    for sp in speech:
        n = sp['end'] - sp['start']
        if cur_len + n > CHUNK_SEC * SR:
            groups.append(cur)
            cur, cur_len = [sp], n
        else:
            cur.append(sp)
            cur_len += n
    groups.append(cur)
    audio_chunks, meta = collect_chunks(audio, speech, sampling_rate=SR, max_duration=CHUNK_SEC)
    lengths = [sum(sp['end'] - sp['start'] for sp in g) for g in groups]
    if lengths != [len(a) for a in audio_chunks]:
        raise RuntimeError('chunk merging differs from faster_whisper.vad.collect_chunks')
    ts_map = SpeechTimestampsMap(speech, SR)
    chunks = []
    for g, m, n in zip(groups, meta, lengths, strict=True):
        if not g:
            continue
        start = ts_map.get_original_time(m['offset'])
        end = ts_map.get_original_time(m['offset'] + n / SR, is_end=True)
        chunks.append({'start': start, 'end': end, 'samples': n, 'parts': [[sp['start'], sp['end']] for sp in g]})
    data = {'faster_whisper': faster_whisper.__version__, 'vad_parameters': dict(VAD_PARAMETERS, max_speech_duration_s=CHUNK_SEC),
            'limit_sec': limit_sec, 'audio_samples': len(audio), 'audio_sha1': audio_digest(audio),
            'speech_segments': len(speech), 'vad_sec': round(time.time() - t0, 1), 'chunks': chunks}
    Path(out_json).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    longest = max((c['samples'] for c in chunks), default=0) / SR
    print('chunks %d (speech segments %d), longest %.2f s, VAD %.0f s' % (len(chunks), len(speech), longest, data['vad_sec']))


def default_vad_python():
    """The main Python behind the running virtual environment (it has faster-whisper); outside a venv — this one."""
    base = Path(sys.base_prefix)
    for cand in (base / 'python.exe', base / 'bin' / 'python3', base / 'bin' / 'python'):
        if cand.is_file():
            return cand
    return Path(sys.executable)


def vad_chunks(audio_path, limit_sec, vad_python):
    """Runs make_chunks under the main Python and returns its result."""
    fd, tmp = tempfile.mkstemp(prefix='g2h_vad_', suffix='.json')
    os.close(fd)
    try:
        env = {k: v for k, v in os.environ.items() if k.upper() not in ('__PYVENV_LAUNCHER__', 'VIRTUAL_ENV', 'PYTHONHOME', 'PYTHONPATH')}
        env['PYTHONIOENCODING'] = 'utf-8'
        cmd = [str(vad_python), str(Path(__file__).resolve()), '--vad-only', tmp, '--audio', str(audio_path)]
        if limit_sec:
            cmd += ['--limit-sec', repr(limit_sec)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env,
                           creationflags=NO_WINDOW)
        if r.returncode != 0 or not Path(tmp).stat().st_size:
            raise RuntimeError('VAD under %s failed (code %s):\n%s' % (vad_python, r.returncode, (r.stdout + r.stderr)[-2000:]))
        data = json.loads(Path(tmp).read_text(encoding='utf-8'))
        log('    VAD: %s (faster-whisper %s, %s)' % (r.stdout.strip().splitlines()[-1], data['faster_whisper'], vad_python))
        return data
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_wav(path, limit_sec=0):
    import numpy as np
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise RuntimeError('%s is not a 16 kHz mono 16-bit WAV (ffmpeg -i <in> -ac 1 -ar 16000 -vn -acodec pcm_s16le <out>.wav)' % path)
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return audio[:int(limit_sec * SR)] if limit_sec else audio


# ---------------------------------------------------------------- dictionary -> tokens

def load_terms(path):
    """One term per line, spelled as it should appear in the transcript; '#' starts a comment."""
    return [t for t in (line.split('#')[0].strip() for line in Path(path).read_text(encoding='utf-8').splitlines()) if t]


def is_abbrev(term):
    return term == term.upper() and any(c.isalpha() for c in term)


def term_forms(term):
    """Forms of a term: as written; a term not in capitals also with a lower-case and an upper-case first letter."""
    if is_abbrev(term):
        return [term]
    return list(dict.fromkeys([term, term[0].lower() + term[1:], term[0].upper() + term[1:]]))


class Lexicon:
    """Prefix tree of the token sequences of all term forms (up to NBEST best SentencePiece splits per form) plus the
    token properties the bonus needs: does a token start a word, how many letters it carries."""

    def __init__(self, sp, terms):
        self.sp = sp
        n = sp.get_piece_size()
        self.blank = n
        pieces = [sp.id_to_piece(i) for i in range(n)]
        special = [sp.is_unknown(i) or sp.is_control(i) for i in range(n)]
        self.ws = [p.startswith('▁') and not s for p, s in zip(pieces, special, strict=True)] + [False]
        self.letters = [0 if s else sum(ch.isalpha() for ch in p.replace('▁', '')) for p, s in zip(pieces, special, strict=True)] + [0]
        self.children, self.terminal = [{}], [None]
        self.forms, self.rejected = [], []
        for term in terms:
            for form in term_forms(term):
                seqs = []
                for ids in sp.nbest_encode_as_ids(form, NBEST):
                    why = None
                    if any(special[i] for i in ids):
                        why = 'a special or unknown token'
                    elif sp.decode_ids(ids) != form:
                        why = 'decodes back as "%s"' % sp.decode_ids(ids)
                    elif not self.ws[ids[0]]:
                        why = 'the first token does not start a word'
                    if why:
                        self.rejected.append((term, form, [pieces[i] for i in ids], why))
                        continue
                    seqs.append(ids)
                    self.add(ids, form)
                self.forms.append((term, form, [[pieces[i] for i in ids] for ids in seqs]))

    def add(self, ids, form):
        node = 0
        for t in ids:
            nxt = self.children[node].get(t)
            if nxt is None:
                nxt = len(self.children)
                self.children.append({})
                self.terminal.append(None)
                self.children[node][t] = nxt
            node = nxt
        self.terminal[node] = form

    def sequences(self):
        return sum(len(s) for _, _, s in self.forms)


# ---------------------------------------------------------------- decoders

# Bonus state of a hypothesis: (mode, tree node, tokens walked, tokens up to the last form end, letters after the form
# end, tokens already secured). Mode 0: outside a form; 1: walking a form from a word start; 2: the form is complete
# and the word goes on with an ending (at most ENDING letters).
INIT = (0, -1, 0, 0, 0, 0)


def step(lex, st, c):
    """Bonus state after token c: +w for every token of a form walked from a word start; leaving a form before its end
    drops what the form collected, completing it keeps it; an ending of up to ENDING letters may follow."""
    mode, node, depth, lcd, tl, fin = st
    ws, lt = lex.ws[c], lex.letters[c]
    if mode == 1:
        nxt = lex.children[node].get(c)
        if nxt is not None:
            if lex.terminal[nxt] is not None:
                return (1, nxt, depth + 1, depth + 1, 0, fin)
            return (1, nxt, depth + 1, lcd, tl + lt, fin)
        if lcd and not ws and tl + lt <= ENDING:
            return (2, -1, 0, lcd, tl + lt, fin)
        if lcd and ws and tl <= ENDING:
            fin += lcd
    elif mode == 2:
        if not ws:
            return (2, -1, 0, lcd, tl + lt, fin) if tl + lt <= ENDING else (0, -1, 0, 0, 0, fin)
        fin += lcd
    if ws:
        nxt = lex.children[0].get(c)
        if nxt is not None:
            return (1, nxt, 1, 1 if lex.terminal[nxt] is not None else 0, 0, fin)
    return (0, -1, 0, 0, 0, fin)


def live_bonus(st):
    """Bonus tokens of a hypothesis during the search: secured ones plus the form in progress (or a complete one
    followed by an ending)."""
    mode, _, depth, lcd, _, fin = st
    return fin + (depth if mode == 1 else lcd if mode == 2 else 0)


def final_bonus(st):
    """Bonus tokens at the end of a chunk: an unfinished form is dropped, a complete one (ending <= ENDING letters) stays."""
    mode, _, depth, lcd, tl, fin = st
    if mode == 1:
        return fin + (lcd if lcd and tl <= ENDING else 0)
    if mode == 2:
        return fin + lcd
    return fin


def beam_decode(lex, logp, w, beam=BEAM, prune=PRUNE, reserve=0):
    """CTC prefix beam search with the term bonus. The score of a hypothesis (a prefix) is its best alignment — the
    maximum over alignments, not the sum; with beam 1 and w = 0 this is exactly greedy decoding. logp: [T, C] log-
    probabilities, the blank is the last class.

    reserve: beam slots kept for the best hypotheses ranked without the bonus of an unfinished term. Without them a
    hypothesis that has spelled the beginning of a term keeps that bonus while it waits on blanks, and hypotheses like it
    can push the plain continuation of the speech out of the beam; when the term never completes, the words after it are
    lost. 0 — the search as before (every slot ranked with the bonus)."""
    import numpy as np
    T, C = logp.shape
    blank = C - 1
    parent, token, state, kids = [-1], [-1], [INIT], [{}]
    beams = [(0, 0.0, NEG)]
    thr = logp.max(axis=1) - prune
    for t in range(T):
        lp = logp[t]
        cand = [int(c) for c in np.nonzero(lp >= thr[t])[0] if c != blank]
        lb = float(lp[blank])
        nxt = {}
        for pid, pb, pnb in beams:
            ptot = pb if pb > pnb else pnb
            e = nxt.get(pid)
            if e is None:
                nxt[pid] = [ptot + lb, NEG]
            elif ptot + lb > e[0]:
                e[0] = ptot + lb
            last = token[pid]
            for c in cand:
                p = float(lp[c])
                if c == last:
                    e = nxt[pid]
                    if pnb + p > e[1]:
                        e[1] = pnb + p
                    val = pb + p
                else:
                    val = ptot + p
                cid = kids[pid].get(c)
                if cid is None:
                    cid = len(parent)
                    parent.append(pid)
                    token.append(c)
                    state.append(step(lex, state[pid], c))
                    kids.append({})
                    kids[pid][c] = cid
                e = nxt.get(cid)
                if e is None:
                    nxt[cid] = [NEG, val]
                elif val > e[1]:
                    e[1] = val
        ranked = sorted(nxt.items(), key=lambda kv: -(max(kv[1]) + w * live_bonus(state[kv[0]])))
        if reserve and w:
            keep = sorted(nxt.items(), key=lambda kv: -max(kv[1]))[:min(reserve, beam)]
            seen = {pid for pid, _ in keep}
            for kv in ranked:
                if len(keep) >= beam:
                    break
                if kv[0] not in seen:
                    keep.append(kv)
                    seen.add(kv[0])
            ranked = keep
        beams = [(pid, v[0], v[1]) for pid, v in ranked[:beam]]
    best = max(beams, key=lambda b: max(b[1], b[2]) + w * final_bonus(state[b[0]]))[0]
    ids = []
    while best > 0:
        ids.append(token[best])
        best = parent[best]
    return ids[::-1]


def greedy_ids(logp):
    """Greedy CTC decoding, as GigaAM's CTCGreedyDecoding: argmax, repeats collapsed, blanks dropped."""
    lab = logp.argmax(axis=1)
    blank = logp.shape[1] - 1
    return [int(c) for i, c in enumerate(lab) if c != blank and (i == 0 or c != lab[i - 1])]


class Decoder:
    def __init__(self, sp, terms=None, w=DEFAULT_W, dict_name='', reserve=DEFAULT_RESERVE):
        self.sp, self.w, self.reserve = sp, w, reserve
        self.lex = Lexicon(sp, terms) if terms else None
        if self.lex:
            self.desc = ('CTC prefix beam search (best alignment), beam %d, prune -%g%s; dictionary %s: %d terms, %d token '
                         'sequences; w = %g' % (BEAM, PRUNE, ', reserve %d' % reserve if reserve else '', dict_name,
                                                len(terms), self.lex.sequences(), w))
        else:
            self.desc = 'greedy CTC, no dictionary'

    def ids(self, logp):
        return beam_decode(self.lex, logp, self.w, reserve=self.reserve) if self.lex else greedy_ids(logp)

    def text(self, logp):
        return self.sp.decode_ids(self.ids(logp)).strip()


# ---------------------------------------------------------------- word times and confidence

def ctc_align(logp, ids):
    """Forced CTC alignment (the best path) of a decoded token sequence to its chunk's log-probabilities: for every
    token its first and last frame and its best log-probability on them. None if the sequence does not fit the frames."""
    import numpy as np
    T, C = logp.shape
    if not ids:
        return []
    blank = C - 1
    ext = np.array([blank] + [x for c in ids for x in (c, blank)])
    S = len(ext)
    if T < len(ids):
        return None
    skip = np.zeros(S, dtype=bool)                   # s may be reached from s - 2: a token differing from the one before
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    low = -1e30
    alpha = np.full(S, low)
    alpha[0], alpha[1] = logp[0, ext[0]], logp[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    emit = logp[:, ext]
    for t in range(1, T):
        prev1 = np.concatenate(([low], alpha[:-1]))
        prev2 = np.where(skip, np.concatenate(([low, low], alpha[:-2])), low)
        best = np.maximum(np.maximum(alpha, prev1), prev2)
        back[t] = np.where(best == alpha, 0, np.where(best == prev1, 1, 2))
        alpha = best + emit[t]
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= low / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, 0, -1):
        path[t] = s
        s -= int(back[t, s])
    path[0] = s
    if s > 1:
        return None
    out = []
    for k, c in enumerate(ids):
        fr = np.nonzero(path == 2 * k + 1)[0]
        if not len(fr):
            return None
        out.append((int(fr[0]), int(fr[-1]), float(logp[fr, c].max())))
    return out


def frame_clock(T, samples=None, parts=None, start=0.0, end=0.0):
    """Frame index of a chunk -> time in the recording. A chunk is its speech parts glued together, so the time goes
    through the parts; without them — linearly between the chunk's start and end."""
    if parts:
        total = samples or sum(e - s for s, e in parts)
        spans, acc = [], 0
        for s, e in parts:
            spans.append((acc, s, e))
            acc += e - s

        def clock(f):
            x = min(max(f * total / T, 0.0), float(acc))
            for a0, s, e in spans:
                if x <= a0 + (e - s):
                    return (s + x - a0) / SR
            return parts[-1][1] / SR
        return clock
    return lambda f: start + (end - start) * min(max(f / T, 0.0), 1.0)


def chunk_words(sp, logp, ids, clock):
    """(confidence of the chunk, [[word, start, end, confidence], …]); the confidence is the geometric mean of the
    tokens' best probabilities on their aligned frames. (None, []) for an empty chunk or one that does not align."""
    import numpy as np
    al = ctc_align(logp, ids)
    if not al:
        return None, []
    words = []
    for c, (f0, f1, lp) in zip(ids, al, strict=True):
        if not words or sp.id_to_piece(c).startswith('▁'):
            words.append([[c], f0, f1, [lp]])
        else:
            words[-1][0].append(c)
            words[-1][2] = f1
            words[-1][3].append(lp)
    out = [[sp.decode_ids(w_ids).strip(), round(clock(f0), 2), round(clock(f1 + 1), 2), round(float(np.exp(np.mean(lps))), 3)]
           for w_ids, f0, f1, lps in words]
    return round(float(np.exp(np.mean([lp for _, _, lp in al]))), 3), [w for w in out if w[0]]


def write_words(path, head, chunks):
    Path(path).write_text(json.dumps(dict(head, chunks=chunks), ensure_ascii=False, separators=(',', ':')),
                          encoding='utf-8')
    log('    word times: %s (%d chunks, %d words)' % (path, len(chunks), sum(len(c['words']) for c in chunks)))


def load_tokenizer(models):
    import sentencepiece
    path = Path(models) / TOKENIZER_FILE
    if not path.is_file():
        raise RuntimeError('no tokenizer %s: run once with --audio, the model and its tokenizer are downloaded then' % path)
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(str(path))
    return sp


# ---------------------------------------------------------------- output

def write_outputs(out, srt, title, info, segs):
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# %s\n# %s\n\n' % (title, info))
        for a, b, text in segs:
            f.write('[%s → %s] %s\n' % (fmt(a), fmt(b), text))
    with open(srt, 'w', encoding='utf-8', newline='\n') as f:
        k = 0
        for a, b, text in segs:
            if text:
                k += 1
                f.write('%d\n%s --> %s\n%s\n\n' % (k, srt_ts(a), srt_ts(b), text))


def save_logprobs(path, lps, segs, meta):
    import numpy as np
    arrays = {'c%04d' % i: lp for i, lp in enumerate(lps)}
    np.savez(path, starts=np.array([a for a, _, _ in segs], dtype=np.float64),
             ends=np.array([b for _, b, _ in segs], dtype=np.float64), meta=np.array(json.dumps(meta, ensure_ascii=False)),
             **arrays)
    log('    CTC log-probabilities saved: %s (%.0f MB)' % (path, Path(path).stat().st_size / 2**20))


def load_logprobs(path):
    """Chunks, their times and the metadata of a file saved by --keep-logprobs. A file without chunk times (the chunk
    arrays c0000... only) takes them from chunks_25s.json in the same folder."""
    import numpy as np
    z = np.load(path)
    keys = sorted(k for k in z.files if re.fullmatch(r'c\d{4}', k))
    lps = [z[k] for k in keys]
    if 'starts' in z.files:
        starts, ends = [float(x) for x in z['starts']], [float(x) for x in z['ends']]
        meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
    else:
        side = Path(path).with_name('chunks_25s.json')
        if not side.is_file():
            raise RuntimeError('%s has no chunk times and there is no %s next to it' % (path, side.name))
        chunks = json.loads(side.read_text(encoding='utf-8'))['chunks']
        starts, ends = [c['start'] for c in chunks], [c['end'] for c in chunks]
        meta = {'chunks_from': side.name}
        if all('parts' in c and 'samples' in c for c in chunks):
            meta.update(samples=[c['samples'] for c in chunks], parts=[c['parts'] for c in chunks])
    if len(starts) != len(lps):
        raise RuntimeError('%s: %d chunks, but %d chunk times' % (path, len(lps), len(starts)))
    return lps, starts, ends, meta


# ---------------------------------------------------------------- runs

def partial_path(out):
    return Path(out).with_name(Path(out).stem + '.partial.txt')


def run_audio(a, terms):
    """Full run: VAD -> GigaAM encoder -> decoder, chunk by chunk, with a partial file."""
    import numpy as np
    if not a.audio or not a.audio.is_file() or a.audio.stat().st_size == 0:
        raise NoAudio('no audio file: %s' % a.audio)
    t_start = time.time()
    data = vad_chunks(a.audio, a.limit_sec, a.vad_python or default_vad_python())
    chunks = data['chunks']
    if not chunks:
        raise NoAudio('no speech found in %s' % a.audio)
    audio = read_wav(a.audio, a.limit_sec)
    if len(audio) != data['audio_samples'] or audio_digest(audio) != data['audio_sha1']:
        raise RuntimeError('the audio read here differs from the audio the VAD read')
    log('    audio %s, chunks %d, longest %.2f s' % (fmt(len(audio) / SR), len(chunks), max(c['samples'] for c in chunks) / SR))
    import torch
    import gigaam
    from gigaam.model import LONGFORM_THRESHOLD
    torch.set_num_threads(a.threads)
    t_load = time.time()
    model = gigaam.load_model(MODEL_NAME, fp16_encoder=False, use_flash=False, device='cpu', download_root=str(a.models))
    load_sec = time.time() - t_load
    sp = load_tokenizer(a.models)
    dec = Decoder(sp, terms, a.w, a.dict.name if a.dict else '', a.reserve)
    log('    GigaAM %s loaded in %.0f s (%.0fM parameters), torch threads %d; decoder: %s' % (
        MODEL_NAME, load_sec, sum(p.numel() for p in model.parameters()) / 1e6, torch.get_num_threads(), dec.desc))
    segs, lps, empty, mism, wchunks = [], [], 0, [], []
    t0 = last = time.time()
    part = partial_path(a.out)
    with open(part, 'w', encoding='utf-8', newline='\n') as pf:
        pf.write('# PARTIAL transcript, written while g2h runs: %s\n\n' % a.audio.name)
        for i, ch in enumerate(chunks):
            wav_np = np.concatenate([audio[s:e] for s, e in ch['parts']])
            with torch.inference_mode():
                wav = torch.from_numpy(wav_np).to(model._device).to(model._dtype).unsqueeze(0)
                length = torch.full([1], wav.shape[-1], device=model._device)
                if length.item() > LONGFORM_THRESHOLD:
                    raise ValueError('chunk %d is longer than 25 s: %d samples' % (i, length.item()))
                encoded, encoded_len = model.forward(wav, length)
                lp = model.head(encoder_output=encoded)[0, :int(encoded_len[0])].float().cpu().numpy()
                own = model._decode(encoded, encoded_len, length)[0][0].strip()
            if sp.decode_ids(greedy_ids(lp)).strip() != own:     # greedy over the log-probabilities = GigaAM's decoding
                mism.append(i)
            ids = dec.ids(lp)
            text = sp.decode_ids(ids).strip()
            if a.words:
                conf, words = chunk_words(sp, lp, ids, frame_clock(len(lp), ch['samples'], ch['parts']))
                wchunks.append({'start': ch['start'], 'end': ch['end'], 'conf': conf, 'words': words})
            if a.keep_logprobs:
                lps.append(lp)
            empty += not text
            segs.append((ch['start'], ch['end'], text))
            pf.write('[%s → %s] %s\n' % (fmt(ch['start']), fmt(ch['end']), text))
            pf.flush()
            if time.time() - last > 120:
                last = time.time()
                log('    %s of %s (chunk %d of %d)' % (fmt(ch['end']), fmt(len(audio) / SR), i + 1, len(chunks)))
    recog = time.time() - t0
    total, peak = time.time() - t_start, peak_ram_mb()
    if mism:
        log('    ! greedy decoding of the log-probabilities differs from GigaAM on chunks %s' % mism[:20])
    info = ('Generated %s by g2h: GigaAM %s, CPU, threads %d; VAD faster-whisper %s, pause %d ms, chunks up to %d s%s; '
            'decoder: %s; chunks %d, empty %d; time %s (VAD %.0f s, model load %.0f s, recognition %s); peak RAM %s MB' % (
                datetime.now().isoformat(timespec='seconds'), MODEL_NAME, a.threads, data['faster_whisper'],
                VAD_PARAMETERS['min_silence_duration_ms'], CHUNK_SEC, ', first %g s only' % a.limit_sec if a.limit_sec else '',
                dec.desc, len(segs), empty, fmt(total), data['vad_sec'], load_sec, fmt(recog), peak))
    write_outputs(a.out, a.srt, a.title or 'Transcript of %s' % a.audio.name, info, segs)
    if a.words:
        write_words(a.words, {'audio': a.audio.name, 'model': MODEL_NAME, 'decoder': dec.desc}, wchunks)
    if a.keep_logprobs:
        save_logprobs(a.keep_logprobs, lps, segs, {'model': MODEL_NAME, 'audio': a.audio.name, 'audio_sha1': data['audio_sha1'],
                                                   'faster_whisper': data['faster_whisper'], 'vad_parameters': data['vad_parameters'],
                                                   'limit_sec': a.limit_sec, 'threads': a.threads, 'greedy_mismatch': mism,
                                                   'samples': [c['samples'] for c in chunks], 'parts': [c['parts'] for c in chunks]})
    part.unlink(missing_ok=True)
    return len(segs), empty, total


def run_logprobs(a, terms):
    """Re-decoding of saved CTC log-probabilities: no audio, no model."""
    t_start = time.time()
    lps, starts, ends, meta = load_logprobs(a.from_logprobs)
    sp = load_tokenizer(a.models)
    dec = Decoder(sp, terms, a.w, a.dict.name if a.dict else '', a.reserve)
    log('    %d chunks from %s; decoder: %s' % (len(lps), a.from_logprobs, dec.desc))
    segs, wchunks = [], []
    parts, samples = meta.get('parts'), meta.get('samples')
    if a.words and not parts:
        log('    ! no chunk parts saved with the log-probabilities: word times are spread linearly over each chunk')
    part = partial_path(a.out)
    with open(part, 'w', encoding='utf-8', newline='\n') as pf:
        pf.write('# PARTIAL transcript, written while g2h runs: %s\n\n' % Path(a.from_logprobs).name)
        for i, (s, e, lp) in enumerate(zip(starts, ends, lps, strict=True)):
            ids = dec.ids(lp)
            segs.append((s, e, sp.decode_ids(ids).strip()))
            if a.words:
                clock = frame_clock(len(lp), samples[i], parts[i]) if parts else frame_clock(len(lp), start=s, end=e)
                conf, words = chunk_words(sp, lp, ids, clock)
                wchunks.append({'start': s, 'end': e, 'conf': conf, 'words': words})
            pf.write('[%s → %s] %s\n' % (fmt(s), fmt(e), segs[-1][2]))
            pf.flush()
    empty, total = sum(1 for _, _, t in segs if not t), time.time() - t_start
    info = 'Generated %s by g2h: re-decoded from %s (GigaAM %s); decoder: %s; chunks %d, empty %d; time %s' % (
        datetime.now().isoformat(timespec='seconds'), Path(a.from_logprobs).name, meta.get('model', MODEL_NAME), dec.desc,
        len(segs), empty, fmt(total))
    write_outputs(a.out, a.srt, a.title or 'Transcript re-decoded from %s' % Path(a.from_logprobs).name, info, segs)
    if a.words:
        write_words(a.words, {'audio': meta.get('audio', Path(a.from_logprobs).name), 'model': meta.get('model', MODEL_NAME),
                              'decoder': dec.desc, 'approximate_times': not parts}, wchunks)
    part.unlink(missing_ok=True)
    return len(segs), empty, total


def dict_check(a, terms):
    """How every form of every term maps to tokens; rejected splits with the reason. Exit 2 if a form has no usable
    split at all (the bonus never applies to it)."""
    lex = Lexicon(load_tokenizer(a.models), terms)
    width = max((len(f) for _, f, _ in lex.forms), default=4)
    print('Dictionary %s: %d terms, %d forms; token sequences accepted %d, rejected %d' % (
        a.dict, len(terms), len(lex.forms), lex.sequences(), len(lex.rejected)))
    for _term, form, seqs in lex.forms:
        print('  %-*s  %s' % (width, form, ' ; '.join(' '.join(s) for s in seqs) or '— no usable split'))
    for term, form, pieces, why in lex.rejected:
        print('  rejected: %s (%s): %s — %s' % (form, term, ' '.join(pieces), why))
    unusable = [f for _, f, s in lex.forms if not s]
    if unusable:
        print('Forms the bonus never applies to: %s' % ', '.join(unusable))
    return EXIT_FAILED if unusable else EXIT_OK


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--audio', type=Path, help='WAV, 16 kHz mono 16-bit')
    ap.add_argument('--out', type=Path, help='transcript to write')
    ap.add_argument('--srt', type=Path, help='subtitles to write (default: next to --out)')
    ap.add_argument('--title', help='first header line of the transcript')
    ap.add_argument('--dict', type=Path, help='term dictionary: one term per line (without it: plain greedy GigaAM)')
    ap.add_argument('--w', type=float, default=DEFAULT_W, help='bonus per term token (default %(default)g)')
    ap.add_argument('--reserve', type=int, default=DEFAULT_RESERVE, metavar='K',
                    help='beam slots kept for hypotheses ranked without the bonus of an unfinished term (default %(default)d; 0 turns it off)')
    ap.add_argument('--words', type=Path, metavar='JSON', help='also write word times and confidences (chunks → words)')
    ap.add_argument('--threads', type=int, default=DEFAULT_THREADS, help='torch CPU threads (default %(default)d)')
    ap.add_argument('--keep-logprobs', type=Path, metavar='NPZ', help='also save the CTC log-probabilities')
    ap.add_argument('--from-logprobs', type=Path, metavar='NPZ', help='re-decode saved log-probabilities instead of --audio')
    ap.add_argument('--dict-check', action='store_true', help='show how the dictionary maps to tokens and exit')
    ap.add_argument('--limit-sec', type=float, default=0, help='transcribe the first N seconds only (a probe)')
    ap.add_argument('--log', type=Path, help='append the log to this file as well')
    ap.add_argument('--models', type=Path, default=HERE / 'models', help='GigaAM weights (default: models/ next to this script)')
    ap.add_argument('--vad-python', type=Path, help='Python with faster-whisper (default: the main Python behind this venv)')
    ap.add_argument('--vad-only', metavar='JSON', help=argparse.SUPPRESS)    # internal: the main-Python VAD step
    a = ap.parse_args(argv)
    if a.vad_only or a.dict_check:
        if a.dict_check and not a.dict:
            ap.error('--dict-check needs --dict')
        return a
    if not a.out or not (a.audio or a.from_logprobs) or (a.audio and a.from_logprobs):
        ap.error('give --out and exactly one of --audio / --from-logprobs')
    a.srt = a.srt or a.out.with_suffix('.srt')
    return a


def main(argv=None):
    global LOG_PATH
    if sys.stdout:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    a = parse_args(argv)
    if a.vad_only:
        make_chunks(a.audio, a.limit_sec, a.vad_only)
        return EXIT_OK
    terms = load_terms(a.dict) if a.dict else []
    if a.dict_check:
        return dict_check(a, terms)
    LOG_PATH = a.log
    keep_awake()
    src = a.audio if a.audio else a.from_logprobs
    log('=== START g2h: %s -> %s%s' % (src, a.out, ' (dictionary %s, %d terms, w = %g)' % (a.dict.name, len(terms), a.w) if terms else ''))
    try:
        n, empty, total = run_audio(a, terms) if a.audio else run_logprobs(a, terms)
    except NoAudio as e:
        log('=== FAILED g2h: %s' % e)
        return EXIT_NO_AUDIO
    except Exception:
        log('=== FAILED g2h\n' + traceback.format_exc())
        return EXIT_FAILED
    log('=== DONE g2h: %d chunks (empty %d) in %s -> %s' % (n, empty, fmt(total), a.out))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
