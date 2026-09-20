# gigaam-hotwords

*[Русская версия](README.ru.md)*

Russian speech to text on a plain CPU with [GigaAM v3](https://github.com/salute-developers/GigaAM) and an optional
dictionary of terms. One script, `g2h.py` (GigaAM with hotwords): a long recording in (a two- or three-hour
meeting), a timestamped transcript and subtitles out, in a few minutes. The dictionary nudges the decoder towards the
spelling your documents use for abbreviations and names ("ТТН", "ТН ВЭД") without letting the acoustic model invent
words that were not said.

## Why it exists

Whisper large-v3 transcribes Russian business meetings well, but on a CPU it is slow even in batched mode (about 0.4
of the recording length), and inside a 25-second chunk it now and then skips a whole phrase. GigaAM v3 is a Russian
model whose CTC variant needs a couple of minutes for a two-hour recording. On two long meetings (2 h 23 min and
2 h 47 min), measured against batched Whisper large-v3 on the same chunks:

- 10 % more words, no 30-second windows where it lost most of the speech (Whisper: 14), and far fewer phrases missing
  that a third, independent transcription confirms (95 words against 589);
- about 3 minutes per recording on 16 threads instead of about an hour.

Plain GigaAM writes project abbreviations by ear, though. A CTC model outputs a probability for every token on every
frame, so the decoder can be told what the words of the domain look like: the dictionary mode below adds a small
bonus to the paths that spell a known term, and the acoustic evidence still decides.

## How it works

1. **Chunking.** The voice activity detector of faster-whisper (Silero VAD, pause 500 ms) cuts the recording into
   speech chunks of at most 25 s, merged exactly as faster-whisper's batched pipeline merges them. A batched
   faster-whisper transcription of the same file gets the same chunks when it runs with the same VAD settings and
   `chunk_length=25` (its own default is 30 s); then two transcripts can be compared chunk by chunk. The detector runs
   in the main Python (the one with faster-whisper); `g2h.py` calls it as a subprocess, so the GigaAM environment stays
   small.
2. **Acoustics.** Every chunk goes through the GigaAM `v3_e2e_ctc` encoder on the CPU (16 threads by default).
3. **Decoding.**
   - Without a dictionary: greedy CTC, exactly GigaAM's own decoding.
   - With a dictionary: a CTC prefix beam search. The score of a hypothesis is its best alignment (the maximum over
     alignments rather than their sum), the beam is 16 wide, and on every frame the blank and the tokens within 10 of
     the frame's best log-probability are tried; a token with letters that spells a dictionary term is tried only
     within 5, so the bonus does not write a term over frames that do not sound like it. Every term is split into
     SentencePiece tokens (up to five splits per form). A term written in lower case also gets a form with a capital
     first letter, for the start of a sentence; a term that begins with a capital (a name: "Озон"), is written in
     capitals or has a capital inside its first word (an abbreviation, a brand: "ТТН на возврат", "ePASS") is spelled
     only as written. A hypothesis gets `w` for every token of a term it spells from the start of a word; leaving a
     term before its end takes the bonus back, completing it keeps it, and an ending of up to four letters may follow
     ("НДС" → "НДСы").
   - With `w = 0` the beam search returns the greedy result at any beam width; the tests check this.

## Requirements

- A CPU; Windows or Linux. About 2.5 GB of RAM for a 2.5-hour recording.
- Python 3.12 or newer (developed on 3.13) in two roles:
  - a virtual environment for GigaAM: `torch` and `torchaudio` 2.10 CPU builds (`constraints.txt` pins what the tool
    was verified with), GigaAM installed from its repository (verified at commit `7447938`), `sentencepiece`, `numpy`;
  - the main Python with `faster-whisper` (verified with 1.2.1) for the chunking step.
- `ffmpeg` to take the sound out of a video.

The GigaAM weights (`v3_e2e_ctc`, about 450 MB with the tokenizer) are downloaded by the `gigaam` package on the first
run into `models/` next to the script, from GigaAM's own distribution server; the package checks their checksum.

## Setup

```bash
git clone https://github.com/salute-developers/GigaAM
python -m venv venv
venv/Scripts/python -m pip install -c constraints.txt torch torchaudio --index-url https://download.pytorch.org/whl/cpu
venv/Scripts/python -m pip install -c constraints.txt -e ./GigaAM
python -m pip install faster-whisper
```

On Linux the interpreter of the environment is `venv/bin/python`. The main Python is found as the one the virtual
environment was created from; another one can be given with `--vad-python`.

`pyproject.toml` lists the same runtime packages (`dependencies`, torch pins as in `constraints.txt`) so that GitHub's
dependency graph and security alerts see them. It is not an install path: `pip install .` would take torch from PyPI
instead of PyTorch's CPU index.

## Usage

```bash
ffmpeg -i meeting.webm -ac 1 -ar 16000 -vn -acodec pcm_s16le meeting.wav
venv/Scripts/python g2h.py --audio meeting.wav --out meeting.txt --dict terms.txt
```

| Option | What it does |
|---|---|
| `--audio`, `--out` | the WAV (16 kHz, mono, 16-bit) and the transcript to write |
| `--srt` | where to write the subtitles (default: next to `--out`) |
| `--title` | the first header line of the transcript |
| `--dict`, `--w` | the term dictionary and the bonus per term token (default 3); without `--dict` — plain GigaAM |
| `--reserve K` | beam slots kept for the hypotheses that are best by the acoustic score plus the bonus of the terms they have completed, not of a term still being spelled (default 4; 0 turns it off); see "The dictionary" |
| `--words JSON` | also write word times and confidences: `{"chunks": [{"start", "end", "conf", "words": [[word, start, end, conf], …]}]}` |
| `--threads` | torch CPU threads (default 16) |
| `--keep-logprobs NPZ` | also save the CTC output of every chunk (about 200 MB for 2.5 hours) |
| `--from-logprobs NPZ` | re-decode a saved CTC output instead of `--audio`: no model, about 20 s for 2.5 hours |
| `--dict-check` | show how every term maps to tokens, with the rejected splits, and exit (code 2 when a form has no usable split) |
| `--limit-sec N` | the first N seconds only — a quick probe of the setup |
| `--log FILE` | append the log to a file as well |
| `--models`, `--vad-python` | the weights folder and the Python with faster-whisper, if not the defaults |

**Output.** Two header lines starting with `#`, then one line per chunk: `[H:MM:SS → H:MM:SS] text`. Subtitles go to
an `.srt` next to it. While the script runs, every chunk is appended to a `.partial.txt` file named after `--out`
without its extension (`rec.txt` → `rec.partial.txt`), so a killed run leaves the text it had; the partial file is
removed at the end. The
log has `=== START g2h`, `=== DONE g2h` and `=== FAILED g2h` markers; a line with `!` is a warning (a dictionary form
that never gets the bonus, a log or partial file that cannot be written or removed), and the run goes on. The exit
code is 0 on success, 1 when there is nothing to transcribe (no file, an empty one, no speech), 2 on a failure; a
wrong argument or a missing output folder is refused before any work is done. On Windows the script keeps the machine
awake while it runs.

**Speed.** A 2 h 23 min recording on 16 threads: chunking 10 s, encoder 2 min, beam decoding with a 36-term
dictionary 20 s.

## The dictionary

A UTF-8 text file, one term per line, written the way it should appear in the transcript; abbreviations in capitals;
`#` starts a comment (see `terms.example.txt`). What an editor hides — a byte order mark, zero-width characters,
no-break and doubled spaces — is cleaned on reading, and a repeated term counts once. Put in only the terms that are
actually said and that plain decoding gets wrong: every term is a path the decoder is invited to take.

Two kinds of entries do harm. An ordinary word the model writes by itself: only the dictionary form gets the bonus, so
the word is rewritten into that form and its ending changes ("номенклатуры" → "номенклатура", a genitive turned into
the nominative); on three recorded meetings five such words gave no confirmed fix and most of the changed endings. A
term of two letters: its tokens are frequent sounds, and it gets written into other words — of about 20 new
occurrences of one such term 6–7 were right. Write a name with a capital ("Озон"): it is then spelled only that way;
a term in lower case also gets the capitalised form.

`w` trades terms against speech. A small bonus fixes a term the model was unsure about; a large one lets a hypothesis
that has spelled the beginning of a term push the real continuation out of the beam — if the term never completes,
the words after it are lost. Check a dictionary before relying on it:

1. transcribe once with `--keep-logprobs rec.npz`;
2. re-decode with every candidate dictionary or `w`: `--from-logprobs rec.npz --dict terms.txt --w 1` (seconds each),
   and once without `--dict`;
3. compare: the dictionary version should keep at least 99 % of the words of the plain one, and no term should appear
   where nothing was said (a run of dictionary terms in a row is the typical sign).

`--reserve K` removes that loss: K beam slots are kept for the hypotheses that are best by the acoustic score plus the
bonus they would keep if the chunk ended now — the terms they have completed, not a term still being spelled. The
plain continuation of the speech stays in the beam, and a term that never completes loses to it at the end of the
chunk; a corrected term counts in the reserve, so the reserved slots follow the corrected path and its continuation
stays in the beam too. On three recorded meetings (1 h 50 min to 2 h 47 min each, about 7 hours of speech), checked
against an independent Whisper transcript: `w = 2` without a reserve lost 9 phrases (112 words); with `--reserve 4`
no phrase was lost. At `w = 3 --reserve 4`, the default, a 33-term dictionary fixed 44 term spellings with no
confirmed worsening, and outside the terms it lost no word and distorted one against plain decoding. What remains is
7–8 false short terms per 7 hours — a three-letter abbreviation in place of a similar-sounding one — so read short
abbreviations against a second transcript where they matter. A full run of a recording (sound → chunks → encoder →
decoder) matches the re-decoding of its saved CTC output word for word; `--w 1 --reserve 0` reproduces the default
before 18.09.2026. The setting was checked with one dictionary: check yours (the three steps above and
`--dict-check`) before relying on it.

`--dict-check` shows whether every form of every term has a usable token split; a form without one never gets the
bonus. Its exit code is 2 when there is such a form and 0 otherwise; an ordinary run names these forms in its log and
goes on.

A term gets no bonus when it begins right after the first words of a longer dictionary term that is still matching:
with "ставка НДС" in the dictionary, "НДФЛ" in "ставка НДФЛ" is decoded as without a dictionary.

## Word times and confidence

`--words rec.words.json` aligns the decoded tokens back to the CTC frames (the best path) and writes, for every chunk,
its words with start and end times in the recording and a confidence — the geometric mean of the tokens' best
probabilities on their frames — plus the chunk's own confidence. A chunk is its speech parts glued together, so the
times go through those parts; files saved with `--keep-logprobs` carry them, and re-decoding keeps exact times. Next
to `chunks` the file names the `audio`, the `model` and the `decoder`; a re-decoding adds `approximate_times` — `true`
when the saved file carries no speech parts and the times are spread linearly over each chunk. A chunk without text
has `"conf": null`. Low chunk confidence marks the places worth a second look: on one meeting, 6 of 10 places where an
independent transcript heard speech the dictionary run did not were in the 6 % least confident chunks.

## Development

Run the checks before every commit — CI runs the same on every push:

```bash
ruff check .
pytest
```

The tests run on synthetic data — a stand-in tokenizer and hand-built log-probabilities — so neither the model nor
`torch` is needed; `requirements-dev.txt` lists what they use. The repository is English-only; line endings are LF in
git (`.gitattributes`).

## License and third-party components

The code in this repository is licensed under Apache-2.0 (`LICENSE`).

It is an independent project, not affiliated with or endorsed by the authors of GigaAM. Nothing of theirs is copied
into the repository: the components below are installed from their own sources, and the model weights are downloaded
by the GigaAM package from its own distribution.

| Component | Role | License |
|---|---|---|
| [GigaAM](https://github.com/salute-developers/GigaAM) (code and the `v3_e2e_ctc` weights), © 2024 GigaChat Team | acoustic model | MIT |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with the Silero VAD model it ships | speech detection and chunking; the chunk-merging rule in `make_chunks` follows `faster_whisper.vad.collect_chunks` | MIT |
| [PyTorch](https://github.com/pytorch/pytorch), [SentencePiece](https://github.com/google/sentencepiece), [NumPy](https://github.com/numpy/numpy) | runtime | BSD-3-Clause, Apache-2.0, BSD-3-Clause |

If you use GigaAM in research, its authors ask to cite *GigaAM: Efficient Self-Supervised Learner for Speech
Recognition* (Interspeech 2025, [arXiv:2506.01192](https://arxiv.org/abs/2506.01192)).

## Author

Vitali Butkevich — [LinkedIn](https://www.linkedin.com/in/vitalybutkevich)
