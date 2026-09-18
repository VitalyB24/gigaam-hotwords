"""Decoder and file-format tests on synthetic data: no model, no audio, no tokenizer file.

The tokenizer is a small stand-in with the interface of sentencepiece.SentencePieceProcessor that g2h uses; the
log-probabilities are built by hand, so every test knows the answer in advance.
"""
import json
import math

import numpy as np
import pytest

import g2h

LETTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


class FakeSP:
    """Pieces: '▁' alone, every letter with and without the word-start mark '▁'; id 0 is <unk>.
    Encoding is greedy longest match, one split per text."""

    def __init__(self):
        self.pieces = ['<unk>', '▁'] + ['▁' + ch for ch in LETTERS] + list(LETTERS)
        self.index = {p: i for i, p in enumerate(self.pieces)}

    def get_piece_size(self):
        return len(self.pieces)

    def id_to_piece(self, i):
        return self.pieces[i]

    def piece_to_id(self, p):
        return self.index.get(p, 0)

    def is_unknown(self, i):
        return i == 0

    def is_control(self, i):
        return False

    def decode_ids(self, ids):
        return ''.join(self.pieces[i] for i in ids).replace('▁', ' ').lstrip(' ')

    def nbest_encode_as_ids(self, text, n):
        s, ids = '▁' + text.replace(' ', '▁'), []
        while s:
            best = max((p for p in self.pieces[1:] if s.startswith(p)), key=len, default=None)
            ids.append(self.index[best] if best else 0)
            s = s[len(best):] if best else s[1:]
        return [ids]


SP = FakeSP()
BLANK = SP.get_piece_size()          # the blank is the class after the last piece
C = BLANK + 1


def ids_of(*pieces):
    return [SP.piece_to_id(p) for p in pieces]


def frames(steps):
    """A frame per step with the given token probabilities, a blank frame after each; rows are log-probabilities."""
    rows = []
    for probs in steps:
        f = np.full(C, -40.0)
        for tok, p in probs.items():
            f[tok] = math.log(p)
        g = np.full(C, -40.0)
        g[BLANK] = 0.0
        rows += [f, g]
    lp = np.array(rows)
    return lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))


def random_logprobs(seed, t=40):
    x = np.random.default_rng(seed).normal(size=(t, C)) * 3
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def final_state(lex, ids):
    st = g2h.INIT
    for t in ids:
        st = g2h.step(lex, st, t)
    return st


@pytest.mark.parametrize('seed', range(12))
def test_zero_bonus_is_greedy_at_any_beam(seed):
    lex = g2h.Lexicon(SP, ['ВМС', 'надбавка импортёра'])
    lp = random_logprobs(seed)
    greedy = g2h.greedy_ids(lp)
    assert g2h.beam_decode(lex, lp, 0) == greedy
    assert g2h.beam_decode(lex, lp, 0, beam=1) == greedy


def test_term_that_loses_by_one_wins_with_bonus():
    lex = g2h.Lexicon(SP, ['ВМС'])
    term, alt = ids_of('▁В', 'М', 'С'), SP.piece_to_id('е')
    p_lo, p_hi = 1 / (1 + math.e), math.e / (1 + math.e)        # ln(p_hi) - ln(p_lo) = 1
    lp = frames([{term[0]: 1.0}, {term[1]: 1.0}, {term[2]: p_lo, alt: p_hi}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 0)) == 'ВМе'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2)) == 'ВМС'


def test_abandoned_term_keeps_no_bonus():
    lex = g2h.Lexicon(SP, ['ВМС'])
    started = ids_of('▁В', 'М', 'е')                              # the term begins and leaves before its end
    other = ids_of('▁а', 'б', 'в')                                # a slightly likelier word at every step
    lp = frames([{a: 0.45, b: 0.55} for a, b in zip(started, other, strict=True)])
    assert g2h.beam_decode(lex, lp, 5) == other
    assert g2h.final_bonus(final_state(lex, started)) == 0


@pytest.mark.parametrize('ending, kept', [('', True), ('ом', True), ('ами', True), ('ского', False)])
def test_ending_after_a_term(ending, kept):
    lex = g2h.Lexicon(SP, ['ВМС'])
    term = ids_of('▁В', 'М', 'С')
    bonus = g2h.final_bonus(final_state(lex, term + ids_of(*ending)))
    assert bonus == (len(term) if kept else 0)


def test_forms_of_a_term():
    assert g2h.term_forms('ВМС') == ['ВМС']
    assert g2h.term_forms('наименование') == ['наименование', 'Наименование']
    assert g2h.term_forms('Озон') == ['Озон', 'озон']


def test_form_with_an_unknown_character_is_rejected():
    lex = g2h.Lexicon(SP, ['ВМ1', 'ВМС'])
    assert [(f, why) for _, f, _, why in lex.rejected] == [('ВМ1', 'a special or unknown token')]
    assert [f for _, f, s in lex.forms if not s] == ['ВМ1']
    assert lex.sequences() == 1


def test_decoder_without_dictionary_is_greedy():
    lp = random_logprobs(7)
    assert g2h.Decoder(SP).text(lp) == SP.decode_ids(g2h.greedy_ids(lp)).strip()
    assert g2h.Decoder(SP, ['ВМС'], w=0).text(lp) == g2h.Decoder(SP).text(lp)


def test_load_terms_skips_comments_and_blank_lines(tmp_path):
    p = tmp_path / 'terms.txt'
    p.write_text('# header\n\nВМС\nТН ВЭД   # a comment\n  \n', encoding='utf-8')
    assert g2h.load_terms(p) == ['ВМС', 'ТН ВЭД']


def test_time_formats():
    assert g2h.fmt(3661.9) == '1:01:01'
    assert g2h.srt_ts(3661.5) == '01:01:01,500'
    assert g2h.srt_ts(0.0014) == '00:00:00,001'


def test_outputs_have_the_project_format(tmp_path):
    out, srt = tmp_path / 'rec.txt', tmp_path / 'rec.srt'
    g2h.write_outputs(out, srt, 'Title', 'Info', [(0.0, 24.5, 'Первый кусок.'), (24.5, 30.0, ''), (31.2, 50.0, 'Третий.')])
    assert out.read_text(encoding='utf-8').splitlines() == [
        '# Title', '# Info', '', '[0:00:00 → 0:00:24] Первый кусок.', '[0:00:24 → 0:00:30] ', '[0:00:31 → 0:00:50] Третий.']
    assert srt.read_text(encoding='utf-8').split('\n\n')[:2] == [
        '1\n00:00:00,000 --> 00:00:24,500\nПервый кусок.', '2\n00:00:31,200 --> 00:00:50,000\nТретий.']


def test_logprobs_round_trip(tmp_path):
    lps = [random_logprobs(1, 5).astype(np.float32), random_logprobs(2, 7).astype(np.float32)]
    segs = [(0.0, 10.5, 'a'), (12.0, 20.0, 'b')]
    path = tmp_path / 'rec.npz'
    g2h.save_logprobs(path, lps, segs, {'model': 'm'})
    got, starts, ends, meta = g2h.load_logprobs(path)
    assert [x.tolist() for x in got] == [x.tolist() for x in lps]
    assert (starts, ends, meta) == ([0.0, 12.0], [10.5, 20.0], {'model': 'm'})


def test_logprobs_without_times_take_them_from_the_chunk_file(tmp_path):
    np.savez(tmp_path / 'old.npz', c0000=random_logprobs(3, 4), c0001=random_logprobs(4, 4))
    (tmp_path / 'chunks_25s.json').write_text(json.dumps({'chunks': [{'start': 1.0, 'end': 2.0}, {'start': 3.0, 'end': 4.5}]}),
                                              encoding='utf-8')
    _, starts, ends, _ = g2h.load_logprobs(tmp_path / 'old.npz')
    assert (starts, ends) == ([1.0, 3.0], [2.0, 4.5])


def test_reserve_keeps_the_plain_continuation():
    # two terms start after "▁В"; with beam 2 their unfinished prefixes, both holding the bonus, crowd out the plain "Ве"
    lex = g2h.Lexicon(SP, ['ВМС', 'ВНЗ'])
    v, m, n, e, a, b = ids_of('▁В', 'М', 'Н', 'е', 'а', 'б')
    lp = frames([{v: 1.0}, {m: 0.3, n: 0.3, e: 0.4}, {a: 1.0}, {b: 1.0}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 0, beam=2)) == 'Веаб'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 5, beam=2)) in ('ВМаб', 'ВНаб')
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 5, beam=2, reserve=1)) == 'Веаб'


def test_reserve_keeps_a_term_that_completes():
    lex = g2h.Lexicon(SP, ['ВМС'])
    term, alt = ids_of('▁В', 'М', 'С'), SP.piece_to_id('е')
    p_lo, p_hi = 1 / (1 + math.e), math.e / (1 + math.e)
    lp = frames([{term[0]: 1.0}, {term[1]: 1.0}, {term[2]: p_lo, alt: p_hi}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2, reserve=4)) == 'ВМС'


@pytest.mark.parametrize('seed', range(4))
def test_reserve_with_zero_bonus_is_greedy(seed):
    lex = g2h.Lexicon(SP, ['ВМС', 'надбавка импортёра'])
    lp = random_logprobs(seed)
    assert g2h.beam_decode(lex, lp, 0, reserve=8) == g2h.greedy_ids(lp)


def test_ctc_align_finds_the_frames_of_every_token():
    v, m = ids_of('▁В', 'М')
    lp = frames([{v: 1.0}, {m: 1.0}])                            # frames: В, blank, М, blank
    al = g2h.ctc_align(lp, [v, m])
    assert [(f0, f1) for f0, f1, _ in al] == [(0, 0), (2, 2)]
    assert all(p > -1e-6 for _, _, p in al)
    assert g2h.ctc_align(lp, []) == []
    assert g2h.ctc_align(lp[:1], [v, m]) is None                 # two tokens do not fit one frame


def test_chunk_words_times_and_confidence():
    v, m, a = ids_of('▁В', 'М', '▁а')
    lp = frames([{v: 1.0}, {m: 1.0}, {a: 1.0}])                  # 6 frames
    conf, words = g2h.chunk_words(SP, lp, [v, m, a], g2h.frame_clock(len(lp), start=10.0, end=16.0))
    assert [w for w, _, _, _ in words] == ['ВМ', 'а']
    assert [(s, e) for _, s, e, _ in words] == [(10.0, 13.0), (14.0, 15.0)]
    assert conf == 1.0 and all(p == 1.0 for *_, p in words)


def test_frame_clock_goes_through_the_speech_parts():
    clock = g2h.frame_clock(100, 32000, [[0, 16000], [32000, 48000]])     # two 1 s parts with 1 s of silence between
    assert clock(0) == 0.0
    assert clock(25) == 0.5
    assert clock(75) == 2.5
    assert clock(100) == 3.0


def test_words_file_from_saved_logprobs(tmp_path, monkeypatch):
    monkeypatch.setattr(g2h, 'load_tokenizer', lambda models: SP)
    v, m = ids_of('▁В', 'М')
    lps = [frames([{v: 1.0}, {m: 1.0}]).astype(np.float32)]
    g2h.save_logprobs(tmp_path / 'rec.npz', lps, [(5.0, 6.0, 'ВМ')], {'model': 'm', 'samples': [16000], 'parts': [[[80000, 96000]]]})
    out, words = tmp_path / 'rec.txt', tmp_path / 'rec.words.json'
    assert g2h.main(['--from-logprobs', str(tmp_path / 'rec.npz'), '--out', str(out), '--words', str(words)]) == 0
    data = json.loads(words.read_text(encoding='utf-8'))
    assert data['approximate_times'] is False
    assert data['chunks'][0]['words'] == [['ВМ', 5.0, 5.75, 1.0]]
