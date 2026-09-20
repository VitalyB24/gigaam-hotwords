"""Decoder and file-format tests on synthetic data: no model, no audio, no tokenizer file.

The tokenizer is a small stand-in with the interface of sentencepiece.SentencePieceProcessor that g2h uses; the
log-probabilities are built by hand, so every test knows the answer in advance.
"""
import io
import json
import math
import os
import subprocess
import sys
import wave

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


class SplitSP(FakeSP):
    """The stand-in with two pieces of two letters and with chosen splits for chosen texts: what the real tokenizer does
    (several splits per form, some of them unusable) and FakeSP cannot show."""

    def __init__(self, splits=None):
        super().__init__()
        for p in ('ск', 'ую'):
            self.index[p] = len(self.pieces)
            self.pieces.append(p)
        self.splits = splits or {}

    def nbest_encode_as_ids(self, text, n):
        if text in self.splits:
            return [[self.index[p] for p in split] for split in self.splits[text]][:n]
        return super().nbest_encode_as_ids(text, n)


SP = FakeSP()
BLANK = SP.get_piece_size()          # the blank is the class after the last piece
C = BLANK + 1


def ids_of(*pieces):
    return [SP.piece_to_id(p) for p in pieces]


def toks(text):
    return SP.nbest_encode_as_ids(text, 1)[0]


def frames(steps):
    """A frame per step with the given token probabilities, a blank frame after each; rows are log-probabilities.
    The rows are normalised: a lone probability below 1 becomes 1, so give the rest of the frame to another token."""
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


def apart(d):
    """Two probabilities that fill a frame, the second above the first by d in the logarithm."""
    return 1 / (1 + math.exp(d)), math.exp(d) / (1 + math.exp(d))


def rows_without_blanks(tokens):
    """A frame per token and no blank frames between them."""
    lp = np.full((len(tokens), C), -40.0)
    for i, tok in enumerate(tokens):
        lp[i, tok] = 0.0
    return lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))


def random_logprobs(seed, t=40):
    x = np.random.default_rng(seed).normal(size=(t, C)) * 3
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def final_state(lex, ids):
    st = g2h.INIT
    for t in ids:
        st = g2h.step(lex, st, t)
    return st


def one_chunk_npz(folder, meta=None, lp=None):
    """Saved log-probabilities of one chunk that says "ВМС", written without g2h."""
    v, m, s = ids_of('▁В', 'М', 'С')
    lp = frames([{v: 1.0}, {m: 1.0}, {s: 1.0}]) if lp is None else lp
    np.savez(folder / 'rec.npz', starts=np.array([0.0]), ends=np.array([1.0]), meta=np.array(json.dumps(meta or {})), c0000=lp)
    return folder / 'rec.npz'


def make_wav(path, rate, samples):
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.full(samples, 16384, dtype=np.int16).tobytes())


@pytest.fixture
def tool(monkeypatch):
    """main() with the stand-in tokenizer; the module-level log path is put back after the test."""
    monkeypatch.setattr(g2h, 'load_tokenizer', lambda models: SP)
    monkeypatch.setattr(g2h, 'LOG_PATH', None)


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


def test_the_bonus_cannot_pull_in_a_pruned_token():
    lex = g2h.Lexicon(SP, ['ВМС'])
    term, alt = ids_of('▁В', 'М', 'С'), SP.piece_to_id('е')
    p_lo = 1 / (1 + math.exp(12))                                # ln(p_hi) - ln(p_lo) = 12 > PRUNE, the bonus is 15
    lp = frames([{term[0]: 1.0}, {term[1]: 1.0}, {term[2]: p_lo, alt: 1 - p_lo}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 5)) == 'ВМе'


def test_a_dictionary_token_needs_acoustic_support():
    # a token that walks a dictionary form is tried only within 5 of the frame's best: a bonus of 9 pays for a loss of 4.9 and
    # would pay for 5.1, yet there the token is not tried; the reserve does not bring it back
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, m, s, e = ids_of('▁В', 'М', 'С', 'е')
    for d, text in ((4.9, 'ВМС'), (5.1, 'ВМе')):
        p_lo, p_hi = apart(d)
        lp = frames([{v: 1.0}, {m: 1.0}, {s: p_lo, e: p_hi}])
        assert SP.decode_ids(g2h.beam_decode(lex, lp, 0)) == 'ВМе'
        assert SP.decode_ids(g2h.beam_decode(lex, lp, 3)) == text
        assert SP.decode_ids(g2h.beam_decode(lex, lp, 3, reserve=4)) == text


def test_the_term_threshold_is_the_checked_setting():
    # 5 together with the reserve ranking: checked on three recorded meetings; 4 loses confirmed fixes, the frame pruning of 10
    # lets false short terms in
    assert g2h.TERM_PRUNE == 5.0 and g2h.TERM_PRUNE < g2h.PRUNE


def test_a_weak_first_token_does_not_start_a_term():
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, f, m, s = ids_of('▁В', '▁Ф', 'М', 'С')
    for d, text in ((4.0, 'ВМС'), (6.0, 'ФМС')):
        p_lo, p_hi = apart(d)
        assert SP.decode_ids(g2h.beam_decode(lex, frames([{v: p_lo, f: p_hi}, {m: 1.0}, {s: 1.0}]), 3)) == text


def test_a_term_is_not_written_into_silence():
    # three frames of silence where the letters of the term are each d below the blank. 3.5 each: the bonus pays (12 > 10.5) and
    # every token is within the threshold, the term is written. 5.5 each: a bonus of 6 per token would pay, the tokens are not tried
    lex = g2h.Lexicon(SP, ['ВМС'])
    a, b = ids_of('▁а', '▁б')

    def chunk(d):
        p_lo, p_hi = apart(d)
        return frames([{a: 1.0}] + [{tok: p_lo, BLANK: p_hi} for tok in ids_of('▁В', 'М', 'С')] + [{b: 1.0}])
    assert SP.decode_ids(g2h.beam_decode(lex, chunk(3.5), 0)) == 'а б'
    assert SP.decode_ids(g2h.beam_decode(lex, chunk(3.5), 4)) == 'а ВМС б'
    assert SP.decode_ids(g2h.beam_decode(lex, chunk(5.5), 6)) == 'а б'


def test_a_bare_word_mark_inside_a_term_is_outside_the_term_threshold():
    # "ТН ВЭД" said in one breath: the bare "▁" between its words is weak on its frame. A token without letters is tried within
    # the frame pruning, as before (7 below the best: tried, 12: not); a letter of the term 7 below the best is not tried
    sp = SplitSP({'ТН ВЭД': [['▁Т', 'Н', '▁', 'В', 'Э', 'Д']]})
    lex = g2h.Lexicon(sp, ['ТН ВЭД'])
    t, n, mark, v, ee, d, e = (sp.index[p] for p in ('▁Т', 'Н', '▁', 'В', 'Э', 'Д', 'е'))

    def chunk(space, letter):
        return frames([{t: 1.0}, {n: 1.0}, space, {v: 1.0}, letter, {d: 1.0}])
    for gap, text in ((7.0, 'ТН ВЭД'), (12.0, 'ТНВЭД')):
        p_lo, p_hi = apart(gap)
        lp = chunk({mark: p_lo, BLANK: p_hi}, {ee: 1.0})
        assert sp.decode_ids(g2h.beam_decode(lex, lp, 0)) == 'ТНВЭД'
        assert sp.decode_ids(g2h.beam_decode(lex, lp, 3)) == text
    p_lo, p_hi = apart(7.0)
    assert sp.decode_ids(g2h.beam_decode(lex, chunk({mark: 1.0}, {ee: p_lo, e: p_hi}), 3)) == 'ТН ВеД'


def test_a_token_outside_a_term_is_tried_within_the_frame_pruning_as_before():
    # after "ВМС" and an ending of ENDING letters one more letter loses the bonus, a space secures it. The space walks no
    # dictionary form: 7 below the frame's best it is still tried, beyond PRUNE it is not (the bonus of 15 would pay for 12)
    lex = g2h.Lexicon(SP, ['ВМС'])
    ending = 'абвгежзик'[:g2h.ENDING]
    word = ids_of('▁В', 'М', 'С', *ending)
    letter, space = ids_of('д', '▁д')
    for d, text in ((7.0, 'ВМС%s д' % ending), (12.0, 'ВМС%sд' % ending)):
        p_lo, p_hi = apart(d)
        lp = frames([{tok: 1.0} for tok in word] + [{space: p_lo, letter: p_hi}])
        assert SP.decode_ids(g2h.beam_decode(lex, lp, 0)) == 'ВМС%sд' % ending
        assert SP.decode_ids(g2h.beam_decode(lex, lp, 5)) == text


def test_a_repeated_token_needs_a_blank_between():
    lex = g2h.Lexicon(SP, ['Анна'])                               # the dictionary spells the doubled letter, the frames do not
    a1, n, a2 = ids_of('▁А', 'н', 'а')
    assert SP.decode_ids(g2h.beam_decode(lex, rows_without_blanks([a1, n, n, a2]), 2)) == 'Ана'


def test_abandoned_term_keeps_no_bonus():
    lex = g2h.Lexicon(SP, ['ВМС'])
    started = ids_of('▁В', 'М', 'е')                              # the term begins and leaves before its end
    other = ids_of('▁а', 'б', 'в')                                # a slightly likelier word at every step
    lp = frames([{a: 0.45, b: 0.55} for a, b in zip(started, other, strict=True)])
    assert g2h.beam_decode(lex, lp, 5) == other
    assert g2h.final_bonus(final_state(lex, started)) == 0


def test_unfinished_term_has_no_bonus_at_the_end_of_a_chunk_state():
    lex = g2h.Lexicon(SP, ['ВМС'])
    assert g2h.final_bonus(final_state(lex, ids_of('▁В', 'М'))) == 0


def test_unfinished_term_has_no_bonus_at_the_end_of_a_chunk_search():
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, m, e = ids_of('▁В', 'М', 'е')
    lp = frames([{v: 1.0}, {m: 0.45, e: 0.55}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2)) == 'Ве'


@pytest.mark.parametrize('ending, kept', [('', True), ('ом', True), ('ами', True), ('ными', True), ('ского', False)])
def test_ending_after_a_term(ending, kept):
    lex = g2h.Lexicon(SP, ['ВМС'])
    term = ids_of('▁В', 'М', 'С')
    bonus = g2h.final_bonus(final_state(lex, term + ids_of(*ending)))
    assert bonus == (len(term) if kept else 0)


def test_an_ending_is_counted_in_letters_not_in_tokens():
    sp = SplitSP()
    lex = g2h.Lexicon(sp, ['ВМС'])
    term = [sp.index[p] for p in ('▁В', 'М', 'С')]
    sk, uyu = sp.index['ск'], sp.index['ую']
    assert g2h.final_bonus(final_state(lex, term + [sk, uyu])) == 3           # 2 tokens, 4 letters: an ending
    assert g2h.final_bonus(final_state(lex, term + [sk, uyu, sk])) == 0       # 3 tokens, 6 letters: another word


def test_a_four_letter_ending_keeps_the_word_whole():
    # "ВМС" lives on its bonus, and every letter of the ending is cheap to skip. An ending the bonus does not reach is cut to the
    # length it reaches: with three letters allowed the search wrote "ВМСным"
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, m, s, e = ids_of('▁В', 'М', 'С', 'е')
    p_lo, p_hi = apart(2.0)
    lp = frames([{v: 1.0}, {m: 1.0}, {s: p_lo, e: p_hi}] + [{tok: 0.6, BLANK: 0.4} for tok in ids_of('н', 'ы', 'м', 'и')])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 0)) == 'ВМеными'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 3)) == 'ВМСными'


def test_complete_term_keeps_its_bonus_before_the_next_word():
    lex = g2h.Lexicon(SP, ['ВМС'])
    term = ids_of('▁В', 'М', 'С')
    assert g2h.final_bonus(final_state(lex, term + ids_of('▁а', 'б'))) == 3


def test_complete_term_with_an_ending_keeps_its_bonus_before_the_next_word():
    lex = g2h.Lexicon(SP, ['ВМС'])
    term = ids_of('▁В', 'М', 'С')
    assert g2h.final_bonus(final_state(lex, term + ids_of('о', 'м', '▁а'))) == 3


def test_term_with_an_ending_keeps_its_bonus_during_the_search():
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, m, s, o, a = ids_of('▁В', 'М', 'С', 'о', '▁а')
    lp = frames([{v: 1.0}, {m: 1.0}, {s: 1.0}, {o: 0.6, a: 0.4}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2, beam=1)) == 'ВМСо'


def test_one_token_term_is_complete_at_once():
    lex = g2h.Lexicon(SP, ['В'])
    assert g2h.final_bonus(final_state(lex, ids_of('▁В'))) == 1


def test_a_longer_term_does_not_take_the_bonus_of_the_short_one():
    lex = g2h.Lexicon(SP, ['ВМС', 'ВМС на борту'])
    assert g2h.final_bonus(final_state(lex, toks('ВМС на берегу'))) == 3         # the longer term is left inside a word
    assert g2h.final_bonus(final_state(lex, toks('ВМС надолго'))) == 3           # "надолго" is a word, not an ending of "ВМС"
    assert g2h.final_bonus(final_state(lex, toks('ВМС на суше'))) == 3           # the longer term is left at a word start
    assert g2h.final_bonus(final_state(lex, toks('ВМС на'))) == 3                # the chunk ends inside the longer term
    assert g2h.final_bonus(final_state(lex, toks('ВМС на борту'))) == 10         # the longer term: its own tokens, not 3 + 10
    assert g2h.final_bonus(final_state(lex, toks('ВМСского'))) == 0              # the ending rule is not loosened
    lex = g2h.Lexicon(SP, ['ВМС', 'ВМСского типа'])
    assert g2h.final_bonus(final_state(lex, toks('ВМСского тона'))) == 0         # "ВМСского" is another word: nothing to keep


def test_forms_of_a_term():
    assert g2h.term_forms('ВМС') == ['ВМС']
    assert g2h.term_forms('наименование') == ['наименование', 'Наименование']
    assert g2h.term_forms('Озон') == ['Озон']                                    # a name: no "озон"
    assert g2h.term_forms('озон') == ['озон', 'Озон']
    assert g2h.term_forms('ТТН на возврат') == ['ТТН на возврат']                # an abbreviation first: no "тТН на возврат"
    assert g2h.term_forms('ЭДО-оператор') == ['ЭДО-оператор']
    assert g2h.term_forms('ePASS') == ['ePASS']                                  # a capital after the first letter: a brand
    assert g2h.term_forms('В пути') == ['В пути']                                # begins with a capital: as written
    assert g2h.term_forms('в пути') == ['в пути', 'В пути']
    assert g2h.term_forms('В ЭДО') == ['В ЭДО']                                  # all in capitals: as written


def test_a_name_is_not_written_in_lower_case():
    # "Озон" in the dictionary gives no bonus to "озон"; a term written in lower case takes the capital of a sentence start too
    big, small, a, z, o, n = ids_of('▁О', '▁о', '▁а', 'з', 'о', 'н')
    p_lo, p_hi = apart(1.0)

    def text(term, first):
        lp = frames([{first: p_lo, a: p_hi}, {z: 1.0}, {o: 1.0}, {n: 1.0}])
        return SP.decode_ids(g2h.beam_decode(g2h.Lexicon(SP, [term]), lp, 3))
    assert (text('Озон', big), text('Озон', small)) == ('Озон', 'азон')
    assert (text('озон', big), text('озон', small)) == ('Озон', 'озон')
    assert [f for _, f, _ in g2h.Lexicon(SP, ['Озон']).forms] == ['Озон']


def test_forms_of_a_term_reach_the_tree():
    lex = g2h.Lexicon(SP, ['наименование'])
    assert [f for _, f, _ in lex.forms] == ['наименование', 'Наименование']


def test_form_with_an_unknown_character_is_rejected():
    lex = g2h.Lexicon(SP, ['ВМ1', 'ВМС'])
    assert [(f, why) for _, f, _, why in lex.rejected] == [('ВМ1', 'a special or unknown token')]
    assert [f for _, f, s in lex.forms if not s] == ['ВМ1']
    assert lex.sequences() == 1


def test_splits_that_cannot_carry_the_bonus_are_rejected_with_the_reason():
    sp = SplitSP({'ВМС': [['▁В', 'М', 'С'], ['▁В', 'М', 'с'], ['В', 'М', 'С']]})
    lex = g2h.Lexicon(sp, ['ВМС'])
    assert [(pieces, why) for _, _, pieces, why in lex.rejected] == [
        (['▁В', 'М', 'с'], 'decodes back as "ВМс"'), (['В', 'М', 'С'], 'the first token does not start a word')]
    assert lex.sequences() == 1 and lex.forms == [('ВМС', 'ВМС', [['▁В', 'М', 'С']])]


def test_decoder_without_dictionary_is_greedy():
    lp = random_logprobs(7)
    assert g2h.Decoder(SP).text(lp) == SP.decode_ids(g2h.greedy_ids(lp)).strip()
    assert g2h.Decoder(SP, ['ВМС'], w=0).text(lp) == g2h.Decoder(SP).text(lp)


def test_decoder_passes_w_and_reserve_to_the_search(monkeypatch):
    seen, real = {}, g2h.beam_decode

    def spy(lex, logp, w, **kw):
        seen.update(w=w, **kw)
        return real(lex, logp, w, **kw)
    monkeypatch.setattr(g2h, 'beam_decode', spy)
    g2h.Decoder(SP, ['ВМС']).ids(random_logprobs(1))
    assert seen == {'w': 3, 'reserve': 4}


def test_load_terms_skips_comments_and_blank_lines(tmp_path):
    p = tmp_path / 'terms.txt'
    p.write_text('# header\n\nВМС\nТН ВЭД   # a comment\n  \n', encoding='utf-8')
    assert g2h.load_terms(p) == ['ВМС', 'ТН ВЭД']


def test_load_terms_cleans_what_an_editor_hides(tmp_path):
    p = tmp_path / 'terms.txt'
    # a byte order mark before a comment, a no-break space, a doubled space, a zero-width space, a repeated term
    p.write_bytes('﻿# my terms\nНДС\nТН ВЭД\nнадбавка  импортёра\nТ​ТН\nНДС\n'.encode('utf-8'))
    assert g2h.load_terms(p) == ['НДС', 'ТН ВЭД', 'надбавка импортёра', 'ТТН']


def test_time_formats():
    assert g2h.fmt(3661.9) == '1:01:01'
    assert g2h.srt_ts(3661.5) == '01:01:01,500'
    assert g2h.srt_ts(0.0014) == '00:00:00,001'
    assert g2h.srt_ts(0.0016) == '00:00:00,002'                  # rounded, not cut


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


def test_logprobs_are_saved_under_the_name_given(tmp_path):
    lp = random_logprobs(1, 5).astype(np.float32)
    g2h.save_logprobs(tmp_path / 'rec', [lp], [(0.0, 1.0, 'a')], {'model': 'm'})      # a name without ".npz"
    assert sorted(p.name for p in tmp_path.iterdir()) == ['rec']
    got, starts, _, _ = g2h.load_logprobs(tmp_path / 'rec')
    assert got[0].tolist() == lp.tolist() and starts == [0.0]


def test_logprobs_without_times_take_them_from_the_chunk_file(tmp_path):
    np.savez(tmp_path / 'old.npz', c0000=random_logprobs(3, 4), c0001=random_logprobs(4, 4))
    (tmp_path / 'chunks_25s.json').write_text(json.dumps({'chunks': [{'start': 1.0, 'end': 2.0}, {'start': 3.0, 'end': 4.5}]}),
                                              encoding='utf-8')
    _, starts, ends, _ = g2h.load_logprobs(tmp_path / 'old.npz')
    assert (starts, ends) == ([1.0, 3.0], [2.0, 4.5])


def test_speech_parts_come_from_the_chunk_file(tmp_path):
    np.savez(tmp_path / 'old.npz', c0000=random_logprobs(3, 4))
    chunk = {'start': 1.0, 'end': 2.0, 'samples': 16000, 'parts': [[16000, 32000]]}
    (tmp_path / 'chunks_25s.json').write_text(json.dumps({'chunks': [chunk]}), encoding='utf-8')
    *_, meta = g2h.load_logprobs(tmp_path / 'old.npz')
    assert meta == {'chunks_from': 'chunks_25s.json', 'samples': [16000], 'parts': [[[16000, 32000]]]}


def test_logprobs_with_a_wrong_number_of_chunk_times_are_refused(tmp_path):
    np.savez(tmp_path / 'old.npz', c0000=random_logprobs(3, 4), c0001=random_logprobs(4, 4))
    (tmp_path / 'chunks_25s.json').write_text(json.dumps({'chunks': [{'start': 1.0, 'end': 2.0}]}), encoding='utf-8')
    with pytest.raises(RuntimeError, match='2 chunks, but 1 chunk times'):
        g2h.load_logprobs(tmp_path / 'old.npz')


def test_logprobs_without_times_and_without_the_chunk_file_are_refused(tmp_path):
    np.savez(tmp_path / 'old.npz', c0000=random_logprobs(3, 4))
    with pytest.raises(RuntimeError, match='has no chunk times'):
        g2h.load_logprobs(tmp_path / 'old.npz')


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


def test_reserve_leaves_the_other_slots_to_the_bonus_ranking():
    lex = g2h.Lexicon(SP, ['ВМС'])
    v, m, s, e, n = ids_of('▁В', 'М', 'С', 'е', 'н')
    lp = frames([{v: 1.0}, {e: 0.4, n: 0.35, m: 0.25}, {s: 1.0}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2, beam=2, reserve=1)) == 'ВМС'


def test_reserve_does_not_take_a_slot_twice():
    lex = g2h.Lexicon(SP, ['ВМС', 'Век'])
    v, m, e, k = ids_of('▁В', 'М', 'е', 'к')
    lp = frames([{v: 1.0}, {m: 0.6, e: 0.4}, {k: 1.0}])
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 2, beam=2, reserve=1)) == 'Век'


def chunk_after_a_fix(d, crowd, p_o):
    """'ВМ' + a frame where 'С' loses to 'е' by d, so 'ВМС' lives on its secured bonus; then the word 'Коты', whose beginning
    is also the beginning of the dictionary terms 'К?Я' that never complete."""
    lex = g2h.Lexicon(SP, ['ВМС'] + ['К%sЯ' % ch for ch in crowd])
    v, m, s, e = ids_of('▁В', 'М', 'С', 'е')
    k, o, t, y = ids_of('▁К', 'о', 'т', 'ы')
    p_lo, p_hi = 1 / (1 + math.exp(d)), math.exp(d) / (1 + math.exp(d))
    second = {o: p_o}
    second.update({SP.piece_to_id(ch): (1 - p_o) / len(crowd) for ch in crowd})
    return lex, frames([{v: 1.0}, {m: 1.0}, {s: p_lo, e: p_hi}, {k: 1.0}, second, {t: 1.0}, {y: 1.0}])


def test_reserve_counts_the_bonus_a_hypothesis_has_earned():
    # The reserve guards the paths that are best by the acoustic score plus the bonus they would keep if the chunk ended now
    # (secured terms and a complete one), never the bonus of a term still being spelled. After "ВМС" has won on its bonus, the
    # reserved slots follow the winning path, so its plain "Ко" is not crowded out by the terms "К?Я" that never complete.
    lex, lp = chunk_after_a_fix(2.0, 'АБВ', 0.4)
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 3, beam=4, reserve=0)) == 'ВМС Коты'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 3, beam=4, reserve=2)) == 'ВМС Коты'
    lex, lp = chunk_after_a_fix(4.0, 'АБВГДЕЖЗИЛМНП', 0.35)                        # the shipped defaults: beam 16, w 3, reserve 4
    assert SP.decode_ids(g2h.beam_decode(lex, lp, g2h.DEFAULT_W, reserve=0)) == 'ВМС Коты'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, g2h.DEFAULT_W, reserve=g2h.DEFAULT_RESERVE)) == 'ВМС Коты'


def test_reserve_does_not_count_the_bonus_of_a_term_still_being_spelled():
    # Three slots, and without the reserve the terms "К?Я", still being spelled, take them all. One reserved slot saves the plain
    # "Ко" of the corrected path only under the right ranking: by the acoustic score alone it goes to a "ВМе" path, with the live
    # bonus counted — to one more "К?Я".
    lex, lp = chunk_after_a_fix(2.0, 'АБВ', 0.4)
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 3, beam=3, reserve=0)) == 'ВМС КАты'
    assert SP.decode_ids(g2h.beam_decode(lex, lp, 3, beam=3, reserve=1)) == 'ВМС Коты'


def test_ctc_align_finds_the_frames_of_every_token():
    v, m = ids_of('▁В', 'М')
    lp = frames([{v: 1.0}, {m: 1.0}])                            # frames: В, blank, М, blank
    al = g2h.ctc_align(lp, [v, m])
    assert [(f0, f1) for f0, f1, _ in al] == [(0, 0), (2, 2)]
    assert all(p > -1e-6 for _, _, p in al)
    assert g2h.ctc_align(lp, []) == []
    assert g2h.ctc_align(lp[:1], [v, m]) is None                 # two tokens do not fit one frame


def test_ctc_align_lets_different_tokens_meet_without_a_blank():
    v, m = ids_of('▁В', 'М')
    al = g2h.ctc_align(rows_without_blanks([v, m]), [v, m])
    assert [(f0, f1) for f0, f1, _ in al] == [(0, 0), (1, 1)]


def test_ctc_align_needs_a_blank_between_equal_tokens():
    n = ids_of('н')[0]
    assert g2h.ctc_align(rows_without_blanks([n, n]), [n, n]) is None


def test_token_confidence_is_its_best_frame():
    v = ids_of('▁В')[0]
    lp = np.full((2, C), -40.0)
    lp[0, v], lp[1, v] = math.log(0.5), math.log(0.9)
    (f0, f1, best), = g2h.ctc_align(lp, [v])
    assert (f0, f1) == (0, 1) and best == pytest.approx(math.log(0.9))


def test_chunk_words_times_and_confidence():
    v, m, a = ids_of('▁В', 'М', '▁а')
    lp = frames([{v: 1.0}, {m: 1.0}, {a: 1.0}])                  # 6 frames
    clock = g2h.frame_clock(len(lp), start=10.0, end=16.0)
    conf, words = g2h.chunk_words(SP, lp, [v, m, a], clock)
    assert [w for w, _, _, _ in words] == ['ВМ', 'а']
    assert [(s, e) for _, s, e, _ in words] == [(10.0, 13.0), (14.0, 15.0)]
    assert conf == 1.0 and all(p == 1.0 for *_, p in words)
    assert g2h.chunk_words(SP, lp, [], clock) == (None, [])     # an empty chunk: no confidence, "conf": null in the words file


def test_confidences_are_geometric_means():
    v, m, a, x = ids_of('▁В', 'М', '▁а', 'ж')                     # the rest of every frame goes to "ж": rows stay normalised
    lp = frames([{v: 0.9, x: 0.1}, {m: 0.4, x: 0.6}, {a: 0.5, x: 0.5}])
    conf, words = g2h.chunk_words(SP, lp, [v, m, a], g2h.frame_clock(len(lp), start=0.0, end=6.0))
    assert [w[3] for w in words] == [0.6, 0.5]                     # sqrt(0.9 * 0.4), 0.5
    assert conf == 0.565                                           # (0.9 * 0.4 * 0.5) ** (1 / 3)


def test_a_word_without_text_is_left_out():
    bare, a = ids_of('▁', '▁а')
    lp = frames([{bare: 1.0}, {a: 1.0}])
    _, words = g2h.chunk_words(SP, lp, [bare, a], g2h.frame_clock(len(lp), start=0.0, end=4.0))
    assert [w[0] for w in words] == ['а']


def test_frame_clock_goes_through_the_speech_parts():
    clock = g2h.frame_clock(100, 32000, [[0, 16000], [32000, 48000]])     # two 1 s parts with 1 s of silence between
    assert clock(0) == 0.0
    assert clock(25) == 0.5
    assert clock(75) == 2.5
    assert clock(100) == 3.0


def test_a_point_on_the_joint_of_two_parts():
    sr = g2h.SR
    parts = [[0, sr], [10 * sr, 11 * sr]]                                        # 1 s of speech, 9 s of silence, 1 s of speech
    clock = g2h.frame_clock(100, 2 * sr, parts)
    assert clock(50) == 10.0                                                     # a word that starts on the joint: the next part
    assert clock(50, True) == 1.0                                                # a word that ends there: this part
    assert (clock(49), clock(51)) == (0.98, 10.02)
    v, m = ids_of('▁В', '▁М')
    lp = rows_without_blanks([v] * 50 + [m] * 50)                                # one word ends on the joint, the next starts
    _, words = g2h.chunk_words(SP, lp, [v, m], clock)
    assert [w[:3] for w in words] == [['В', 0.0, 1.0], ['М', 10.0, 11.0]]


def test_read_wav_refuses_another_format(tmp_path):
    make_wav(tmp_path / 'a.wav', 8000, 80)
    with pytest.raises(RuntimeError, match='16 kHz mono 16-bit'):
        g2h.read_wav(tmp_path / 'a.wav')


def test_read_wav_reads_and_cuts(tmp_path):
    make_wav(tmp_path / 'a.wav', 16000, 32000)
    audio = g2h.read_wav(tmp_path / 'a.wav')
    assert audio.dtype == np.float32 and len(audio) == 32000 and audio[0] == 0.5
    assert len(g2h.read_wav(tmp_path / 'a.wav', limit_sec=0.5)) == 8000


def test_the_wav_format_is_checked_before_the_vad_pass(tmp_path, monkeypatch):
    make_wav(tmp_path / 'a.wav', 8000, 800)

    def vad_ran_first(*a):
        raise AssertionError('the VAD pass ran before the WAV format was checked')
    monkeypatch.setattr(g2h, 'vad_chunks', vad_ran_first)
    with pytest.raises(RuntimeError, match='not a 16 kHz mono 16-bit WAV'):
        g2h.run_audio(g2h.parse_args(['--audio', str(tmp_path / 'a.wav'), '--out', str(tmp_path / 'o.txt')]), [])


def test_the_vad_child_has_a_timeout(tmp_path, monkeypatch):
    seen = {}

    def spy(cmd, **kw):
        seen.update(kw, tmp=cmd[cmd.index('--vad-only') + 1])
        raise subprocess.TimeoutExpired(cmd, kw.get('timeout'))
    monkeypatch.setattr(g2h.subprocess, 'run', spy)
    with pytest.raises(RuntimeError, match='did not finish'):
        g2h.vad_chunks(tmp_path / 'a.wav', 0, 'python')
    assert seen.get('timeout') == g2h.VAD_TIMEOUT > 0
    assert not os.path.exists(seen['tmp'])                        # the temporary file of the child is removed


def test_words_file_from_saved_logprobs(tmp_path, monkeypatch):
    monkeypatch.setattr(g2h, 'load_tokenizer', lambda models: SP)
    v, m = ids_of('▁В', 'М')
    lps = [frames([{v: 1.0}, {m: 1.0}]).astype(np.float32)]
    g2h.save_logprobs(tmp_path / 'rec.npz', lps, [(5.0, 6.0, 'ВМ')], {'model': 'm', 'samples': [16000], 'parts': [[[80000, 96000]]]})
    out, words = tmp_path / 'rec.txt', tmp_path / 'rec.words.json'
    assert g2h.main(['--from-logprobs', str(tmp_path / 'rec.npz'), '--out', str(out), '--words', str(words)]) == 0
    data = json.loads(words.read_text(encoding='utf-8'))
    assert sorted(data) == ['approximate_times', 'audio', 'chunks', 'decoder', 'model']
    assert data['approximate_times'] is False
    assert data['chunks'][0]['words'] == [['ВМ', 5.0, 5.75, 1.0]]


def test_words_without_saved_parts_are_marked_approximate(tmp_path, tool, capsys):
    words = tmp_path / 'w.json'
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'o.txt'),
                     '--words', str(words)]) == 0
    assert json.loads(words.read_text(encoding='utf-8'))['approximate_times'] is True
    assert 'spread linearly' in capsys.readouterr().out


def test_side_effects_of_a_run_from_saved_logprobs(tmp_path, tool, capsys):
    v, m = ids_of('▁В', 'М')
    lps = [frames([{v: 1.0}, {m: 1.0}]).astype(np.float32)]
    # two 0.5 s speech parts with 0.5 s of silence between: the clock of the parts and the linear one differ
    g2h.save_logprobs(tmp_path / 'rec.npz', lps, [(5.0, 6.5, '')],
                      {'model': 'm', 'samples': [16000], 'parts': [[[80000, 88000], [96000, 104000]]]})
    out, words = tmp_path / 'rec.txt', tmp_path / 'rec.words.json'
    assert g2h.main(['--from-logprobs', str(tmp_path / 'rec.npz'), '--out', str(out), '--words', str(words)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ['rec.npz', 'rec.srt', 'rec.txt', 'rec.words.json']    # no partial file
    assert b'\r' not in out.read_bytes() and b'\r' not in (tmp_path / 'rec.srt').read_bytes()
    assert 'GigaAM' in out.read_text(encoding='utf-8').splitlines()[1]
    assert json.loads(words.read_text(encoding='utf-8'))['chunks'][0]['words'] == [['ВМ', 5.0, 6.25, 1.0]]
    log = capsys.readouterr().out
    assert '=== START g2h: ' in log and '=== DONE g2h: 1 chunks (empty 0)' in log


def test_partial_file_has_the_documented_name(tmp_path, tool, monkeypatch):
    seen, real = [], g2h.write_outputs

    def spy(*args):
        seen.extend(sorted(p.name for p in tmp_path.iterdir()))
        return real(*args)
    monkeypatch.setattr(g2h, 'write_outputs', spy)
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'rec.txt')]) == 0
    assert seen == ['rec.npz', 'rec.partial.txt']


@pytest.mark.skipif(os.name != 'nt', reason='an open file can be removed on POSIX; the defect is a Windows one')
def test_a_partial_file_held_open_does_not_fail_a_finished_run(tmp_path, tool, monkeypatch, capsys):
    out, held = tmp_path / 'o.txt', []
    real = g2h.write_outputs

    def write_then_hold_the_partial(*a, **k):
        real(*a, **k)
        held.append(open(g2h.partial_path(out), encoding='utf-8'))
    monkeypatch.setattr(g2h, 'write_outputs', write_then_hold_the_partial)
    try:
        assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(out)]) == 0
    finally:
        for h in held:
            h.close()
    assert 'the partial file stays' in capsys.readouterr().out


def test_dictionary_reaches_the_transcript_through_main(tmp_path, tool):
    term, alt = ids_of('▁В', 'М', 'С'), SP.piece_to_id('е')
    p_lo, p_hi = 1 / (1 + math.e), math.e / (1 + math.e)
    npz = one_chunk_npz(tmp_path, lp=frames([{term[0]: 1.0}, {term[1]: 1.0}, {term[2]: p_lo, alt: p_hi}]))
    (tmp_path / 'terms.txt').write_text('ВМС\n', encoding='utf-8')
    plain, fixed = tmp_path / 'plain.txt', tmp_path / 'fixed.txt'
    assert g2h.main(['--from-logprobs', str(npz), '--out', str(plain)]) == 0
    assert g2h.main(['--from-logprobs', str(npz), '--out', str(fixed), '--dict', str(tmp_path / 'terms.txt')]) == 0
    assert plain.read_text(encoding='utf-8').splitlines()[3] == '[0:00:00 → 0:00:01] ВМе'
    assert fixed.read_text(encoding='utf-8').splitlines()[3] == '[0:00:00 → 0:00:01] ВМС'


def test_a_run_tells_about_a_form_the_bonus_never_applies_to(tmp_path, tool, capsys):
    (tmp_path / 'terms.txt').write_text('ВМ1\nВМС\n', encoding='utf-8')
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'o.txt'),
                     '--dict', str(tmp_path / 'terms.txt')]) == 0
    assert [ln for ln in capsys.readouterr().out.splitlines() if ' ! ' in ln and 'ВМ1' in ln]


def test_logprobs_of_another_model_are_refused(tmp_path, tool, capsys):
    lp = np.full((4, C - 9), -40.0)                                # fewer classes than the tokenizer has
    lp[:, -1] = 0.0
    lp[1, 5] = 5.0
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path, lp=lp)), '--out', str(tmp_path / 'o.txt')]) == 2
    assert 'another model' in capsys.readouterr().out


def test_the_header_of_a_redecoding_tells_about_the_limit(tmp_path, tool):
    out = tmp_path / 'o.txt'
    npz = one_chunk_npz(tmp_path, meta={'model': 'm', 'limit_sec': 120.0})
    assert g2h.main(['--from-logprobs', str(npz), '--out', str(out)]) == 0
    assert 'first 120 s only' in out.read_text(encoding='utf-8').splitlines()[1]


def test_log_file_gets_the_same_lines(tmp_path, tool):
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'rec.txt'),
                     '--log', str(tmp_path / 'run.log')]) == 0
    text = (tmp_path / 'run.log').read_text(encoding='utf-8')
    assert '=== START g2h: ' in text and '=== DONE g2h: ' in text


@pytest.mark.parametrize('lost', ['while the chunks are decoded', 'after the outputs are written'])
def test_a_log_file_that_fails_does_not_decide_the_run(lost, tmp_path, tool, monkeypatch, capsys):
    log_file, out = tmp_path / 'run.log', tmp_path / 'o.txt'
    name = 'load_logprobs' if lost == 'while the chunks are decoded' else 'write_outputs'
    real = getattr(g2h, name)

    def lose_the_log(*a, **k):
        got = real(*a, **k)
        log_file.unlink()
        log_file.mkdir()                                                         # from now on the log cannot be opened
        return got
    monkeypatch.setattr(g2h, name, lose_the_log)
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(out), '--log', str(log_file)]) == 0
    assert out.is_file() and not g2h.partial_path(out).exists()
    assert capsys.readouterr().out.count('cannot write the log file') == 1      # told once, then the log goes to stdout only


def test_exit_code_1_is_no_audio_and_2_is_a_failure(tmp_path, capsys):
    assert g2h.main(['--audio', str(tmp_path / 'absent.wav'), '--out', str(tmp_path / 'a.txt')]) == 1
    assert '=== FAILED g2h: no audio file' in capsys.readouterr().out
    assert g2h.main(['--from-logprobs', str(tmp_path / 'absent.npz'), '--out', str(tmp_path / 'b.txt')]) == 2
    assert '=== FAILED g2h\n' in capsys.readouterr().out


@pytest.mark.parametrize('case', ['no dictionary file', 'a dictionary not in UTF-8', 'a log in a missing folder'])
def test_a_failure_before_the_run_is_code_2_with_the_marker(case, tmp_path, tool, capsys):
    (tmp_path / 'cp1251.txt').write_bytes('НДС\n'.encode('cp1251'))
    extra = {'no dictionary file': ['--dict', str(tmp_path / 'missing.txt')],
             'a dictionary not in UTF-8': ['--dict', str(tmp_path / 'cp1251.txt')],
             'a log in a missing folder': ['--log', str(tmp_path / 'no_such_folder' / 'run.log')]}[case]
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'o.txt')] + extra) == 2
    assert '=== FAILED g2h' in capsys.readouterr().out
    assert not (tmp_path / 'o.txt').exists()


def test_a_failure_is_not_written_to_the_log_of_an_earlier_run(tmp_path, tool):
    npz, first_log = one_chunk_npz(tmp_path), tmp_path / 'first.log'             # two runs in one process
    assert g2h.main(['--from-logprobs', str(npz), '--out', str(tmp_path / 'a.txt'), '--log', str(first_log)]) == 0
    assert g2h.main(['--from-logprobs', str(npz), '--out', str(tmp_path / 'b.txt'), '--dict', str(tmp_path / 'missing.txt')]) == 2
    assert 'FAILED' not in first_log.read_text(encoding='utf-8')


def test_dict_check_without_a_tokenizer_is_a_failure(tmp_path, monkeypatch, capsys):
    def no_tokenizer(models):
        raise RuntimeError('no tokenizer')
    monkeypatch.setattr(g2h, 'load_tokenizer', no_tokenizer)
    (tmp_path / 'terms.txt').write_text('ВМС\n', encoding='utf-8')
    assert g2h.main(['--dict', str(tmp_path / 'terms.txt'), '--dict-check']) == 2
    assert '=== FAILED g2h' in capsys.readouterr().out


def test_main_runs_with_a_stdout_that_cannot_be_reconfigured(tmp_path, tool, monkeypatch):
    monkeypatch.setattr(sys, 'stdout', io.StringIO())                            # an embedding caller may give any object
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(tmp_path / 'o.txt')]) == 0


def test_a_wrong_output_folder_is_refused_before_the_work(tmp_path, tool):
    out = tmp_path / 'o.txt'
    assert g2h.main(['--from-logprobs', str(one_chunk_npz(tmp_path)), '--out', str(out),
                     '--words', str(tmp_path / 'no_such_folder' / 'w.json')]) == 2
    assert not out.exists() and not g2h.partial_path(out).exists()               # nothing is written before the refusal


@pytest.mark.parametrize('flag', ['--out', '--srt', '--words', '--keep-logprobs'])
def test_every_output_path_is_checked_before_the_work(flag, tmp_path):
    for wrong in (tmp_path / 'no_such_folder' / 'file', tmp_path):               # a missing folder; a folder instead of a file
        paths = dict({'--out': str(tmp_path / 'o.txt')}, **{flag: str(wrong)})
        with pytest.raises(RuntimeError, match=flag):
            g2h.preflight(g2h.parse_args(['--audio', 'a.wav'] + [x for kv in paths.items() for x in kv]))
    g2h.preflight(g2h.parse_args(['--audio', 'a.wav', '--out', str(tmp_path / 'o.txt'), '--words', str(tmp_path / 'w.json')]))


def test_dict_check_tells_a_form_without_a_usable_split(tmp_path, tool, capsys):
    good, bad = tmp_path / 'good.txt', tmp_path / 'bad.txt'
    good.write_text('ВМС\n', encoding='utf-8')
    bad.write_text('ВМС\nВМ1\n', encoding='utf-8')
    assert g2h.main(['--dict-check', '--dict', str(good)]) == 0
    assert g2h.main(['--dict-check', '--dict', str(bad)]) == 2
    assert 'Forms the bonus never applies to: ВМ1' in capsys.readouterr().out


def test_parse_args_refuses_wrong_combinations():
    for argv in ([], ['--out', 'a.txt'], ['--audio', 'a.wav'], ['--audio', 'a.wav', '--from-logprobs', 'a.npz', '--out', 'a.txt'],
                 ['--dict-check']):
        with pytest.raises(SystemExit) as e:
            g2h.parse_args(argv)
        assert e.value.code == 2


@pytest.mark.parametrize('extra', [['--reserve', '-1'], ['--limit-sec', '-5'], ['--w', 'nan'], ['--w', 'inf'], ['--w', '-3'],
                                   ['--threads', '0']], ids=' '.join)
def test_parse_args_refuses_values_that_change_the_meaning(extra):
    with pytest.raises(SystemExit) as e:
        g2h.parse_args(['--audio', 'a.wav', '--out', 'o.txt'] + extra)
    assert e.value.code == 2


@pytest.mark.parametrize('out', ['x.srt', '.'])
def test_parse_args_refuses_an_out_that_cannot_hold_the_transcript(out):
    with pytest.raises(SystemExit) as e:                                         # "x.srt": the subtitles would replace it
        g2h.parse_args(['--audio', 'a.wav', '--out', out])
    assert e.value.code == 2


def test_defaults_are_the_checked_setting():
    # w = 3 with reserve 4: checked on full runs of three recorded meetings (no speech lost, more term fixes than w = 1)
    dec = g2h.Decoder(SP, ['ВМС'])
    assert (g2h.DEFAULT_W, g2h.DEFAULT_RESERVE) == (3, 4)
    assert (dec.w, dec.reserve) == (3, 4)
    assert 'reserve 4' in dec.desc
    a = g2h.parse_args(['--audio', 'a.wav', '--out', 'a.txt'])
    assert (a.w, a.reserve) == (3, 4)
