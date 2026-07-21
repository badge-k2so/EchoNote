"""
Streaming voice chat: VAD -> faster-whisper ASR -> LLM (streaming) -> VOICEVOX TTS
VOICEVOX を事前に起動しておく必要があります（デフォルト: http://localhost:50021）
"""

import argparse
import io
import queue
import re
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd

# Windows コンソール文字化け対策
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SENTENCE_END_RE = re.compile(r'[。？！\n]')
NO_THINK = '思考過程、推論メモ、<think>タグは出力しないでください。最終回答だけを出力してください。'

SYSTEM_PROMPTS = {
    'study': (
        'あなたは小中学生の学習サポートアシスタントです。'
        '答えを直接教えるのではなく、生徒が自分で考えられるような問いかけを返してください。'
        '1回の返答は2文以内の短い日本語にしてください。'
        + NO_THINK
    ),
    'english': (
        'You are a friendly English conversation tutor for Japanese middle school students. '
        'Keep every response to 1-2 short sentences. End with one simple question.'
        + NO_THINK
    ),
    'chat': (
        'あなたは親切なアシスタントです。'
        '短く、わかりやすい日本語で答えてください。返答は2〜3文以内にしてください。'
        + NO_THINK
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description='Streaming voice chat with VOICEVOX TTS')
    p.add_argument('--llm_model',         required=True)
    p.add_argument('--asr_model',         default='small')
    p.add_argument('--mode',              default='study', choices=list(SYSTEM_PROMPTS))
    p.add_argument('--language',          default='ja')
    p.add_argument('--silence_threshold', type=float, default=0.015)
    p.add_argument('--silence_duration',  type=float, default=1.2)
    p.add_argument('--max_tokens',        type=int,   default=150)
    p.add_argument('--voicevox_url',      default='http://localhost:50021')
    p.add_argument('--speaker_id',        type=int,   default=1,  help='VOICEVOX speaker ID (1=ずんだもん)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# VOICEVOX TTS
# ---------------------------------------------------------------------------

def voicevox_synthesize(text: str, url: str, speaker_id: int) -> np.ndarray | None:
    """テキストをVOICEVOX APIで音声に変換し numpy float32 配列で返す。"""
    try:
        r = requests.post(f'{url}/audio_query',
                          params={'text': text, 'speaker': speaker_id}, timeout=10)
        r.raise_for_status()
        query = r.json()

        r = requests.post(f'{url}/synthesis',
                          params={'speaker': speaker_id},
                          json=query, timeout=30)
        r.raise_for_status()

        buf = io.BytesIO(r.content)
        with wave.open(buf) as wf:
            n_frames  = wf.getnframes()
            framerate = wf.getframerate()
            raw       = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, framerate
    except Exception as e:
        print(f'[TTS error] {e}', flush=True)
        return None, None


def tts_worker(tts_q: queue.Queue, url: str, speaker_id: int,
               is_speaking: threading.Event):
    while True:
        text = tts_q.get()
        if text is None:
            break
        audio, sr = voicevox_synthesize(text, url, speaker_id)
        if audio is not None:
            is_speaking.set()
            sd.play(audio, sr)
            sd.wait()
        if tts_q.empty():
            is_speaking.clear()


def check_voicevox(url: str) -> bool:
    try:
        r = requests.get(f'{url}/version', timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VAD録音
# ---------------------------------------------------------------------------

def record_utterance(sample_rate: int, silence_threshold: float,
                     silence_duration: float, is_speaking: threading.Event) -> np.ndarray | None:
    chunk_sec     = 0.08
    chunk_samples = int(sample_rate * chunk_sec)
    silence_need  = int(silence_duration / chunk_sec)

    # TTS が終わるまで待つ
    waited = 0
    while is_speaking.is_set():
        time.sleep(0.05)
        waited += 0.05
        if waited > 30:
            return None
    time.sleep(0.15)

    buffer: list[np.ndarray] = []
    silent_chunks = 0
    speaking      = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32',
                        blocksize=chunk_samples) as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            if is_speaking.is_set():
                return None
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > silence_threshold:
                speaking      = True
                silent_chunks = 0
                buffer.append(chunk.copy())
            elif speaking:
                buffer.append(chunk.copy())
                silent_chunks += 1
                if silent_chunks >= silence_need:
                    break

    return np.concatenate(buffer).flatten() if speaking and buffer else None


# ---------------------------------------------------------------------------
# ストリーミング生成：<think>タグをトークン単位で除去
# ---------------------------------------------------------------------------

_NO_THINK_LOGIT_BIAS = {
    151667: -100.0,  # Qwen3 <think>
    151668: -100.0,  # Qwen3 </think>
    248068: -100.0,  # Qwen3.5 <think>
    248069: -100.0,  # Qwen3.5 </think>
}


def stream_llm(llm, messages: list[dict], max_tokens: int,
               tts_q: queue.Queue) -> str:
    full_response = ''
    buf           = ''
    in_think      = False
    think_buf     = ''
    vocab_size    = llm.n_vocab()
    logit_bias    = {k: v for k, v in _NO_THINK_LOGIT_BIAS.items() if k < vocab_size}

    print('[ai]  ', end='', flush=True)

    for token in llm.create_chat_completion(
        messages=messages, stream=True,
        max_tokens=max_tokens, temperature=0.3,
        logit_bias=logit_bias,
    ):
        delta = (token['choices'][0]['delta'].get('content') or '')
        if not delta:
            continue

        # ---- <think> タグのステートマシン ----
        processed = ''
        i = 0
        while i < len(delta):
            if in_think:
                think_buf += delta[i]
                if think_buf.endswith('</think>'):
                    in_think  = False
                    think_buf = ''
                i += 1
            else:
                # <think> の開始を検索
                rest = delta[i:]
                tag_pos = rest.find('<think>')
                if tag_pos == -1:
                    processed += rest
                    break
                else:
                    processed += rest[:tag_pos]
                    in_think  = True
                    think_buf = ''
                    i += tag_pos + len('<think>')

        if not processed:
            continue

        print(processed, end='', flush=True)
        full_response += processed
        buf           += processed

        # 文末で TTS dispatch
        sentences, buf = split_sentences(buf)
        for s in sentences:
            tts_q.put(s)

    if buf.strip():
        tts_q.put(buf.strip())

    print('', flush=True)
    return full_response


def split_sentences(text: str) -> tuple[list[str], str]:
    parts = SENTENCE_END_RE.split(text)
    seps  = SENTENCE_END_RE.findall(text)
    done  = [parts[i] + seps[i] for i in range(len(seps)) if (parts[i] + seps[i]).strip()]
    return done, (parts[-1] if parts else '')


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print('VOICEVOX 接続確認中...', flush=True)
    if not check_voicevox(args.voicevox_url):
        print(f'[ERROR] VOICEVOX が起動していません: {args.voicevox_url}')
        print('VOICEVOXアプリを起動してから再試行してください。')
        sys.exit(1)
    print(f'VOICEVOX OK  speaker_id={args.speaker_id}', flush=True)

    print('ASRモデルをロード中...', flush=True)
    from faster_whisper import WhisperModel
    asr_lang = None if args.language in ('None', 'auto') else args.language
    asr = WhisperModel(args.asr_model, device='cpu', compute_type='int8')

    print('LLMをロード中...', flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=args.llm_model, n_ctx=2048, verbose=False, n_threads=None)

    tts_q       = queue.Queue()
    is_speaking = threading.Event()

    tts_th = threading.Thread(target=tts_worker,
                              args=(tts_q, args.voicevox_url, args.speaker_id, is_speaking),
                              daemon=True)
    tts_th.start()

    # 起動アナウンス
    tts_q.put('準備ができました。話しかけてください。')
    time.sleep(0.5)
    while is_speaking.is_set():
        time.sleep(0.1)

    SAMPLE_RATE = 16000
    history: list[dict] = [{'role': 'system', 'content': SYSTEM_PROMPTS[args.mode]}]

    print(f'\n=== Voice Chat 開始  mode:{args.mode}  Ctrl+C で終了 ===\n', flush=True)

    try:
        while True:
            print('[mic] ...', end='', flush=True)
            audio = record_utterance(SAMPLE_RATE, args.silence_threshold,
                                     args.silence_duration, is_speaking)
            if audio is None:
                continue

            segs, _ = asr.transcribe(audio, language=asr_lang,
                                     vad_filter=True, beam_size=3)
            user_text = ''.join(s.text for s in segs).strip()
            if not user_text:
                continue

            print(f'\r[you] {user_text}', flush=True)
            history.append({'role': 'user', 'content': user_text})

            full_resp = stream_llm(llm, history, args.max_tokens, tts_q)
            history.append({'role': 'assistant', 'content': full_resp})

            # コンテキストを直近 10 ターンに制限
            if len(history) > 21:
                history = [history[0]] + history[-20:]

    except KeyboardInterrupt:
        print('\n終了します。', flush=True)
        tts_q.put('終了します。')
        time.sleep(3)
        tts_q.put(None)


if __name__ == '__main__':
    main()
