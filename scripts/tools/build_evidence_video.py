"""Build a narrated OtoWeave evidence video from verified project facts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "video"
FRAME_DIR = OUT / "frames_v2"
AUDIO_DIR = OUT / "audio_v2"
SEGMENT_DIR = OUT / "segments_v2"
SCREENSHOT = ROOT / "docs" / "images" / "otoweave-main.png"
MECHANISM_IMAGE = ROOT / "docs" / "images" / "otoweave-mechanism.png"
WIDTH, HEIGHT = 1920, 1080
FONT_PATH = Path(r"C:\Windows\Fonts\NotoSansJP-VF.ttf")

BG = "#F5F7FB"
NAVY = "#183153"
BLUE = "#3169B3"
CYAN = "#DCEBFA"
ORANGE = "#E87B35"
RED = "#C83E4D"
GREEN = "#237A57"
GRAY = "#52606D"
WHITE = "#FFFFFF"


SCENES = [
    {
        "title": "OtoWeave",
        "eyebrow": "MY PROJECT / EVIDENCE-BASED STORY",
        "body": ["書くのが苦手でも、", "聞いて理解することはできる。"],
        "note": "読み書きの負担を減らす、オフラインAIノートアプリ",
        "narration": "書くのが苦手でも、聞いて理解することはできる。OtoWeaveは、読み書きに困難のある人が、授業や面談で聞くことに集中できるようにする、オフラインAIノートアプリです。",
        "kind": "title",
    },
    {
        "title": "出発点となった課題",
        "eyebrow": "PROBLEM",
        "body": [
            "聞きながら書くことが難しい",
            "書いたメモを後から読めないことがある",
            "長い文字起こしを読むこと自体が負担",
        ],
        "note": "※ 現時点では開発の出発点。利用後の負担軽減は未測定",
        "narration": "出発点は、話を聞きながらノートを書く難しさです。書いた字が後で読めなかったり、文字起こしがあっても長文を読むこと自体が負担になります。ただし、アプリによって負担が減ったという利用者の前後比較は、まだ取得していません。これは今後検証する課題です。",
        "kind": "bullets",
    },
    {
        "title": "最初の失敗",
        "eyebrow": "FAILURE",
        "body": [
            "Whisper medium：10分音声に 954.2秒",
            "Moonshine：『フロリダ』→『プロレ』",
            "固定30秒分割：『テキストスピーチ』→『ペックストスピーチ』",
        ],
        "note": "同じ日本語面談音声の先頭10分・CPU・並列処理なし",
        "narration": "まず複数の音声認識モデルを、同じ日本語面談音声で比較しました。Whisper mediumは精度が高めでも、10分の音声に約954秒かかりました。軽いMoonshineは、フロリダをプロレと誤認識。ReazonSpeechも固定30秒で切ると、テキストスピーチを誤認識しました。",
        "kind": "bullets",
    },
    {
        "title": "仮説と開発",
        "eyebrow": "HYPOTHESIS → DEVELOPMENT",
        "body": [
            "日本語特化ASRを使う",
            "固定時間ではなく、自然な無音位置で分割する",
            "日本語と英語で、得意なモデルを切り替える",
        ],
        "note": "ReazonSpeech K2 + VAD / 英語候補区間のみ別ASR",
        "narration": "そこで、日本語特化の軽量モデルを使い、固定時間ではなく自然な無音位置で音声を分割すればよいと考えました。また、一つのモデルにすべてを任せず、日本語と英語で得意なモデルを切り替える仕組みを開発しました。",
        "kind": "flow",
    },
    {
        "title": "OtoWeaveのしくみ",
        "eyebrow": "HOW IT WORKS",
        "body": [],
        "note": "",
        "narration": "OtoWeaveでは、授業や面談の音声を取り込み、日本語と英語に適したAIで文字起こしします。次に、誰が話したかを推定して色分けし、ローカルのAIが要約します。最後に、要約や文字起こしを音声で読み上げたり、元の録音を聞き直したりできます。この処理はすべてパソコンの中で行い、音声、文字、要約を外部へ送信しません。",
        "kind": "mechanism",
    },
    {
        "title": "検証結果",
        "eyebrow": "VERIFICATION",
        "body": ["34.8秒", "10分の日本語面談音声", "71分の音声も 238.7秒・失敗0"],
        "note": "ReazonSpeech K2 v2 + 無音位置VAD / 最大RAM 876MB（10分）",
        "narration": "検証の結果、ReazonSpeechと無音位置VADの組み合わせは、10分の音声を34.8秒で処理しました。71分の音声も238.7秒、167チャンクすべて失敗ゼロでした。固定30秒分割で誤認識したテキストスピーチも、正しく認識できました。",
        "kind": "metric",
    },
    {
        "title": "AI要約も失敗した",
        "eyebrow": "FAILURE → SAFEGUARDS",
        "body": [
            "入力にない『生徒会』『集会』を追加",
            "本人の応募理由・体験・希望を欠落",
            "略語を誤って展開",
        ],
        "note": "2段階要約・抽出的fallback・既知語補正・β表示を実装",
        "narration": "AI要約も、最初から正しく動いたわけではありません。入力にない生徒会や集会を追加し、本人の応募理由や体験を落とすことがありました。そこで、二段階要約、元文からの抽出的な補完、既知語の補正、そしてベータ版表示と確認の品質ゲートを実装しました。",
        "kind": "bullets",
    },
    {
        "title": "現在のOtoWeave",
        "eyebrow": "PROTOTYPE",
        "body": [
            "録音・文字起こし・話者分離",
            "要約・AIチューター・読み上げ",
            "すべてローカルPC内で処理",
        ],
        "note": "音声・文字起こしをクラウドAI APIへ送信しない設計",
        "narration": "現在のOtoWeaveは、録音、文字起こし、話者分離、要約、AIチューター、読み上げまでを一つにまとめています。すべてローカルPC内で処理し、音声や文字起こしをクラウドAI APIへ送信しない設計です。",
        "kind": "screenshot",
    },
    {
        "title": "次は、実際の利用場面で検証する",
        "eyebrow": "NEXT VERIFICATION",
        "body": [
            "ノートを書く負担",
            "聞くことへの集中",
            "要点を確認する時間",
            "記録を読み返す負担",
        ],
        "note": "未取得の結果は作らない。試用前後を同じ指標で測る。",
        "narration": "次の課題は、実際の利用場面での検証です。ノートを書く負担、聞くことへの集中、要点を確認する時間、読み返す負担を、同じ利用者の試用前後で測ります。未取得の結果は作らず、証拠に戻れる形で改善を続けます。",
        "kind": "bullets",
    },
]


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def rounded(draw: ImageDraw.ImageDraw, box, fill, radius=28, outline=None, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_text(draw, text, max_width, start_size, min_size=26):
    for size in range(start_size, min_size - 1, -2):
        fnt = font(size)
        if draw.textbbox((0, 0), text, font=fnt)[2] <= max_width:
            return fnt
    return font(min_size)


def draw_header(draw, scene, number):
    draw.text((110, 70), scene["eyebrow"], font=font(27), fill=BLUE)
    draw.text((1810, 70), f"{number:02d} / {len(SCENES):02d}", font=font(23), fill=GRAY, anchor="ra")
    title_font = fit_text(draw, scene["title"], 1680, 66)
    draw.text((110, 125), scene["title"], font=title_font, fill=NAVY)
    draw.line((110, 225, 1810, 225), fill="#CAD5E2", width=3)


def draw_frame(scene, number):
    if scene["kind"] == "mechanism":
        source = Image.open(MECHANISM_IMAGE).convert("RGB")
        source.thumbnail((WIDTH, HEIGHT))
        img = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        img.paste(source, ((WIDTH - source.width) // 2, (HEIGHT - source.height) // 2))
        draw = ImageDraw.Draw(img)
        draw.text((1860, 35), f"{number:02d} / {len(SCENES):02d}", font=font(21), fill=GRAY, anchor="ra")
        return img

    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw_header(draw, scene, number)
    kind = scene["kind"]
    body = scene["body"]

    if kind == "title":
        rounded(draw, (110, 285, 1810, 890), WHITE, 40, "#D8E1EC")
        draw.text((170, 370), body[0], font=font(76), fill=NAVY)
        draw.text((170, 490), body[1], font=font(76), fill=BLUE)
        draw.text((170, 690), scene["note"], font=font(36), fill=GRAY)
        draw.ellipse((1500, 380, 1710, 590), fill=CYAN)
        draw.arc((1540, 420, 1670, 550), 205, 335, fill=BLUE, width=16)
        draw.arc((1570, 445, 1640, 520), 205, 335, fill=ORANGE, width=12)
    elif kind == "metric":
        rounded(draw, (110, 285, 1810, 870), WHITE, 40, "#D8E1EC")
        draw.text((960, 355), body[0], font=font(150), fill=GREEN, anchor="ma")
        draw.text((960, 560), body[1], font=font(42), fill=NAVY, anchor="ma")
        rounded(draw, (440, 655, 1480, 770), CYAN, 24)
        draw.text((960, 712), body[2], font=font(38), fill=BLUE, anchor="mm")
        draw.text((110, 940), scene["note"], font=font(25), fill=GRAY)
    elif kind == "flow":
        colors = ["#E6F0FB", "#EAF6F0", "#FFF0E6"]
        for idx, item in enumerate(body):
            x1 = 110 + idx * 575
            x2 = x1 + 500
            rounded(draw, (x1, 350, x2, 700), colors[idx], 34)
            lines = textwrap.wrap(item, width=14)
            y = 455
            for line in lines:
                draw.text(((x1 + x2) // 2, y), line, font=font(38), fill=NAVY, anchor="ma")
                y += 62
            if idx < 2:
                draw.polygon([(x2 + 25, 505), (x2 + 70, 540), (x2 + 25, 575)], fill=ORANGE)
        draw.text((110, 940), scene["note"], font=font(27), fill=GRAY)
    elif kind == "screenshot":
        shot = Image.open(SCREENSHOT).convert("RGB")
        shot.thumbnail((1050, 650))
        x, y = 760, 280
        img.paste(shot, (x, y))
        draw.rounded_rectangle((x - 4, y - 4, x + shot.width + 4, y + shot.height + 4), radius=12, outline="#AEBCCB", width=4)
        for idx, item in enumerate(body):
            yb = 340 + idx * 145
            rounded(draw, (110, yb, 690, yb + 105), WHITE, 20, "#D8E1EC")
            draw.ellipse((145, yb + 34, 175, yb + 64), fill=[BLUE, ORANGE, GREEN][idx])
            draw.text((205, yb + 52), item, font=font(30), fill=NAVY, anchor="lm")
        draw.text((110, 965), scene["note"], font=font(24), fill=GRAY)
    else:
        for idx, item in enumerate(body):
            y = 305 + idx * 145
            rounded(draw, (110, y, 1810, y + 105), WHITE, 22, "#D8E1EC")
            draw.ellipse((150, y + 34, 183, y + 67), fill=RED if number in (3, 6) else BLUE)
            draw.text((225, y + 52), item, font=fit_text(draw, item, 1490, 37), fill=NAVY, anchor="lm")
        draw.text((110, 940), scene["note"], font=font(25), fill=RED if "未" in scene["note"] else GRAY)

    # Persistent lower-right label keeps the evidence source visible.
    draw.text((1810, 1025), "Source: マイプロジェクト証拠台帳", font=font(20), fill="#788896", anchor="ra")
    return img


def synthesize(text_path: Path, wav_path: Path):
    command = f"""
Add-Type -AssemblyName System.Speech
$text = [IO.File]::ReadAllText('{str(text_path).replace("'", "''")}', [Text.Encoding]::UTF8)
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.Rate = -1
$voice.Volume = 100
$voice.SetOutputToWaveFile('{str(wav_path).replace("'", "''")}')
$voice.Speak($text)
$voice.Dispose()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True)


def main():
    for directory in (OUT, FRAME_DIR, AUDIO_DIR, SEGMENT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path.cwd() / ".video_tools"))
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    concat_lines = []
    narration_lines = ["# OtoWeave 証拠動画 ナレーション台本", ""]

    for index, scene in enumerate(SCENES, 1):
        frame = FRAME_DIR / f"scene_{index:02d}.png"
        text_file = AUDIO_DIR / f"scene_{index:02d}.txt"
        wav = AUDIO_DIR / f"scene_{index:02d}.wav"
        segment = SEGMENT_DIR / f"scene_{index:02d}.mp4"
        # Keep valid intermediate scenes so an interrupted build resumes quickly.
        if not segment.exists() or segment.stat().st_size < 10_000:
            draw_frame(scene, index).save(frame)
            text_file.write_text(scene["narration"], encoding="utf-8")
            synthesize(text_file, wav)
            subprocess.run(
                [
                    ffmpeg, "-y", "-loop", "1", "-framerate", "30", "-i", str(frame),
                    "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
                    "-movflags", "+faststart", str(segment),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        concat_lines.append(f"file '{segment.as_posix()}'")
        narration_lines.extend([f"## {index}. {scene['title']}", "", scene["narration"], ""])

    concat_file = SEGMENT_DIR / "concat.txt"
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    output_video = OUT / "OtoWeave_証拠台帳_紹介動画.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(output_video)],
        check=True,
    )
    (OUT / "ナレーション台本.md").write_text("\n".join(narration_lines), encoding="utf-8")
    print(output_video)


if __name__ == "__main__":
    main()
