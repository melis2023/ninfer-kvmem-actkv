"""Regenerate the README diagram with Python 3.11; no third-party dependencies."""
from pathlib import Path
from html import escape

out = Path(__file__).resolve().parent
arches = [
    ('sm_70', 'Volta', 'Tesla V100'),
    ('sm_75', 'Turing', 'RTX 20'),
    ('sm_80', 'Ampere', 'A100'),
    ('sm_86', 'Ampere', 'RTX 30'),
    ('sm_89', 'Ada', 'RTX 40'),
    ('sm_120a', 'Blackwell', 'RTX 50 · RTX PRO 6000'),
]
svg = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="430" viewBox="0 0 1440 430" role="img" aria-labelledby="title desc">
<title id="title">GPU architectures in the two NInfer build variants</title>
<desc id="desc">CUDA 12.9.2: sm_70, sm_75, sm_80, sm_86, sm_89. CUDA 13.4.3: sm_75, sm_80, sm_86, sm_89, sm_120a. Shared architectures lie inside both outlines.</desc>
<rect width="1440" height="430" fill="#ffffff"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#18232f">
<text x="48" y="51" font-size="28" font-weight="700">GPU architectures in NInfer builds</text>
<rect x="272" y="144" width="896" height="164" rx="12" fill="#eef5fa"/>
<rect x="36" y="112" width="1132" height="196" rx="24" fill="none" stroke="#cb7a14" stroke-width="3"/>
<rect x="272" y="144" width="1132" height="196" rx="24" fill="none" stroke="#2670c5" stroke-width="3"/>
<rect x="63" y="95" width="212" height="34" fill="white"/>
<text x="77" y="121" font-size="26" font-weight="700" fill="#aa6006">CUDA 12.9.2</text>
<rect x="1169" y="322" width="211" height="34" fill="white"/>
<text x="1182" y="349" font-size="26" font-weight="700" fill="#2162ae">CUDA 13.4.3</text>''']
for i, (sm, family, model) in enumerate(arches):
    x = 154 + 224 * i
    svg.extend([
        f'<text x="{x}" y="211" text-anchor="middle" font-size="31" font-weight="700">{sm}</text>',
        f'<text x="{x}" y="246" text-anchor="middle" font-size="23">{family}</text>',
        f'<text x="{x}" y="278" text-anchor="middle" font-size="{16 if i == 5 else 19}" fill="#566575">{escape(model)}</text>',
    ])
svg.extend(['''<text x="720" y="391" text-anchor="middle" font-size="19" fill="#566575">Shared architectures: sm_75 · sm_80 · sm_86 · sm_89</text>
</g></svg>'''])
(out / 'cuda-architectures.svg').write_text('\n'.join(svg))
