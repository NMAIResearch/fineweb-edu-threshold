"""Generate shared content-token summaries and the dataset card from checked inputs."""
import json
from pathlib import Path
import re


def main():
    metrics = json.loads(Path('content_metrics.json').read_text())
    r = metrics['populations']['retained']
    a = metrics['populations']['all']
    paragraph = (
        f"Among the {r['n']:,} retained documents in the {a['n']:,}-document tokenisation sample, "
        f"{r['truncated']:,} exceeded the {metrics['content_budget']}-token content budget, or "
        f"{r['truncated_pct']:.1f}%, N = {r['n']:,} retained documents. The measured prefixes contain "
        f"{r['visible_content_tokens']:,} of {r['content_tokens']:,} content tokens, or "
        f"{r['visible_pct']:.1f}%, across those same documents. The CLS and SEP special tokens are "
        "excluded from both token counts. This is a token-weighted sample estimate of prefix exposure, "
        "not a word count or a full-corpus census."
    )
    for name in ('README.md', 'article_draft_v2.md'):
        p = Path(name)
        text = p.read_text()
        pattern = r'Among the [\d,]+ retained documents in the [\d,]+-document tokenisation sample, .*?(?=\n\n)'
        text, count = re.subn(pattern, lambda _: paragraph, text, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f'{name}: unique main token summary required')
        p.write_text(text)
    header = '---\nlicense: cc-by-4.0\nlanguage:\n  - en\ntags:\n  - data-curation\n  - reproducibility\n  - fineweb\npretty_name: What Actually Admits a Document to FineWeb-Edu\n---\n\n'
    Path('README_hf.md').write_text(header + Path('README.md').read_text())


if __name__ == '__main__':
    main()
