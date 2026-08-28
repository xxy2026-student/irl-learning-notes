"""把 README.md 渲染成可直接在浏览器打开的 README.html。

    python notes/build_html.py

数学公式用 MathJax（SVG 输出，颜色随明暗主题走）。
转换前先把 $...$ / $$...$$ 摘出来占位，防止 markdown 把
数学里的下划线当斜体、把表格里的竖线当分栏；转完再回填。
依赖：pip install markdown
"""

import html
import pathlib
import re

import markdown

HERE = pathlib.Path(__file__).parent
SRC = HERE / "README.md"
OUT = HERE / "README.html"

SHELL = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IRL 五篇精读</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap">
<style>
:root{
  --bg:#FAFAF8; --ink:#1F2933; --muted:#5B6B68;
  --accent:#0F766E; --accent-ink:#0B5E58; --warn:#B45309; --warn-bg:#FBF4EA;
  --line:#D9E2E0; --quote-bg:#EEF4F2; --code-bg:#ECF0EF;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#101715; --ink:#E6EBE9; --muted:#8AA19C;
    --accent:#3AB5A8; --accent-ink:#5BCCC0; --warn:#E0A458; --warn-bg:#221B12;
    --line:#24312E; --quote-bg:#16201D; --code-bg:#1B2523;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Noto Sans SC",-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  font-size:15.5px;line-height:1.9;-webkit-font-smoothing:antialiased}
main{max-width:42em;margin:0 auto;padding:2.5rem 1.25rem 6rem}
header.doc{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:1.5rem}
.eyebrow{font-size:.72rem;letter-spacing:.14em;color:var(--accent);font-weight:700;text-transform:uppercase}
h1{font-family:"Noto Serif SC",serif;font-weight:700;font-size:1.55rem;line-height:1.5;margin:.4rem 0 .3rem;text-wrap:balance}
.meta{color:var(--muted);font-size:.85rem}
nav.toc{display:flex;flex-wrap:wrap;gap:.35rem .9rem;padding:.9rem 0 1.4rem;border-bottom:1px solid var(--line);margin-bottom:1rem}
nav.toc a{color:var(--muted);text-decoration:none;font-size:.82rem}
nav.toc a:hover{color:var(--accent)}
h2{font-family:"Noto Serif SC",serif;font-weight:700;font-size:1.22rem;line-height:1.55;
  margin:2.6em 0 .6em;padding-top:1.2em;border-top:1px solid var(--line);text-wrap:balance}
h3{font-family:"Noto Serif SC",serif;font-weight:600;font-size:1.02rem;margin:1.9em 0 .4em}
a{color:var(--accent-ink);text-decoration-thickness:1px;text-underline-offset:3px}
hr{border:0;border-top:1px solid var(--line);margin:2.4em 0}
blockquote{margin:1.3em 0;padding:.7em 1.1em;background:var(--quote-bg);
  border-left:3px solid var(--accent);border-radius:0 6px 6px 0}
blockquote p{margin:.3em 0}
blockquote.warn{background:var(--warn-bg);border-left-color:var(--warn)}
code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:.86em;
  background:var(--code-bg);padding:.12em .38em;border-radius:4px}
pre{background:var(--code-bg);padding:.9em 1.1em;border-radius:8px;overflow-x:auto;line-height:1.65}
pre code{background:none;padding:0}
.tw{overflow-x:auto;margin:1.3em 0}
table{border-collapse:collapse;font-size:.88rem;line-height:1.7;min-width:100%}
th{font-weight:700;text-align:left;border-bottom:2px solid var(--line);padding:.45em .8em .45em 0;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:.5em .8em .5em 0;vertical-align:top}
mjx-container[display="true"]{display:block;overflow-x:auto;overflow-y:hidden;padding:.35em 0;margin:.9em 0}
li{margin:.3em 0}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
</style>
</head>
<body>
<main>
<header class="doc">
  <div class="eyebrow">irl-learning-notes / notes</div>
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</header>
<nav class="toc">__TOC__</nav>
__BODY__
</main>
<script>
window.MathJax={tex:{inlineMath:[['$','$']],displayMath:[['$$','$$']]},svg:{fontCache:'global'}};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js"></script>
</body>
</html>
"""


def build():
    src = SRC.read_text()

    # 1. 摘出数学，占位保护（纯字母数字占位符，markdown 不会动它）
    store = []

    def stash(m):
        store.append(m.group(0))
        return f"QQMATH{len(store) - 1}ZZ"

    body = re.sub(r"\$\$.*?\$\$", stash, src, flags=re.S)
    body = re.sub(r"\$[^\$\n]+?\$", stash, body)

    # 2. markdown -> html
    body = markdown.markdown(body, extensions=["tables", "fenced_code"])

    # 3. 回填数学（转义 & < >，MathJax 在文本层读回原样）
    body = re.sub(
        r"QQMATH(\d+)ZZ",
        lambda m: html.escape(store[int(m.group(1))], quote=False),
        body,
    )
    assert not re.search(r"QQMATH\d+ZZ", body), "有数学占位符没回填"

    # 4. 结构后处理
    title_m = re.search(r"<h1>(.*?)</h1>", body)
    title = title_m.group(1) if title_m else SRC.name
    body = re.sub(r"<h1>.*?</h1>\s*", "", body, count=1)

    secs = []

    def h2id(m):
        i = f"s{len(secs)}"
        label = re.sub(r"<[^>]+>", "", m.group(1))
        label = re.sub(r"\$[^$]*\$", "…", label).strip().rstrip("：: ")
        secs.append((i, label))
        return f'<h2 id="{i}">{m.group(1)}</h2>'

    body = re.sub(r"<h2>(.*?)</h2>", h2id, body)
    body = body.replace("<blockquote>\n<p>⚠️", '<blockquote class="warn">\n<p>⚠️')
    body = re.sub(r"<table>", '<div class="tw"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)

    toc = "\n".join(f'<a href="#{i}">{t}</a>' for i, t in secs)
    meta = f"由 <code>build_html.py</code> 从 <code>{SRC.name}</code> 生成 —— 改笔记后重跑一次即可同步"
    page = (
        SHELL.replace("__TITLE__", title)
        .replace("__META__", meta)
        .replace("__TOC__", toc)
        .replace("__BODY__", body)
    )
    OUT.write_text(page)
    print(f"{OUT.name}: {len(page) / 1024:.0f} KB, {len(store)} 个公式, {len(secs)} 节")


if __name__ == "__main__":
    build()
