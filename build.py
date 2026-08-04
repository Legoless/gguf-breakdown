#!/usr/bin/env python3
"""Build the standalone page from templates/ and data/.

    python build.py

The page ends up self-contained — no external requests, no build tooling, no
server. Open the HTML straight off disk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
DATA = ROOT / "data"

PAGES = [
    {"src": "index.html", "out": "index.html",
     "desc": "How a GGUF model file is loaded and run, and what is actually inside "
             "two real ones — Gemma 4 12B and Kimi K3.",
     "favicon": "📦", "data": "files.json"},
]

# Minimal reset. The artifact host supplied one; standalone pages need their own.
RESET = """*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0}
img,svg,video{max-width:100%;height:auto}
button{font:inherit;color:inherit}
table{border-collapse:collapse}"""



# The viewer's theme toggle was part of the artifact chrome. Locally the page
# follows the OS by default and this button overrides it.
THEME_CSS = """.themetoggle{position:fixed;top:14px;right:14px;z-index:99;
  font-family:var(--f-mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;
  background:var(--paper-2);color:var(--muted);border:1px solid var(--rule);
  border-radius:3px;padding:7px 11px;cursor:pointer;line-height:1;
  transition:color .18s,border-color .18s}
.themetoggle:hover{color:var(--ink);border-color:var(--muted)}
.themetoggle:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
@media print{.themetoggle{display:none}}"""

THEME_JS = """(function(){
  var KEY="gguf-breakdown-theme", root=document.documentElement;
  var btn=document.createElement("button");
  btn.className="themetoggle"; btn.type="button";
  function label(m){ return m==="auto" ? "auto" : m; }
  function apply(m){
    if(m==="auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme",m);
    btn.textContent=label(m);
    btn.setAttribute("aria-label","Colour theme: "+m+". Click to change.");
  }
  var mode=localStorage.getItem(KEY)||"auto";
  apply(mode);
  btn.addEventListener("click",function(){
    mode = mode==="auto" ? "light" : mode==="light" ? "dark" : "auto";
    localStorage.setItem(KEY,mode); apply(mode);
  });
  document.body.appendChild(btn);
})();"""


def favicon_tag(emoji: str) -> str:
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
           f"<text y='.9em' font-size='56'>{emoji}</text></svg>")
    from urllib.parse import quote
    return f'<link rel="icon" href="data:image/svg+xml,{quote(svg)}">'


def build_page(page: dict) -> Path:
    body = (TEMPLATES / page["src"]).read_text()

    if "data" in page:
        payload = (DATA / page["data"]).read_text()
        body = body.replace("__DATA__", payload)

    for leftover in re.findall(r"__[A-Z_]+__", body):
        raise SystemExit(f"{page['src']}: unreplaced placeholder {leftover}")

    # Templates open with <title>; lift it into <head> and leave the rest as body.
    m = re.search(r"<title>(.*?)</title>\s*", body, re.S)
    if not m:
        raise SystemExit(f"{page['src']}: no <title>")
    title, body = m.group(1).strip(), body[:m.start()] + body[m.end():]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{page['desc']}">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
{favicon_tag(page['favicon'])}
<style>{RESET}
{THEME_CSS}</style>
</head>
<body>
{body.strip()}
<script>{THEME_JS}</script>
</body>
</html>
"""
    out = ROOT / page["out"]
    out.write_text(html)
    return out


def main() -> None:
    for page in PAGES:
        out = build_page(page)
        print(f"  {out.name:<16} {out.stat().st_size / 1024:>7.1f} KB")
    for page in PAGES:
        print(f"  file://{ROOT / page['out']}")


if __name__ == "__main__":
    main()
