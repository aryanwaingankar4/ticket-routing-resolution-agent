"""Screenshots of the HTTP service and the Streamlit demo for the project report.

Needs Playwright (`pip install playwright` then `playwright install chromium`),
which is deliberately NOT in requirements.txt: it is only used to take these
pictures. Start both apps from the project root first:

    uvicorn src.service.api:app --port 8000
    streamlit run src/app/streamlit_app.py --server.headless true --server.port 8501

then run:

    python scripts/capture_black_book_screenshots.py

Writes PNGs to reports/black_book/fig/. The /triage call uses the API default
generate_resolution=false, so it spends no Gemini quota. The Streamlit demo
drafts a resolution for a ticket that clears the retrieval gate, so the
"proceed" screenshot spends ONE Gemini call; the escalated ticket spends none.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

from playwright.sync_api import sync_playwright

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(PROJECT_ROOT, "reports", "black_book", "fig")
API = "http://localhost:8000"
DEMO = "http://localhost:8501"

# The demo's own quick-fill examples (src/app/streamlit_app.py DEMO_EXAMPLES).
PROCEED_LABEL = os.environ.get("BB_PROCEED_EXAMPLE", "VPN issue")
ESCALATE_LABEL = "Unrelated (weather)"
TRIAGE_BODY = {
    "title": "Cannot connect to company VPN",
    "description": ("Since this morning I keep getting a timeout error when I try to "
                    "connect to the corporate VPN from home. The client says "
                    "'connection failed' after about 30 seconds. Restarting the client "
                    "and my router did not help."),
}


def triage_json() -> dict:
    req = urllib.request.Request(
        f"{API}/triage", data=json.dumps(TRIAGE_BODY).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=120))


def shot_docs(page):
    page.goto(f"{API}/docs", wait_until="networkidle")
    page.wait_for_selector(".opblock", timeout=60000)
    page.screenshot(path=os.path.join(OUT, "s_api_docs.png"),
                    clip={"x": 0, "y": 0, "width": 1280, "height": 1060})
    print("  wrote s_api_docs.png")


def shot_triage(page):
    result = triage_json()
    pretty = json.dumps(result, indent=2)

    # The figure shows a condensed copy: two of the five neighbours, long
    # strings shortened. The full response is saved beside it as JSON.
    def shorten(o):
        if isinstance(o, dict):
            return {k: shorten(v) for k, v in o.items()}
        if isinstance(o, list):
            return [shorten(v) for v in o]
        if isinstance(o, str) and len(o) > 70:
            return o[:67] + "..."
        return o
    view = json.loads(pretty)
    ret = view.get("result", {}).get("retrieval", {})
    for key in ("retrieved", "results", "neighbours"):
        if isinstance(ret.get(key), list) and len(ret[key]) > 2:
            n = len(ret[key])
            ret[key] = ret[key][:2] + [f"... {n - 2} more neighbours omitted in this figure"]
    lines = json.dumps(shorten(view), indent=2).splitlines()
    html = ("<html><body style='margin:0;background:#1e1e1e'>"
            "<div style='font:13px Consolas,monospace;color:#9cdcfe;padding:14px'>"
            "<div style='color:#ce9178;margin-bottom:8px'>POST /triage "
            "(generate_resolution=false) &rarr; HTTP 200</div><pre style='margin:0;"
            "color:#d4d4d4'>" + "\n".join(lines).replace("<", "&lt;") +
            "</pre></div></body></html>")
    page.set_content(html)
    page.screenshot(path=os.path.join(OUT, "s_triage_response.png"), full_page=True)
    with open(os.path.join(PROJECT_ROOT, "reports", "black_book", "data",
                           "triage_response_example.json"), "w", encoding="utf-8") as fh:
        fh.write(pretty)
    print("  wrote s_triage_response.png")


def shot_demo(page, label, name, wait_text):
    page.set_viewport_size({"width": 1280, "height": 2300})
    page.goto(DEMO, wait_until="networkidle")
    page.get_by_role("button", name=label).click(timeout=180000)
    page.wait_for_timeout(1500)
    page.get_by_role("button", name="Analyze Ticket").click()
    # innerText skips hidden elements (the "How this works" expander repeats
    # some of these phrases), so this waits for the visible result.
    page.wait_for_function("t => document.body.innerText.includes(t)",
                           arg=wait_text, timeout=240000)
    page.wait_for_timeout(3000)
    path = os.path.join(OUT, name)
    page.screenshot(path=path)
    autocrop_bottom(path)
    print(f"  wrote {name}")


def autocrop_bottom(path, margin=30):
    """Trim the blank space a tall viewport leaves under the last element."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    # Judge only the main column: Streamlit's sidebar runs the full height.
    main = im.crop((im.width // 4, 0, im.width, im.height))
    bg = Image.new("RGB", main.size, main.getpixel((main.width // 2, main.height - 1)))
    box = ImageChops.difference(main, bg).getbbox()
    if box:
        im.crop((0, 0, im.width, min(im.height, box[3] + margin))).save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        if not only or "api" in only:
            shot_docs(page)
            shot_triage(page)
        if not only or "demo" in only:
            shot_demo(page, ESCALATE_LABEL, "s_demo_escalated.png", "Escalated to a human agent")
            shot_demo(page, PROCEED_LABEL, "s_demo_proceed.png", "Suggested resolution")
        browser.close()


if __name__ == "__main__":
    main()
