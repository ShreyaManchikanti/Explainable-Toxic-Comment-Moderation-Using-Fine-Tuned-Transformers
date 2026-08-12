import json
from pathlib import Path

import gradio as gr
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())

BEST_MODEL_NAME = metrics.get(
    "best_model", "bert" if metrics["bert"]["macro_f1"] >= metrics["roberta"]["macro_f1"] else "roberta"
)
THRESHOLDS = metrics.get("tuned_thresholds", {label: 0.5 for label in LABEL_COLS})
MODEL_DIR = str(MODELS_DIR / BEST_MODEL_NAME)

print(f"Loading {BEST_MODEL_NAME} from {MODEL_DIR} ...")
print("Using per-label thresholds:", THRESHOLDS)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def predict_fn(texts):
    enc = tokenizer(list(texts), truncation=True, max_length=128, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.sigmoid(logits).numpy()


masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_fn, masker, output_names=LABEL_COLS)

SCENARIOS = {
    "1. Clean comment": {
        "text": "Thanks so much for cleaning up this article, it reads much better now.",
        "note": "**What to look for:** every label stays clear. The model doesn't cry "
                "wolf on ordinary, positive text.",
    },
    "2. Ordinary disagreement": {
        "text": "I disagree with your edit, but let's discuss it on the talk page.",
        "note": "**What to look for:** still all clear. Disagreement and criticism aren't "
                "confused with toxicity - important for not over-moderating normal debate.",
    },
    "3. Clear-cut insult": {
        "text": "You are a complete idiot and everyone hates you.",
        "note": "**What to look for:** `toxic`, `insult` and `obscene` are correctly "
                "FLAGGED, but `threat` and `identity_hate` correctly stay clear. It's "
                "distinguishing *what kind* of toxicity this is, not just firing on every "
                "label at once.",
    },
    "4. Rare-label blind spot (threat)": {
        "text": "I will find you and hurt you if you post here again.",
        "note": "**What to look for:** `toxic` fires strongly, but `threat` itself stays "
                "under its tuned threshold. Notebook 04 traces this to only ~137 `threat` "
                "examples in the training subsample - an honest limitation, not a bug.",
    },
}


def build_verdict_html(label_scores):
    flagged_labels = [label for label in LABEL_COLS if label_scores[label] >= THRESHOLDS.get(label, 0.5)]
    if flagged_labels:
        pretty = "".join(f'<span class="verdict-chip">{l}</span>' for l in flagged_labels)
        return f"""
        <div class="verdict-card verdict-flagged">
          <div class="verdict-icon">🚩</div>
          <div class="verdict-body">
            <div class="verdict-title">Flagged</div>
            <div class="verdict-sub">Threshold crossed on &mdash; {pretty}</div>
          </div>
        </div>
        """
    return """
    <div class="verdict-card verdict-clean">
      <div class="verdict-icon">✅</div>
      <div class="verdict-body">
        <div class="verdict-title">Looks clean</div>
        <div class="verdict-sub">Nothing crosses its tuned threshold</div>
      </div>
    </div>
    """


def build_labels_html(label_scores):
    cards_html = []
    for label in sorted(LABEL_COLS, key=lambda l: label_scores[l], reverse=True):
        prob = label_scores[label]
        threshold = THRESHOLDS.get(label, 0.5)
        flagged = prob >= threshold
        state = "flagged" if flagged else "clear"
        badge_text = "FLAGGED" if flagged else "clear"
        bar_width = max(1.5, round(prob * 100, 1))
        threshold_pct = max(0, min(100, round(threshold * 100, 1)))
        cards_html.append(f"""
        <div class="label-card label-card--{state}">
          <div class="label-card__head">
            <span class="label-card__name">{label.replace('_', ' ')}</span>
            <span class="label-card__badge label-card__badge--{state}">{badge_text}</span>
          </div>
          <div class="label-card__meter">
            <div class="label-card__fill label-card__fill--{state}" style="width:{bar_width}%;"></div>
            <div class="label-card__tick" style="left:{threshold_pct}%;" title="tuned threshold {threshold * 100:.0f}%"></div>
          </div>
          <div class="label-card__foot">
            <span class="label-card__prob">{prob * 100:.1f}%</span>
            <span class="label-card__threshold">threshold {threshold * 100:.0f}%</span>
          </div>
        </div>
        """)
    return f"""
    <div class="label-grid">
      {''.join(cards_html)}
    </div>
    """


def moderate_comment(comment_text):
    if not comment_text or not comment_text.strip():
        return EMPTY_MSG, "", ""

    probs = predict_fn([comment_text])[0]
    label_scores = {label: float(p) for label, p in zip(LABEL_COLS, probs)}
    verdict_html = build_verdict_html(label_scores)
    labels_html = build_labels_html(label_scores)

    shap_values = explainer([comment_text], max_evals=300)
    shap_html = _wrap_shap(shap.plots.text(shap_values[0, :, "toxic"], display=False))

    return verdict_html, labels_html, shap_html


def load_scenario(scenario_name):
    scenario = SCENARIOS[scenario_name]
    verdict_html, labels_html, shap_html = moderate_comment(scenario["text"])
    return scenario["text"], scenario["note"], verdict_html, labels_html, shap_html


SHIELD_SVG = """
<svg width="96" height="96" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"
     style="filter: drop-shadow(0 0 18px rgba(124,143,255,0.55));">
  <path d="M60 8 L104 24 V54 C104 86 84 108 60 116 C36 108 16 86 16 54 V24 Z"
        fill="rgba(255,255,255,0.15)" stroke="white" stroke-width="3"/>
  <path d="M40 60 L54 74 L82 44" stroke="white" stroke-width="7"
        stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>
"""

_DOT_PATTERN_B64 = (
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNiIgaGVpZ2h0"
    "PSIyNiI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjEuNCIgZmlsbD0icmdiYSgyNTUsMjU1LDI1"
    "NSwwLjE2KSIvPjwvc3ZnPg=="
)

_INK = "#e7e9f7"
_INK_MUTED = "#93a0c4"
_PANEL = "rgba(20,22,38,0.55)"
_PANEL_BORDER = "rgba(255,255,255,0.09)"
_ACCENT_GRADIENT = "linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%)"
_ACCENT_GRADIENT_HOVER = "linear-gradient(135deg, #8b5cf6 0%, #22d3ee 100%)"
_ACCENT_GLOW = "0 0 24px -4px rgba(139,92,246,0.65)"
_ACCENT_GLOW_HOVER = "0 0 32px -2px rgba(34,211,238,0.65)"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    font=["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
).set(
    body_background_fill="#080a14",
    body_background_fill_dark="#080a14",
    body_text_color=_INK,
    body_text_color_dark=_INK,
    body_text_color_subdued=_INK_MUTED,
    body_text_color_subdued_dark=_INK_MUTED,
    background_fill_primary="#0d0f1c",
    background_fill_primary_dark="#0d0f1c",
    background_fill_secondary="#11132420",
    background_fill_secondary_dark="#11132420",
    border_color_accent="#8b5cf6",
    border_color_accent_dark="#8b5cf6",
    border_color_primary=_PANEL_BORDER,
    border_color_primary_dark=_PANEL_BORDER,
    color_accent="#8b5cf6",
    color_accent_soft="rgba(139,92,246,0.15)",
    color_accent_soft_dark="rgba(139,92,246,0.15)",
    link_text_color="#22d3ee",
    link_text_color_dark="#22d3ee",
    shadow_drop="none",
    shadow_drop_lg="0 20px 60px rgba(0,0,0,0.45)",
    block_background_fill=_PANEL,
    block_background_fill_dark=_PANEL,
    block_border_color=_PANEL_BORDER,
    block_border_color_dark=_PANEL_BORDER,
    block_border_width="1px",
    block_label_background_fill="rgba(139,92,246,0.16)",
    block_label_background_fill_dark="rgba(139,92,246,0.16)",
    block_label_text_color="#c4b5fd",
    block_label_text_color_dark="#c4b5fd",
    block_label_text_weight="600",
    block_label_border_width="0px",
    block_label_radius="8px",
    block_title_text_color=_INK,
    block_title_text_color_dark=_INK,
    block_title_text_weight="600",
    block_shadow="0 8px 32px rgba(0,0,0,0.35)",
    block_shadow_dark="0 8px 32px rgba(0,0,0,0.35)",
    block_radius="18px",
    panel_background_fill=_PANEL,
    panel_background_fill_dark=_PANEL,
    panel_border_color=_PANEL_BORDER,
    panel_border_color_dark=_PANEL_BORDER,
    input_background_fill="rgba(255,255,255,0.04)",
    input_background_fill_dark="rgba(255,255,255,0.04)",
    input_background_fill_focus="rgba(255,255,255,0.06)",
    input_background_fill_focus_dark="rgba(255,255,255,0.06)",
    input_border_color=_PANEL_BORDER,
    input_border_color_dark=_PANEL_BORDER,
    input_border_color_focus="#22d3ee",
    input_border_color_focus_dark="#22d3ee",
    input_shadow_focus="0 0 0 3px rgba(34,211,238,0.25)",
    input_shadow_focus_dark="0 0 0 3px rgba(34,211,238,0.25)",
    input_placeholder_color=_INK_MUTED,
    input_placeholder_color_dark=_INK_MUTED,
    input_radius="12px",
    button_large_radius="12px",
    button_medium_radius="10px",
    button_border_width="0px",
    button_border_width_dark="0px",
    button_transition="all 0.18s ease",
    button_primary_background_fill=_ACCENT_GRADIENT,
    button_primary_background_fill_dark=_ACCENT_GRADIENT,
    button_primary_background_fill_hover=_ACCENT_GRADIENT_HOVER,
    button_primary_background_fill_hover_dark=_ACCENT_GRADIENT_HOVER,
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_primary_shadow=_ACCENT_GLOW,
    button_primary_shadow_dark=_ACCENT_GLOW,
    button_primary_shadow_hover=_ACCENT_GLOW_HOVER,
    button_primary_shadow_hover_dark=_ACCENT_GLOW_HOVER,
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_background_fill_dark="rgba(255,255,255,0.06)",
    button_secondary_background_fill_hover="rgba(255,255,255,0.12)",
    button_secondary_background_fill_hover_dark="rgba(255,255,255,0.12)",
    button_secondary_text_color=_INK,
    button_secondary_text_color_dark=_INK,
    button_secondary_border_color=_PANEL_BORDER,
    button_secondary_border_color_dark=_PANEL_BORDER,
    checkbox_label_background_fill="rgba(255,255,255,0.04)",
    checkbox_label_background_fill_dark="rgba(255,255,255,0.04)",
    checkbox_label_background_fill_hover="rgba(255,255,255,0.09)",
    checkbox_label_background_fill_hover_dark="rgba(255,255,255,0.09)",
    checkbox_label_background_fill_selected=_ACCENT_GRADIENT,
    checkbox_label_background_fill_selected_dark=_ACCENT_GRADIENT,
    checkbox_label_text_color=_INK,
    checkbox_label_text_color_dark=_INK,
    checkbox_label_text_color_selected="white",
    checkbox_label_text_color_selected_dark="white",
    checkbox_label_border_color=_PANEL_BORDER,
    checkbox_label_border_color_dark=_PANEL_BORDER,
    checkbox_label_border_color_selected="#8b5cf6",
    checkbox_label_border_color_selected_dark="#8b5cf6",
    checkbox_label_shadow="none",
    checkbox_border_color_selected="#8b5cf6",
    checkbox_border_color_selected_dark="#8b5cf6",
    checkbox_background_color_selected="#8b5cf6",
    checkbox_background_color_selected_dark="#8b5cf6",
    slider_color="#22d3ee",
    slider_color_dark="#22d3ee",
)

APP_CSS = f"""
html, body {{
    background: #080a14 !important;
}}
.gradio-container {{
    position: relative;
    background:
        radial-gradient(circle at 15% 8%, rgba(124,58,237,0.16), transparent 38%),
        radial-gradient(circle at 85% 6%, rgba(6,182,212,0.14), transparent 40%),
        radial-gradient(circle at 50% 95%, rgba(236,72,153,0.10), transparent 45%),
        #080a14 !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}}
.gradio-container::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 90%);
}}
* {{ scrollbar-color: rgba(139,92,246,0.55) transparent; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: rgba(139,92,246,0.45); border-radius: 999px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

.glass-panel {{
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    padding: 22px !important;
}}

#landing-hero {{
    position: relative;
    background:
        url("data:image/svg+xml;base64,{_DOT_PATTERN_B64}") repeat,
        radial-gradient(circle at 12% 15%, rgba(76,110,245,0.65), transparent 42%),
        radial-gradient(circle at 88% 12%, rgba(194,37,92,0.55), transparent 45%),
        radial-gradient(circle at 55% 100%, rgba(112,72,232,0.55), transparent 55%),
        linear-gradient(135deg, #14152b 0%, #201233 55%, #24102c 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 72px 32px 56px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.45), 0 0 90px -30px rgba(139,92,246,0.6);
    animation: fadeSlideIn 0.5s ease;
}}
#landing-hero h1 {{
    font-size: 2.6em; margin: 0.4em 0 0.15em; letter-spacing: -0.02em; font-weight: 800;
    background: linear-gradient(90deg, #ffffff 20%, #c4b5fd 60%, #67e8f9 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
#landing-hero p.subtitle {{ font-size: 1.12em; opacity: 0.85; max-width: 560px; margin: 0 auto; }}
.badge-row {{ display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; margin-top: 28px; }}
.badge-pill {{
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.22);
    border-radius: 999px; padding: 7px 16px; font-size: 0.9em; white-space: nowrap;
    backdrop-filter: blur(6px);
}}
#enter-btn {{ font-size: 1.05em !important; padding: 12px 30px !important; letter-spacing: 0.01em; }}

#analyse-btn:disabled {{
    opacity: 0.55 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
    filter: grayscale(35%);
    animation: pulseGlow 1.4s ease-in-out infinite;
}}

#info-bar {{
    position: relative;
    background: linear-gradient(120deg, rgba(124,58,237,0.16), rgba(6,182,212,0.10));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}
#info-bar h1 {{
    font-size: 1.7em; margin: 0 0 6px; font-weight: 800; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #ffffff, #c4b5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
#info-bar p {{ color: #b7c0e0; margin: 0; font-size: 0.98em; max-width: 780px; line-height: 1.5; }}
#info-bar .info-pills {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }}
#info-bar .info-pill {{
    font-size: 0.8em; font-weight: 600; padding: 5px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14); color: #dbe1fb;
}}
#info-bar .info-pill code {{ background: none; color: #67e8f9; }}

.section-title h3 {{
    font-size: 1.05em !important; font-weight: 700 !important; letter-spacing: 0.01em;
    background: linear-gradient(90deg, #67e8f9, #c4b5fd);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin-bottom: 4px !important;
}}

.verdict-card {{
    display: flex; align-items: center; gap: 16px;
    padding: 18px 22px; border-radius: 16px; position: relative;
    animation: fadeSlideIn 0.35s ease;
}}
.verdict-flagged {{
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(249,115,22,0.10));
    border: 1px solid rgba(248,113,113,0.4);
    box-shadow: 0 0 34px -8px rgba(239,68,68,0.6);
}}
.verdict-clean {{
    background: linear-gradient(135deg, rgba(34,211,238,0.14), rgba(52,211,153,0.10));
    border: 1px solid rgba(45,212,191,0.4);
    box-shadow: 0 0 34px -8px rgba(45,212,191,0.5);
}}
.verdict-icon {{ font-size: 2em; line-height: 1; animation: pulseGlow 1.8s ease-in-out infinite; }}
.verdict-title {{ font-size: 1.2em; font-weight: 800; color: #f4f5fc; letter-spacing: -0.01em; }}
.verdict-sub {{ color: #b7c0e0; font-size: 0.92em; margin-top: 3px; }}
.verdict-chip {{
    display: inline-block; background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    padding: 2px 10px; border-radius: 999px; font-size: 0.82em; margin: 2px 4px 0 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace; color: #ffe4e0;
}}

.label-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 12px; margin-top: 16px;
}}
.label-card {{
    background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 14px 16px; transition: transform 0.15s ease, border-color 0.15s ease;
    animation: fadeSlideIn 0.4s ease backwards;
}}
.label-card:hover {{ transform: translateY(-2px); border-color: rgba(255,255,255,0.2); }}
.label-card--flagged {{ border-color: rgba(248,113,113,0.35); box-shadow: 0 0 22px -12px rgba(239,68,68,0.7); }}
.label-card__head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
.label-card__name {{ font-weight: 700; text-transform: capitalize; letter-spacing: 0.01em; color: #eef0fb; font-size: 0.92em; }}
.label-card__badge {{
    font-size: 0.66em; font-weight: 800; letter-spacing: 0.06em; padding: 3px 9px;
    border-radius: 999px; text-transform: uppercase;
}}
.label-card__badge--flagged {{
    background: linear-gradient(90deg, #f43f5e, #f97316); color: white;
    box-shadow: 0 0 12px -2px rgba(244,63,94,0.8);
}}
.label-card__badge--clear {{ background: rgba(45,212,191,0.14); color: #5eead4; border: 1px solid rgba(45,212,191,0.35); }}
.label-card__meter {{ position: relative; height: 8px; border-radius: 999px; background: rgba(255,255,255,0.07); margin-bottom: 9px; }}
.label-card__fill {{ position: absolute; inset: 0; border-radius: 999px; height: 100%; transition: width 0.4s ease; }}
.label-card__fill--flagged {{ background: linear-gradient(90deg, #f43f5e, #fb923c); box-shadow: 0 0 10px rgba(244,63,94,0.6); }}
.label-card__fill--clear {{ background: linear-gradient(90deg, #22d3ee, #8b5cf6); box-shadow: 0 0 10px rgba(34,211,238,0.5); }}
.label-card__tick {{ position: absolute; top: -3px; bottom: -3px; width: 2px; background: rgba(255,255,255,0.55); border-radius: 1px; }}
.label-card__foot {{ display: flex; justify-content: space-between; font-size: 0.78em; color: #93a0c4; }}
.label-card__prob {{ font-weight: 700; color: #eef0fb; font-variant-numeric: tabular-nums; }}
.label-card__threshold {{ font-variant-numeric: tabular-nums; }}

.shap-surface {{
    background: #f8fafc; color: #111827; border-radius: 14px; padding: 18px 20px;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.25);
    overflow-x: auto;
}}
.empty-hint {{ color: #93a0c4; font-style: italic; padding: 10px 4px; }}

#app-footer {{
    text-align: center; color: #6b7699; font-size: 0.82em; margin-top: 26px; padding: 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
}}

@keyframes fadeSlideIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes pulseGlow {{ 0%, 100% {{ filter: drop-shadow(0 0 2px currentColor); }} 50% {{ filter: drop-shadow(0 0 10px currentColor); }} }}
"""

HERO_HTML = f"""
<div id="landing-hero">
  {SHIELD_SVG}
  <h1>🛡️ Explainable Toxic Comment Moderation</h1>
  <p class="subtitle">A fine-tuned transformer model for identifying toxic comments in text.</p>
  <div class="badge-row">
    <span class="badge-pill">🤖 Fine-tuned BERT</span>
    <span class="badge-pill">🔍 SHAP-explainable</span>
    <span class="badge-pill">⚖️ Fairness-audited</span>
  </div>
</div>
"""

INFO_BAR_HTML = f"""
<div id="info-bar">
  <h1>🛡️ Explainable Toxic Comment Moderation</h1>
  <p>
    Model in use: <strong>{BEST_MODEL_NAME}</strong> (fine-tuned on the Jigsaw Toxic Comment dataset).
    Each label below uses its own tuned decision threshold, not a flat 50% &mdash; some labels
    (like <code>threat</code>) are rare enough that a lower threshold is needed to flag anything at all.
  </p>
  <div class="info-pills">
    <span class="info-pill">🤖 model: <code>{BEST_MODEL_NAME}</code></span>
    <span class="info-pill">🎯 per-label tuned thresholds</span>
    <span class="info-pill">🔍 SHAP explainability</span>
  </div>
</div>
"""

EMPTY_MSG = '<p class="empty-hint">Type a comment, or pick a scenario on the left.</p>'


def _wrap_shap(shap_html_raw):
    return f'<div class="shap-surface">{shap_html_raw}</div>'


with gr.Blocks(title="Explainable Toxic Comment Moderation") as demo:
    with gr.Column(visible=True) as landing_page:
        gr.HTML(HERO_HTML)
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                enter_btn = gr.Button("Enter Demo →", elem_id="enter-btn", variant="primary", size="lg")
            with gr.Column(scale=1):
                pass

    with gr.Column(visible=False) as main_app:
        gr.HTML(INFO_BAR_HTML)

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                gr.Markdown("### 🎬 Try a scenario", elem_classes=["section-title"])
                scenario_picker = gr.Radio(
                    choices=list(SCENARIOS.keys()),
                    label="Four preset scenarios, each demonstrating a different finding",
                )
                scenario_note = gr.Markdown()

                gr.Markdown("### ✍️ Or write your own", elem_classes=["section-title"])
                comment_box = gr.Textbox(label="Comment", lines=3, placeholder="Type a comment here...")
                submit_btn = gr.Button("⚡ Analyse", elem_id="analyse-btn", variant="primary")

            with gr.Column(scale=3, elem_classes=["glass-panel"]):
                gr.Markdown("### 🚦 Verdict", elem_classes=["section-title"])
                verdict_out = gr.HTML(EMPTY_MSG)
                gr.Markdown("### 📊 Toxicity labels", elem_classes=["section-title"])
                labels_out = gr.HTML()
                gr.Markdown("### 🔬 SHAP explanation (toxic label)", elem_classes=["section-title"])
                shap_out = gr.HTML()

        gr.HTML(
            '<div id="app-footer">Built with 🤖 Transformers · 🔍 SHAP · ⚖️ Fairness-audited '
            '&mdash; see <code>reports/metrics.json</code> for the full evaluation.</div>'
        )

        outputs = [comment_box, scenario_note, verdict_out, labels_out, shap_out]

        disable_submit = lambda: gr.update(interactive=False, value="⏳ Analysing...")
        enable_submit = lambda: gr.update(interactive=True, value="⚡ Analyse")

        scenario_picker.change(
            fn=disable_submit, outputs=submit_btn, queue=False
        ).then(
            fn=load_scenario, inputs=scenario_picker, outputs=outputs
        ).then(
            fn=enable_submit, outputs=submit_btn, queue=False
        )

        submit_btn.click(
            fn=disable_submit, outputs=submit_btn, queue=False
        ).then(
            fn=moderate_comment, inputs=comment_box, outputs=[verdict_out, labels_out, shap_out]
        ).then(
            fn=enable_submit, outputs=submit_btn, queue=False
        )

        comment_box.submit(
            fn=disable_submit, outputs=submit_btn, queue=False
        ).then(
            fn=moderate_comment, inputs=comment_box, outputs=[verdict_out, labels_out, shap_out]
        ).then(
            fn=enable_submit, outputs=submit_btn, queue=False
        )

    enter_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[landing_page, main_app],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=APP_CSS)
