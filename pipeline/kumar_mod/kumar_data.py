"""Verbatim port of Kumar et al.'s rule-based-moderation data + prompt logic.

Source of truth: external/kumar_llm_content_mod/scripts/rule_based_moderation.py (the authors'
released GPT-3.5 harness). Functions below reproduce that script EXACTLY so our local-model
replication is faithful by construction:

  - unmark()              : identical markdown-stripping (markdown lib, plain output format).
  - extract_rules_text()  : identical -- skip kind=='link' rules, unmark, number from 1.
  - FIRST/SECOND/THIRD/SAMPLE strings : byte-for-byte the prompt turns Kumar sent GPT-3.5,
                            including the canned "You are a stupid idiot." few-shot prime and the
                            8-space line in the first user turn (preserved from his source indentation).

The ONLY differences from Kumar are intentional and documented:
  (1) we target local instruct models via a chat template instead of openai.ChatCompletion;
  (2) we read his released balanced CSVs from external/kumar_llm_content_mod/ instead of rule_dataset/.
Nothing about the prompt content or rule construction changes.
"""
from __future__ import annotations
import csv, json, copy
from io import StringIO
from pathlib import Path

from markdown import Markdown

ROOT = Path(__file__).resolve().parents[2]
KUMAR_DIR = ROOT / "external" / "kumar_llm_content_mod" / "data" / "rule_moderation"
RULES_JSONL = KUMAR_DIR / "subreddit_rules_w_description.jsonl"
CSV_DIR = KUMAR_DIR / "subreddit_balanced_datasets"


# politics_2 is a degenerate duplicate community in the released CSVs; Kumar drops it.
EXCLUDE_SUBS = {"politics_2"}


def _unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        _unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()


# Register a "plain" output format on the markdown lib that just concatenates element text/tails,
# yielding markdown-stripped plain text. stripTopLevelTags=False keeps the top-level wrapper text so
# nothing is dropped. This is Kumar's exact unmarking recipe.
Markdown.output_formats["plain"] = _unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False


def unmark(text):
    """Strip markdown -> plain text. Verbatim from Kumar."""
    return __md.convert(text)


def extract_rules_text(rules):
    """Verbatim from Kumar: skip link rules, unmark, number from 1."""
    rules_text = []
    count = 1
    for r in rules:
        # Link rules govern submitted URLs, not comment text, so they never apply to the comment
        # moderation task; skipping them also keeps the visible numbering 1..N contiguous.
        if r["kind"] == "link":
            continue
        cleaned_rule = unmark(r["description"]).strip()
        rules_text.append(str(count) + ". " + cleaned_rule)
        count += 1
    return rules_text


def get_rules_string(rules_text):
    return "\n".join(rules_text)


# The lone 8-space line ('        \n') is preserved from Kumar's source indentation; it reaches the
# model verbatim, so it is kept rather than normalized to avoid any prompt drift.
FIRST_TMPL = ('You are a helpful content moderation assistant for the online subreddit called '
              '{subreddit}. The subreddit has the following description: "{description}".\n'
              '        \n'
              'Here are the rules for the community:\n\n{rules_string}')
# Canned few-shot prime: a fixed toxic exemplar that conditions the model on the JSON answer format
# before the real comment is shown. Identical to Kumar's harness.
PRIME_COMMENT = 'Consider the following comment: "You are a stupid idiot."'
ACK_RULES = "OK, I understand the rules of the community."
ACK_COMMENT = "OK, I understand the comment."
THIRD_STRING = ('Return a JSON object with five fields, "would_moderate," that is either "yes" or '
                '"no" depending on if you would remove this comment from the subreddit, "rule" which '
                'is the text of the rule being violated, "rule_nums" which are a comma-separated list '
                'of rules being violated, "explanation" which contains a reason for your decision, '
                'and "rating" which is a score from 1 to 5 depending on how violative the comment is '
                'based on the rules.')
SAMPLE_JSON = ('{"would_moderate": "yes", "rule": "<rule_being_violated>", "rule_nums": "5,6,7", '
               '"explanation": "<explanation>", "rating": 3}')


def comment_turn(body):
    return f'Consider the following comment: "{body}"'


def base_chat(subreddit, description, rules_string):
    """The 6-message few-shot prime Kumar builds once per subreddit (verbatim structure)."""
    first = FIRST_TMPL.format(subreddit=subreddit, description=description, rules_string=rules_string)
    return [
        {"role": "user", "content": first},
        {"role": "assistant", "content": ACK_RULES},
        {"role": "user", "content": PRIME_COMMENT},
        {"role": "assistant", "content": ACK_COMMENT},
        {"role": "user", "content": THIRD_STRING},
        {"role": "assistant", "content": SAMPLE_JSON},
    ]


def messages_for_comment(subreddit, description, rules_string, body):
    """Full message list for one comment: base prime + the comment's 3 turns (ends on a user turn
    so the model generates the JSON). Matches Kumar's per-comment mod_chat exactly."""
    # deepcopy so appending this comment's turns does not mutate the shared per-subreddit prime.
    chat = copy.deepcopy(base_chat(subreddit, description, rules_string))
    chat.append({"role": "user", "content": comment_turn(body)})
    chat.append({"role": "assistant", "content": ACK_COMMENT})
    # End on the user turn that re-issues the JSON-format instruction, so generation produces the
    # decision object rather than another acknowledgement.
    chat.append({"role": "user", "content": THIRD_STRING})
    return chat


def load_rules():
    """{subreddit: (description, rules_string)} from Kumar's released JSONL, his extraction."""
    desc, rules = {}, {}
    for line in open(RULES_JSONL):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        sub = d["subreddit"]
        desc[sub] = d["description"]
        rules[sub] = get_rules_string(extract_rules_text(d["rules"]))
    return desc, rules


def load_comments(subreddit):
    """List of (body, moderated_int) from Kumar's released balanced CSV (body,subreddit,moderated).
    FAITHFUL to Kumar's harness, which stores comments in a dict `body_to_decision[body]=moderated`
    (rule_based_moderation.py) -- so identical bodies are DEDUPLICATED, last-occurrence label wins,
    and iteration is first-seen order. We replicate that exactly. CSV-aware (bodies contain commas)."""
    f = CSV_DIR / f"{subreddit}.csv"
    bd = {}
    with open(f, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # header: body,subreddit,moderated
        for row in reader:
            if len(row) < 3:
                continue
            # Keying on body reproduces Kumar's dict-based dedup: a repeated body overwrites, so the
            # last-occurrence label wins and only one copy survives.
            bd[row[0]] = int(row[2])
    # dict preserves first-seen insertion order, matching Kumar's iteration order over the corpus.
    return list(bd.items())


def clean_subreddits():
    """The 95 reported subreddits: each has a balanced CSV, a rules record, and at least one
    COMMENT-applicable rule. clean_subreddits applies exactly three filters to the released
    balanced CSVs -- it excludes the degenerate politics_2, any CSV without a rules record, and
    the two subreddits whose comment-applicable rules_string is empty because all their rules were
    link-type (GlobalOffensiveTrade, SuicideWatch). The result is the promptable corpus of 95 communities with a rules
    record minus the two empty-rule subs, matching the denominators table in the paper."""
    desc, rules = load_rules()
    subs = []
    for f in sorted(CSV_DIR.glob("*.csv")):
        s = f.stem
        # Drop the degenerate sub and any CSV lacking a rules record in the JSONL.
        if s in EXCLUDE_SUBS or s not in desc:
            continue
        # Empty rules_string means every rule was kind=='link' (e.g. GlobalOffensiveTrade,
        # SuicideWatch): no comment-applicable rule to prompt with, so the sub is unpromptable.
        if len(rules[s].strip()) == 0:
            continue
        subs.append(s)
    return subs


PARSE_STATS = {"fallback_hits": 0}


def reset_parse_stats():
    PARSE_STATS["fallback_hits"] = 0


def parse_decision(text):
    """Return (would_moderate in {'yes','no',None}, rating in 1..5 or None) from a model's JSON-ish
    output. Robust to extra prose around the JSON and to minor format drift."""
    wm, rating = None, None
    if not text:
        return wm, rating

    obj = None
    start = text.find("{")
    if start != -1:
        # Brace-balance scan to isolate the first complete {...} block, tolerating prose or code
        # fences around it; json.loads only the balanced span rather than the whole response.
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except Exception:
                        obj = None
                    break
    if isinstance(obj, dict):
        v = obj.get("would_moderate")
        if isinstance(v, str):
            v = v.strip().lower()
            if v in ("yes", "no"):
                wm = v
        elif isinstance(v, bool):
            wm = "yes" if v else "no"
        r = obj.get("rating")
        try:
            rating = int(r)
            # Ratings outside Kumar's 1..5 violation scale are treated as unparseable.
            if not (1 <= rating <= 5):
                rating = None
        except Exception:
            rating = None
    if wm is None:
        # Fallback for malformed JSON: locate the would_moderate key as raw text and read whichever
        # of "yes"/"no" appears first within a 40-char window starting at the would_moderate key.
        # Counted so the fallback rate is auditable.
        low = text.lower()
        iy = low.find('"would_moderate"')
        if iy != -1:
            PARSE_STATS["fallback_hits"] += 1
            seg = low[iy:iy + 40]


            iyes = seg.find('"yes"')
            ino = seg.find('"no"')
            if iyes != -1 and (ino == -1 or iyes < ino):
                wm = "yes"
            elif ino != -1 and (iyes == -1 or ino < iyes):
                wm = "no"
    return wm, rating
