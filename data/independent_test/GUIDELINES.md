# Independent test set: writing and labelling guidelines

This set answers the mentor's request for "a larger, independently prepared
test set". Its value depends on one property: **it is independent of
everything the pipeline was built from.**

The training corpus was generated from templates. The 45-ticket benchmark,
the deployment set and the paraphrased calibration set were all written by
Gemini. This set is written **by hand, by one author (AW)**, without seeing
any of that material. It is then labelled a second time, blind, by **one
second labeller** (name to be confirmed; their code goes in the `labeller`
column).

The set is scored by `scripts/eval_independent.py`. No model is retrained and
no threshold is changed for it.

---

## 1. Hard rules (break one and the set is no longer independent)

1. **Do not look at** any of these before or while you write:
   - `data/synthetic_tickets.csv`
   - `data/generate_dataset.py`
   - any file under `data/` ending in `tickets*.json`
   - the benchmark sets
   - the Streamlit demo's retrieved tickets
   - the README's example tickets
   - the paper's examples

   You have worked on this project and have seen some of them, so write from
   your own experience of IT problems, not from what you remember of them.
2. **No LLM help of any kind.** Do not use ChatGPT, Gemini, Copilot, Claude, a
   paraphraser, a "humanizer" or grammar rewriting. Spell-check in your editor
   is fine. The whole point is text that no language model wrote.
3. **Do not copy** tickets from the internet, forums or other datasets. Write
   each one yourself.
4. **No real personal data.** Use made-up names, machines and locations.
5. **Do not run the pipeline** on the tickets until the ticket file and the
   second labels are both complete. Tuning a ticket after seeing how the
   system handles it would contaminate the result.

## 2. What to write

**About 100 tickets** in total, all by AW.

| Kind | Count | `true_category` | `expected_outcome` |
|---|---|---|---|
| Actionable IT problems, **about 13 per category** across all seven categories | ~91 | one of the seven categories | `proceed` |
| Off-topic (not an IT request at all: canteen, travel claims, room bookings, …) | ~5 | `Out of scope` | `escalate` |
| Too vague to act on (no system and no symptom: "it's broken again", "help pls") | ~5 | `Out of scope` | `escalate` |

**Write the way employees write to a help desk.** Because one person writes
every ticket, deliberately vary the voice to avoid a single house style.
- Imagine different senders (a new joiner, a manager in a hurry, a non-technical user, an engineer).
- Vary the length (one line to a short paragraph) and the tone (calm, rushed, annoyed).
- Vary the vocabulary: say "the shared drive", "the K: drive", "the team folder", not only "file share".
- Occasional typos, missing punctuation and informal wording are realistic and welcome.
- Describe the symptom the user sees, not the technical cause. Users rarely know the cause.
- About a third of tickets can have an empty `title` (users often leave it blank) or a title that is just a few words.
- Do not write two tickets that differ only by a name or a number.

## 3. The seven categories

Label a ticket with **the team that would have to fix it**.

| Category | The team that handles … |
|---|---|
| **Access Management** | accounts, passwords, MFA, single sign-on, **requests to be granted access** (to a folder, mailbox, group or system), joiners, leavers and role changes |
| **Application** | business or office software misbehaving: errors, crashes, wrong output, features not working, licences, installing a program |
| **Database** | data held in a database: records missing, wrong or duplicated; reports or queries failing or very slow; database errors |
| **Infrastructure** | servers and back-end systems: a server or internal service down or slow, capacity, virtual machines, hosting platforms |
| **Network** | connectivity: Wi-Fi, LAN, VPN, internet access, DNS, firewalls blocking something, latency |
| **Security** | phishing, malware or ransomware, suspicious activity, lost or stolen devices, security alerts or vulnerabilities |
| **Storage** | where files live: shared drives that are **broken or unreachable** (not access requests), disk space, quotas, backups and restores |

**Boundary rules, so the writer and the second labeller label the same way:**
- *"Please give me access to the finance folder"* is **Access Management**, a request. *"The finance folder stopped opening for everyone"* is **Storage**, a fault.
- A program that is slow because the whole network is slow is **Network**. A program that is slow while everything else works is **Application**.
- Physical device problems (printer, laptop hardware, monitor) have no category of their own. Avoid them, or label them by the team above that would actually be involved, and say why in `notes`.
- If you genuinely cannot pick one category, the ticket is ambiguous: rewrite it, or drop it.

## 4. Filling in the ticket file (AW)

1. Copy `TEMPLATE.csv` to `tickets_AW.csv`.
2. One row per ticket:

| Column | Content |
|---|---|
| `ticket_id` | `AW-001`, `AW-002`, … unique, never reused |
| `author` | `AW` |
| `title` | may be empty |
| `description` | the ticket body, required |
| `true_category` | one of the seven names exactly as written above, or `Out of scope` |
| `expected_outcome` | `proceed` (an actionable IT problem) or `escalate` (off-topic or too vague for anyone to act on without asking the user) |
| `notes` | optional; anything a reader should know |

3. Save as **UTF-8 CSV**. In Excel use "CSV UTF-8". Quote any text that contains commas, which Excel does for you.

## 5. Second labelling (for Cohen's κ)

The second labeller labels **every** ticket and **must not see AW's labels**.

1. When `tickets_AW.csv` is complete, give the second labeller a copy with only `ticket_id`, `title` and `description`. Delete the label columns from the copy, and shuffle the rows if you can.
2. The second labeller copies `SECOND_LABELS_TEMPLATE.csv` to `second_labels_AW.csv` and fills in, for every ticket:
   - `labeller`: their own short code, for example their initials, not `AW`;
   - `second_category`;
   - `second_expected_outcome`.

   They use sections 2–3 of this file only, and do not discuss any ticket with AW.
3. κ is computed from these two independent labels **before** any discussion, and is reported as it comes out.

## 6. Resolving disagreements

After κ is recorded:
1. `python scripts/eval_independent.py --check-only` lists every ticket where the two labels disagree.
2. AW and the second labeller discuss each one and agree a final label.
3. Record each agreement in `adjudication.csv` (copied from `ADJUDICATION_TEMPLATE.csv`). Put both codes in `resolved_by` (for example `AW+XY`) and a short reason.
4. The evaluation uses the agreed label, or the resolved one. It refuses to run while any disagreement is unresolved. Nobody edits `true_category` or the second labels after this point: the original labels are what κ measured.

## 7. If no second labeller is available

The set can be evaluated with AW's labels alone:

```powershell
python scripts/eval_independent.py --single-annotator
```

κ is then not computed, and **the paper must say the set was labelled by a
single annotator**. The script prints this reminder and records
`annotation = single annotator` in its summary. Use this mode only if the
second labelling truly cannot happen. A second, blind label is what lets the
paper claim the labels are not one person's opinion.

## 8. Running the evaluation (after everything above is done)

```powershell
python scripts/eval_independent.py --check-only   # validate, kappa, list disagreements; no model
python scripts/eval_independent.py                # route every ticket through the unchanged pipeline
```

The evaluation reports:
- category accuracy for the cascade and for Tier 2 alone, on the in-scope tickets, with Wilson 95% intervals;
- the escalation rate at 0.67;
- whether each ticket's escalate or proceed outcome matched its expected outcome;
- the share handled by Tier 1;
- per-category recall, and how many misclassified tickets were drafted rather than escalated.

No language model is called.
