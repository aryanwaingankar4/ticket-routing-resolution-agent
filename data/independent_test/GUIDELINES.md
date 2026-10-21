# Independent test set: writing and labelling guidelines

This set answers the mentor's request for "a larger, independently prepared
test set". Its value depends on one property: **it is independent of
everything the pipeline was built from.**

The training corpus was generated from templates. The 45-ticket benchmark,
the deployment set and the paraphrased calibration set were all written by
Gemini. This set is written **by hand, by two authors: AW (Aryan) and SP
(Shreyansh)**, about 54 tickets each, without seeing any of that material.
Each author then labels **the other author's tickets, blind**.

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

   If you have worked on this project and have seen some of them, write from
   your own experience of IT problems, not from what you remember of them.
2. **No LLM help of any kind.** Do not use ChatGPT, Gemini, Copilot, Claude, a
   paraphraser, a "humanizer" or grammar rewriting. Spell-check in your editor
   is fine. The whole point is text that no language model wrote.
3. **Do not copy** tickets from the internet, forums or other datasets. Write
   each one yourself.
4. **No real personal data.** Use made-up names, machines and locations.
5. **Do not show your tickets to the other author** before they label them,
   and do not discuss any ticket until both label files are complete.
6. **Do not run the pipeline** on the tickets until the ticket files, the
   second labels and the adjudication are all complete. Tuning a ticket
   after seeing how the system handles it would contaminate the result.

## 2. What each author writes

**About 54 tickets per author**: AW in `tickets_AW.csv`, SP in `tickets_SP.csv`.

| Kind | Count per author | `true_category` | `expected_outcome` |
|---|---|---|---|
| Actionable IT problems, **7 per category** across all seven categories | 49 | one of the seven categories | `proceed` |
| Off-topic or too vague to act on | 5 in total | see below | `escalate` |

- **Off-topic:** not an IT request at all (canteen, travel claims, room bookings, …). Use `true_category` = `NONE`.
- **Too vague:** "it's broken again", "help pls". Give it the category of the team it would go to if one is clear; if no team is identifiable, use `NONE`. The expected outcome is `escalate` either way.

**`NONE` is allowed only with `expected_outcome = escalate`.** The script refuses a ticket with `NONE` and `proceed`. Category accuracy is computed only on tickets with a real category; escalation correctness uses every ticket.

**Write the way employees write to a help desk.**
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
| **Access Management** | accounts, passwords, MFA, single sign-on, **requests to be granted access** to a share, mailbox, group or system, joiners, leavers and role changes |
| **Application** | business or office software misbehaving: errors, crashes, wrong output, features not working, licences, installing a program |
| **Database** | data held in a database: records missing, wrong or duplicated; reports or queries failing or very slow; database errors |
| **Infrastructure** | servers and back-end systems: a server or internal service down or slow, capacity, virtual machines, hosting platforms |
| **Network** | connectivity: Wi-Fi, LAN, VPN, internet access, DNS, firewalls blocking something, latency |
| **Security** | phishing, malware or ransomware, suspicious activity, lost or stolen devices, security alerts or vulnerabilities |
| **Storage** | where files live: a **share or drive fault** (broken, unreachable, full, slow), disk space, quotas, backups and restores |

**Boundary rules, so both authors label the same way:**
- **An access request for a share, mailbox or system is Access Management. A share or drive fault is Storage.** *"Please give me access to the finance folder"* is Access Management. *"The finance folder stopped opening for everyone"* is Storage.
- A program that is slow because the whole network is slow is **Network**. A program that is slow while everything else works is **Application**.
- Physical device problems (printer, laptop hardware, monitor) have no category of their own. Avoid them, or label them by the team above that would actually be involved, and say why in `notes`.
- If you genuinely cannot pick one category for an actionable ticket, it is ambiguous: rewrite it, or drop it.

## 4. Filling in your ticket file

`tickets_AW.csv` and `tickets_SP.csv` already exist with only the header row. Each author fills in their own file, one row per ticket:

| Column | Content |
|---|---|
| `ticket_id` | `AW001`, `AW002`, … or `SP001`, `SP002`, … (three digits, no gaps needed, never reused) |
| `author` | `AW` or `SP` |
| `title` | may be empty |
| `description` | the ticket body, required |
| `true_category` | one of the seven names exactly as written above, or `NONE` |
| `expected_outcome` | `proceed` (an actionable IT problem) or `escalate` (off-topic, or too vague for anyone to act on without asking the user) |
| `notes` | optional; anything a reader should know |

Save as **UTF-8 CSV**. In Excel use "CSV UTF-8". Quote any text that contains commas, which Excel does for you. Do not rename the file or change the columns.

Check your file at any time; this works when only one file is filled:

```powershell
python scripts/eval_independent.py --check-only
```

## 5. Cross-labelling (for Cohen's κ)

When **both** ticket files are complete:

1. AW generates the two blind files:

   ```powershell
   python scripts/eval_independent.py --make-second-label-files
   ```

   This writes `second_labels_on_AW_by_SP.csv` (AW's tickets, for SP to label) and `second_labels_on_SP_by_AW.csv` (SP's tickets, for AW to label). Each holds only `ticket_id`, `title`, `description`, an empty `second_label`, an empty `second_expected_outcome` and the `labeller` code. The rows are shuffled, and the writer's labels are not in the file.
2. Each author fills **both** label columns for **every** row of the file addressed to them:
   - `second_label`: one of the seven categories, or `NONE` for a ticket that is not an IT request;
   - `second_expected_outcome`: `proceed` or `escalate`, by the same rules as section 2. `NONE` requires `escalate`.

   Use sections 2–3 of this file only, and do not discuss any ticket with the writer. A row counts as labelled only when both columns are filled.
3. **Do not edit your own ticket file after this point.** The script refuses a second-label file whose ticket text no longer matches the ticket file.

Both the **category** and the **expected outcome** are cross-labelled. κ is computed for each **separately**, from these first, independent labels, **before** any discussion. It is given pooled and per direction, and is reported as it comes out.

## 6. Resolving disagreements

After both second-label files are complete:

1. List the disagreements and pre-fill the adjudication file:

   ```powershell
   python scripts/eval_independent.py --check-only
   python scripts/eval_independent.py --make-adjudication-file
   ```

   `adjudication.csv` gets one row per ticket where the category, the outcome or both disagree. Columns: `ticket_id, author_label, second_label, final_label, author_outcome, second_outcome, final_outcome, resolved_by, reason`. The author and second columns are filled in. Where one of the two fields already agrees, its `final_…` column is pre-filled with the agreed value.
2. AW and SP discuss each row and agree both final values. Fill in `final_label`, `final_outcome`, `resolved_by` (`AW+SP`) and a short `reason`. A `final_label` of `NONE` requires `final_outcome` = `escalate`.
3. The evaluation uses the agreed values where the two matched and the `final_…` values where they did not. It refuses to run while any row lacks `final_label` or `final_outcome`. **Nobody edits the ticket files or the second-label files after this point**: the original labels are what κ measured.

## 7. Fallback: if cross-labelling cannot happen

The set can be evaluated with each writer's own labels:

```powershell
python scripts/eval_independent.py --single-annotator
```

κ is then not computed, and **the paper must say the set was labelled by a
single annotator**. The script prints this reminder and records
`annotation = single annotator` in its summary. It refuses this mode if any
second label has been filled in. Use it only if the cross-labelling truly
cannot happen.

## 8. Running the evaluation (after everything above is done)

```powershell
python scripts/eval_independent.py --check-only   # must end with "Ready for the final run."
python scripts/eval_independent.py                # route every ticket through the unchanged pipeline
```

The evaluation reports:
- Cohen's κ for category and, separately, for expected outcome (each pooled and per direction) with bootstrap 95% intervals, and the number of disagreements on each;
- category accuracy for the cascade and for Tier 2 alone, on the tickets with a real category, with Wilson 95% intervals;
- the escalation rate at 0.67, and whether each ticket's escalate or proceed outcome matched its agreed (or adjudicated) expected outcome;
- the share handled by Tier 1;
- per-category recall, and how many misclassified tickets were drafted rather than escalated.

No language model is called.
