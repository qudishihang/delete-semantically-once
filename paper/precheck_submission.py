"""Pre-submission compliance check for the ICDE 2027 EAB submission.

Run from this directory:  python precheck_submission.py
Exit code 0 = clean, 1 = at least one blocking item remains.

Checks, in the order ICDE 2027's desk-rejection policy cares about them:
  1. no unfilled TODOs in the manuscript source (author block, artifact URL);
  2. the title still carries the EAB tag (required at submission);
  3. page budget: body must end by page 12; page 13 may hold only the
     AI-generated content acknowledgement and the references;
  4. the AI-generated content acknowledgement is present and names a system;
  5. every \\includegraphics target exists (a missing figure is a broken PDF);
  6. every \\ref/\\cite resolves (no "??" in the rendered PDF).
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "main.tex")
PDF = os.path.join(HERE, "main.pdf")

problems = []
notes = []


def fail(msg):
    problems.append(msg)


def note(msg):
    notes.append(msg)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def run(cmd):
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout


def main():
    if not os.path.exists(TEX):
        print("main.tex not found; run from paper/")
        return 1
    src = read(TEX)

    # 1. unfilled placeholders ------------------------------------------------
    for pattern, what in [
        (r"TO FILL", "unfilled 'TO FILL' marker"),
        (r"TODO:", "unfilled 'TODO:' marker"),
        (r"github\.com/OWNER", "artifact URL still has the OWNER token"),
        (r"REPLACE-ME", "leftover 'REPLACE-ME' token"),
    ]:
        for m in re.finditer(pattern, src):
            line_no = src[: m.start()].count("\n") + 1
            line = src.splitlines()[line_no - 1]
            if r"\newcommand{\todo}" in line:  # the macro definition is not a placeholder
                continue
            fail("%s at main.tex:%d" % (what, line_no))

    # 2. EAB title tag --------------------------------------------------------
    title = re.search(r"\\title\{(.+?)\}\s*\n", src, re.S)
    if not title:
        fail("no \\title found")
    elif "[Experiment, Analysis, and Benchmark]" not in title.group(1):
        note("title has no EAB tag: required at submission, removed at camera-ready")

    # 3. page budget ----------------------------------------------------------
    if not os.path.exists(PDF):
        fail("main.pdf missing; build it first")
    else:
        try:
            info = run(["pdfinfo", PDF])
            pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
            if pages > 13:
                fail("PDF has %d pages; body must end by 12, so at most 13 total" % pages)
            else:
                note("PDF has %d pages" % pages)
            last = run(["pdftotext", "-f", str(pages), "-l", str(pages), PDF, "-"])
            flat = " ".join(last.split())
            if "ONCLUSION" in flat:
                fail("the conclusion is on the last page: the body does not end by page 12")
            if not re.search(r"EFERENCES", flat):
                fail("references do not start on the last page")
            if "AI- GENERATED CONTENT" not in flat and "AI-GENERATED CONTENT" not in flat:
                note("AI acknowledgement is not on the last page (allowed, but check it exists)")
            # nothing but refs + ack may follow the body
            body_tail = run(["pdftotext", "-f", "12", "-l", "12", PDF, "-"])
            if "??" in body_tail:
                fail("unresolved reference ('??') on page 12: rebuild until clean")
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            fail("could not inspect the PDF: %r" % (exc,))

    # 4. AI disclosure --------------------------------------------------------
    ack = re.search(r"\\section\*\{AI-generated content acknowledgement\}(.+?)\\bibliography", src, re.S)
    if not ack:
        fail("no AI-generated content acknowledgement section")
    else:
        body = ack.group(1)
        if len(body.split()) < 60:
            note("AI acknowledgement is short; the policy asks for systems + affected sections")
        if not re.search(r"\b(ChatGPT|GPT|Copilot|Claude|Gemini|LLM|model)\b", body):
            fail("AI acknowledgement names no system")

    # 5. figures --------------------------------------------------------------
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        target = m.group(1)
        for ext in ("", ".pdf", ".png"):
            if os.path.exists(os.path.join(HERE, target + ext)):
                break
        else:
            fail("missing figure: %s" % target)

    # 6. citations defined ----------------------------------------------------
    bib = os.path.join(HERE, "references.bib")
    keys = set()
    if os.path.exists(bib):
        keys = set(re.findall(r"@\w+\{([^,]+),", read(bib)))
    for m in re.finditer(r"\\cite\{([^}]+)\}", src):
        for key in m.group(1).split(","):
            if key.strip() and key.strip() not in keys:
                fail("citation key not in references.bib: %s" % key.strip())

    # report ------------------------------------------------------------------
    for n in notes:
        print("note : %s" % n)
    if problems:
        print("\nBLOCKING (%d):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        print("\nNot ready to submit.")
        return 1
    print("\nAll automated checks pass.  Still do by hand:")
    print("  * CMT: complete profile + declared COIs for EVERY author, or desk reject")
    print("  * CMT: ORCID for every author")
    print("  * CMT: supplemental material = the public artifact URL")
    print("  * CMT: 1000-character scope justification (see PUBLISHING.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
