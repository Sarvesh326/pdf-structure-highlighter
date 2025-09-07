import re
from typing import List
from .utils import LABEL2ID

SEC_HEADERS = {
    "abstract", "introduction", "related", "methods",
    "conclusion", "conclusions", "references", "acknowledgments"
}

# Roman numeral regex (I., II., III., … up to 3999)
ROMAN_REGEX = re.compile(
    r"^(M{0,4}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})"
    r"(IX|IV|V?I{0,3}))\.$"
)
# Roman numeral + subsection number (II.6., III.1., etc.)
ROMAN_SUBSEC_REGEX = re.compile(
    r"^(M{0,4}(CM|CD|D?C{0,3})"
    r"(XC|XL|L?X{0,3})"
    r"(IX|IV|V?I{0,3}))\.\d+\.$"
)

MAX_HEADER_LEN = 15  # hard cutoff to avoid capturing whole paragraphs

def label_tokens(words: List[str]) -> List[int]:
    labels = [LABEL2ID["O"]] * len(words)
    i = 0
    while i < len(words):
        token = words[i].strip(":").strip()
        lw = token.lower()

        # --- Case A: direct match with known section headers ---
        if lw in SEC_HEADERS:
            labels[i] = LABEL2ID["B-SEC"]
            j = i + 1
            count = 1
            while j < len(words) and count < MAX_HEADER_LEN:
                if words[j].lower() not in SEC_HEADERS:
                    break
                labels[j] = LABEL2ID["I-SEC"]
                j += 1
                count += 1
            i = j
            continue

        # --- Case B: Roman numeral headers (I., II., III. ...) ---
        if ROMAN_REGEX.match(token):
            # labels[i] = LABEL2ID["B-SEC"]
            j = i + 1
            count = 1
            while j < len(words) and count < MAX_HEADER_LEN:
                nxt = words[j]
                if not nxt.isupper() and nxt.lower() not in SEC_HEADERS:
                    break
                labels[i] = LABEL2ID["B-SEC"]
                labels[j] = LABEL2ID["I-SEC"]
                j += 1
                count += 1
            i = j
            continue

        # --- Case C: Roman numeral subsection (II.6.) ---
        if ROMAN_SUBSEC_REGEX.match(token):
            labels[i] = LABEL2ID.get("B-SUBSEC", LABEL2ID["B-SEC"])
            j, count = i + 1, 1
            while j < len(words) and count < MAX_HEADER_LEN:
                if words[j].endswith("."):  # likely end of header
                    labels[j] = LABEL2ID.get("I-SUBSEC", LABEL2ID["I-SEC"])
                    j += 1
                    break
                labels[j] = LABEL2ID.get("I-SUBSEC", LABEL2ID["I-SEC"])
                j, count = j + 1, count + 1
            i = j
            continue

        i += 1

    return labels
