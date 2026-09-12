"""
audit_sign_spans.py

Day 31 pre-decision audit: before committing to a boundary-label
scheme, check whether the point-anchor pattern (START_FRAME ==
END_FRAME) seen for '(1h)THIS' in utterance 8400279 is typical, or
whether real DOMINANT_HAND spans (START_FRAME != END_FRAME) are
actually the majority case, across a real sample -- not one utterance.

Reports, per sign, whether DOMINANT_HAND resolved to a point anchor,
a real span, or was missing entirely (falling back to
NON_DOMINANT_HAND or nothing) -- real counts, not an impression from
a single hand-inspected example.
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path

XML_DIR = Path("data/raw/asllrp/xml")
SAMPLE_SIZE = 30  # utterances, not signs -- gives a real spread across
                   # different collections/signers, not just one file

random.seed(31)  # reproducible, per this project's standing preference
                  # for deterministic rather than freshly-random sampling


def extract_span(sign_el, tag):
    el = sign_el.find(tag)
    if el is None:
        return None
    try:
        s = int(el.get("START_FRAME", -1))
        e = int(el.get("END_FRAME", -1))
    except (TypeError, ValueError):
        return None
    if s <= 0 or e < s:
        return None
    return (s, e)


def main():
    xml_files = sorted(XML_DIR.glob("*.xml"))
    print(f"{len(xml_files)} XML files found")

    all_utterances = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for utt in tree.getroot().iter("UTTERANCE"):
            all_utterances.append((xml_path, utt))

    print(f"{len(all_utterances)} total utterances found")
    sample = random.sample(all_utterances, min(SAMPLE_SIZE, len(all_utterances)))

    total_signs = 0
    point_anchor_dominant = 0     # DOMINANT_HAND, S == E
    real_span_dominant = 0        # DOMINANT_HAND, S != E
    missing_dominant_has_nondominant = 0
    missing_both = 0
    span_widths = []              # for real spans only

    sign_type_counts = {}         # e.g. Lexical vs Fingerspelled vs Number
    sign_type_point_counts = {}

    for xml_path, utt in sample:
        for sign in utt.iter("SIGN"):
            total_signs += 1

            sign_type_el = sign.find("SIGN_TYPE")
            sign_type = sign_type_el.text.strip("'") if sign_type_el is not None and sign_type_el.text else "UNKNOWN"
            sign_type_counts[sign_type] = sign_type_counts.get(sign_type, 0) + 1

            dom = extract_span(sign, "DOMINANT_HAND")
            if dom is not None:
                s, e = dom
                if s == e:
                    point_anchor_dominant += 1
                    sign_type_point_counts[sign_type] = sign_type_point_counts.get(sign_type, 0) + 1
                else:
                    real_span_dominant += 1
                    span_widths.append(e - s + 1)
                continue

            nondom = extract_span(sign, "NON_DOMINANT_HAND")
            if nondom is not None:
                missing_dominant_has_nondominant += 1
            else:
                missing_both += 1

    print(f"\n=== Results across {len(sample)} sampled utterances, {total_signs} total signs ===")
    print(f"DOMINANT_HAND real span (S != E):     {real_span_dominant} ({100*real_span_dominant/total_signs:.1f}%)")
    print(f"DOMINANT_HAND point anchor (S == E):   {point_anchor_dominant} ({100*point_anchor_dominant/total_signs:.1f}%)")
    print(f"DOMINANT_HAND missing, NON_DOM present: {missing_dominant_has_nondominant} ({100*missing_dominant_has_nondominant/total_signs:.1f}%)")
    print(f"Both missing:                           {missing_both} ({100*missing_both/total_signs:.1f}%)")

    if span_widths:
        span_widths.sort()
        n = len(span_widths)
        print(f"\nReal span widths (frames): min={span_widths[0]}, "
              f"median={span_widths[n//2]}, max={span_widths[-1]}, "
              f"mean={sum(span_widths)/n:.1f}")

    print(f"\n=== Point-anchor rate by SIGN_TYPE ===")
    for stype, count in sorted(sign_type_counts.items(), key=lambda x: -x[1]):
        pt = sign_type_point_counts.get(stype, 0)
        print(f"  {stype}: {pt}/{count} point-anchored ({100*pt/count:.1f}%)")


if __name__ == "__main__":
    main()