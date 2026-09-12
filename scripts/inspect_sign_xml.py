import xml.etree.ElementTree as ET
from pathlib import Path

xml_dir = Path("data/raw/asllrp/xml")
xml_path = sorted(xml_dir.glob("*.xml"))[0]  # adjust if utterance 8400279 isn't in the first file

tree = ET.parse(xml_path)
root = tree.getroot()

for utterance in root.iter("UTTERANCE"):
    if utterance.get("ID", "") != "8400279":
        continue
    for i, sign in enumerate(utterance.iter("SIGN")):
        label_el = sign.find("LABEL")
        label = label_el.text if label_el is not None else "?"
        print(f"\n--- SIGN {i}: {label} ---")
        print(ET.tostring(sign, encoding="unicode"))
    break