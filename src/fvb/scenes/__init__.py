from pathlib import Path

SEED_XML = Path(__file__).with_name("peg_in_hole.xml")


def seed_xml_text() -> str:
    return SEED_XML.read_text()
