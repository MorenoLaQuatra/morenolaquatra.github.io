#!/usr/bin/env python3
"""Estrae il testo dei tre input della demo, senza inviarlo in rete.

Dalla cartella del materiale:
  uv run --no-project --with pymupdf python leggi-documenti.py
Dopo aver preparato la cache:
  uv run --offline --no-project --with pymupdf python leggi-documenti.py

Non e' un lettore universale: questo esempio gestisce il PDF testuale,
il DOCX e il CSV inclusi. Non esegue OCR e non modifica gli input.
"""
from pathlib import Path
from zipfile import ZipFile
import sys
import xml.etree.ElementTree as ET


def estrai(cartella: Path) -> str:
    import pymupdf

    blocchi = []
    with pymupdf.open(cartella / "menu.pdf") as documento:
        testo_pdf = "\n".join(pagina.get_text() for pagina in documento)
    if not testo_pdf.strip():
        raise ValueError("Il PDF non contiene testo estraibile: serve un altro lettore o OCR")
    blocchi.append(("menu.pdf", testo_pdf.strip()))

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(cartella / "ricette.docx") as documento:
        xml = ET.fromstring(documento.read("word/document.xml"))
    paragrafi = []
    for paragrafo in xml.findall(".//w:p", ns):
        testo = "".join(nodo.text or "" for nodo in paragrafo.findall(".//w:t", ns))
        if testo:
            paragrafi.append(testo)
    if not paragrafi:
        raise ValueError("Il DOCX non contiene paragrafi leggibili")
    blocchi.append(("ricette.docx", "\n".join(paragrafi)))
    blocchi.append(("dispensa.csv", (cartella / "dispensa.csv").read_text(encoding="utf-8-sig").strip()))
    return "\n\n".join(f"=== {nome} ===\n{testo}" for nome, testo in blocchi)


if __name__ == "__main__":
    try:
        base = Path(__file__).resolve().parent / "cena"
        print(estrai(base))
    except (ImportError, OSError, ValueError, KeyError, ET.ParseError) as errore:
        print(f"Lettura non riuscita: {errore}", file=sys.stderr)
        sys.exit(1)
