"""Render the verified Stage 11 Markdown report and its figures to PDF."""
from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Spacer

from stage11_core import OUT

SOURCE=OUT/"reports/STAGE11_INTERVENTION_GEOMETRY_REPORT.md"
TARGET=OUT/"reports/STAGE11_INTERVENTION_GEOMETRY_REPORT.pdf"
FONT=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
BOLD=Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
if FONT.exists() and BOLD.exists():
    pdfmetrics.registerFont(TTFont("StageArial",str(FONT)))
    pdfmetrics.registerFont(TTFont("StageArialBold",str(BOLD)))
    pdfmetrics.registerFontFamily("StageArial",normal="StageArial",bold="StageArialBold")
    FAMILY="StageArial"
else:
    FAMILY="Helvetica"


def markup(value: str) -> str:
    # ReportLab's built-in fonts and source layout are clearer with ASCII dashes.
    value=value.replace("—","-").replace("–","-").replace("×","x")
    value=html.escape(value)
    value=re.sub(r"\*\*(.*?)\*\*",r"<b>\1</b>",value)
    value=re.sub(r"`([^`]+)`",r"<font color='#34536a'>\1</font>",value)
    value=re.sub(r"\[([^]]+)\]\(([^)]+)\)",r"<u>\1</u>",value)
    return value


def footer(canvas,doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(colors.HexColor("#cad6df"))
    canvas.line(1.5*cm,1.35*cm,w-1.5*cm,1.35*cm)
    canvas.setFont(FAMILY,8)
    canvas.setFillColor(colors.HexColor("#456172"))
    canvas.drawString(1.5*cm,1.02*cm,"SCAT Stage 11 | Pilot week 20161015-20161021 | FL290-FL410")
    canvas.drawRightString(w-1.5*cm,1.02*cm,str(doc.page))
    canvas.restoreState()


def make_pdf():
    sheet=getSampleStyleSheet()
    styles={
        "title":ParagraphStyle("STitle",parent=sheet["Title"],fontName=FAMILY,fontSize=17,leading=22,
                                textColor=colors.HexColor("#153d59"),spaceAfter=14),
        "h2":ParagraphStyle("SH2",parent=sheet["Heading2"],fontName=FAMILY,fontSize=13,leading=17,
                             textColor=colors.HexColor("#153d59"),spaceBefore=15,spaceAfter=9,keepWithNext=True),
        "body":ParagraphStyle("SBody",parent=sheet["BodyText"],fontName=FAMILY,fontSize=9.2,leading=13.4,
                                spaceAfter=8),
        "caption":ParagraphStyle("SCaption",parent=sheet["Normal"],fontName=FAMILY,fontSize=8.2,leading=11,
                                   alignment=TA_CENTER,textColor=colors.HexColor("#456172"),spaceAfter=8),
    }
    page_w,page_h=A4
    doc=BaseDocTemplate(str(TARGET),pagesize=A4,leftMargin=1.55*cm,rightMargin=1.55*cm,
                        topMargin=1.6*cm,bottomMargin=1.7*cm,
                        title="SCAT Stage 11 - ATC Intervention Geometry and Encounter Dynamics",
                        author="SCAT TFM analysis")
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="normal")
    doc.addPageTemplates(PageTemplate(id="scientific",frames=[frame],onPage=footer))
    story=[]
    lines=SOURCE.read_text(encoding="utf-8").splitlines()
    paragraph=[]
    image_index=0
    with tempfile.TemporaryDirectory(prefix="stage11_pdf_") as scratch:
        temp=Path(scratch)
        def flush():
            if paragraph:
                text=" ".join(s.strip() for s in paragraph)
                story.append(Paragraph(markup(text),styles["body"]))
                paragraph.clear()
        for line in lines:
            if not line.strip():
                flush();continue
            if line.startswith("# "):
                flush();story.append(Paragraph(markup(line[2:]),styles["title"]));continue
            if line.startswith("## "):
                flush();story.append(Paragraph(markup(line[3:]),styles["h2"]));continue
            match=re.match(r"!\[([^]]*)\]\(([^)]+)\)",line)
            if match:
                flush()
                src=(SOURCE.parent/match.group(2)).resolve()
                if not src.exists():
                    raise FileNotFoundError(src)
                if src.name.startswith("STAGE11_DASHBOARD_PAGE_"):
                    story.append(Paragraph(markup(f"Composición de tres paneles disponible en alta resolución: {src.relative_to(OUT)}"),styles["body"]))
                    continue
                with PILImage.open(src) as original:
                    im=original.convert("RGB")
                    if im.width>1900:
                        im.thumbnail((1900,1900*im.height/im.width))
                    dest=temp/f"figure_{image_index:03d}.jpg"
                    im.save(dest,quality=89,optimize=True)
                    aspect=im.height/im.width
                width=doc.width
                height=min(width*aspect,17.5*cm)
                if height<width*aspect:
                    width=height/aspect
                flow=Image(str(dest),width=width,height=height)
                flow.hAlign="CENTER"
                caption=Paragraph(markup(match.group(1).replace("_"," ")),styles["caption"])
                story.append(KeepTogether([flow,Spacer(1,3),caption]))
                image_index+=1
                continue
            paragraph.append(line)
        flush()
        TARGET.parent.mkdir(parents=True,exist_ok=True)
        doc.build(story)
    return TARGET,image_index


if __name__=="__main__":
    path,n=make_pdf()
    print(f"{path} ({n} figures)")
