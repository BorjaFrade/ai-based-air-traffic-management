"""Render Stage 11D Markdown and its 21 figures to a paginated PDF."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

import stage11_pdf as renderer
from stage11d_core import OUT

renderer.OUT=OUT
renderer.SOURCE=OUT/"reports/STAGE11D_VALIDATION_REPORT.md"
renderer.TARGET=OUT/"reports/STAGE11D_VALIDATION_REPORT.pdf"


def footer(canvas,doc):
    canvas.saveState()
    w,_=A4
    canvas.setStrokeColor(colors.HexColor("#cad6df"))
    canvas.line(1.5*cm,1.35*cm,w-1.5*cm,1.35*cm)
    canvas.setFont(renderer.FAMILY,8)
    canvas.setFillColor(colors.HexColor("#456172"))
    canvas.drawString(1.5*cm,1.02*cm,"SCAT Stage 11D | Intervention geometry validation | two weeks")
    canvas.drawRightString(w-1.5*cm,1.02*cm,str(doc.page))
    canvas.restoreState()


renderer.footer=footer


if __name__=="__main__":
    target,n=renderer.make_pdf()
    print(f"{target} ({n} figures)")
