from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfgen import canvas


def annotation_gap(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = 520, 265
    page = canvas.Canvas(str(output), pagesize=(width, height))
    page.setFillColor(colors.white)
    page.rect(0, 0, width, height, fill=1, stroke=0)
    page.setFont("Helvetica-Bold", 13)
    page.setFillColor(colors.HexColor("#17171c"))
    page.drawString(18, 242, "Specialization helps, but safe Arabic annotation remains hard")

    labels = ["Binary toxicity", "Multi-label toxicity", "Dialect region"]
    supervised = [97.0, 49.0, 69.8]
    qwen = [78.0, 25.3, 22.4]
    fanar = [88.0, 31.3, 47.8]
    left, right = 145, 485
    scale = (right - left) / 100
    page.setFont("Helvetica", 8)
    for tick in (0, 25, 50, 75, 100):
        x = left + tick * scale
        page.setStrokeColor(colors.HexColor("#dedee7"))
        page.line(x, 52, x, 220)
        page.setFillColor(colors.HexColor("#60606c"))
        page.drawCentredString(x, 39, str(tick))

    page.setFont("Helvetica", 9)
    for index, label in enumerate(labels):
        y = 188 - index * 56
        page.setFillColor(colors.HexColor("#17171c"))
        page.drawRightString(left - 10, y + 4, label)
        for offset, value, color in (
            (13, supervised[index], "#17171c"),
            (-1, fanar[index], "#12a594"),
            (-15, qwen[index], "#5b4df2"),
        ):
            page.setFillColor(colors.HexColor(color))
            page.roundRect(left, y + offset, value * scale, 10, 3, fill=1, stroke=0)
            page.setFont("Helvetica-Bold", 8)
            page.drawString(left + value * scale + 5, y + offset + 1, f"{value:.1f}")

    page.setFont("Helvetica", 8)
    page.setFillColor(colors.HexColor("#17171c"))
    page.rect(18, 11, 8, 8, fill=1, stroke=0)
    page.drawString(31, 12, "supervised diagnostic")
    page.setFillColor(colors.HexColor("#12a594"))
    page.rect(139, 11, 8, 8, fill=1, stroke=0)
    page.setFillColor(colors.HexColor("#17171c"))
    page.drawString(152, 12, "Fanar 1 9B zero-shot")
    page.setFillColor(colors.HexColor("#5b4df2"))
    page.rect(259, 11, 8, 8, fill=1, stroke=0)
    page.setFillColor(colors.HexColor("#17171c"))
    page.drawString(272, 12, "Qwen 2.5 3B zero-shot")
    page.setFont("Helvetica-Oblique", 7)
    page.setFillColor(colors.HexColor("#60606c"))
    page.drawRightString(504, 12, "Exact match; held-out paired samples")
    page.showPage()
    page.save()


if __name__ == "__main__":
    annotation_gap(Path("paper/figures/annotation-gap.pdf"))
