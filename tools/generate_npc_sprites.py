from pathlib import Path

from PIL import Image, ImageDraw


OUT_DIR = Path(__file__).resolve().parents[1] / "static" / "assets" / "npcs"
SCALE = 4


def rect(draw, xy, fill):
    draw.rectangle(xy, fill=fill)


def outline(draw, xy):
    rect(draw, xy, (15, 13, 12, 255))


def draw_base(draw, p):
    # Keep every major piece separated by transparent pixels before scaling.
    rect(draw, (13, 28, 34, 31), (0, 0, 0, 70))

    # Legs, separated from torso and from each other.
    rect(draw, (16, 21, 20, 29), p["outline"])
    rect(draw, (27, 21, 31, 29), p["outline"])
    rect(draw, (17, 21, 20, 27), p["pants"])
    rect(draw, (27, 21, 30, 27), p["pants"])
    rect(draw, (15, 29, 21, 32), p["shoes"])
    rect(draw, (26, 29, 32, 32), p["shoes"])

    # Torso with a visible shirt panel and coat edges.
    rect(draw, (14, 11, 33, 22), p["outline"])
    rect(draw, (15, 12, 21, 21), p["coat"])
    rect(draw, (22, 12, 25, 21), p["shirt"])
    rect(draw, (26, 12, 32, 21), p["coat_dark"])

    # Arms have one pixel of transparent gap from torso.
    rect(draw, (9, 13, 12, 22), p["outline"])
    rect(draw, (35, 13, 38, 22), p["outline"])
    rect(draw, (9, 13, 12, 17), p["sleeve"])
    rect(draw, (35, 13, 38, 17), p["sleeve"])
    rect(draw, (10, 18, 12, 22), p["skin"])
    rect(draw, (35, 18, 37, 22), p["skin"])

    # Neck and head are slightly separated from the torso silhouette.
    rect(draw, (21, 8, 26, 11), p["skin"])
    rect(draw, (16, 2, 31, 10), p["outline"])
    rect(draw, (17, 3, 30, 10), p["skin"])
    rect(draw, (16, 2, 31, 5), p["hair"])
    rect(draw, (16, 6, 18, 9), p["hair"])
    rect(draw, (29, 6, 31, 9), p["hair"])

    # Face pixels.
    rect(draw, (20, 7, 21, 8), p["eye"])
    rect(draw, (26, 7, 27, 8), p["eye"])
    rect(draw, (22, 10, 25, 10), p["mouth"])


def jordan(draw, p):
    # Red tie and tan folder, separated from the arm.
    rect(draw, (22, 12, 25, 20), (116, 38, 34, 255))
    rect(draw, (21, 20, 26, 21), (116, 38, 34, 255))
    rect(draw, (3, 16, 8, 27), p["outline"])
    rect(draw, (4, 17, 8, 26), (177, 128, 58, 255))
    rect(draw, (4, 16, 8, 17), (225, 184, 88, 255))


def lucas(draw, p):
    # White apron panel distinct from red sleeves; pan held off-body.
    rect(draw, (20, 11, 27, 23), (238, 226, 198, 255))
    rect(draw, (21, 12, 26, 22), (255, 244, 214, 255))
    rect(draw, (17, 1, 30, 3), (238, 226, 198, 255))
    rect(draw, (40, 18, 45, 23), p["outline"])
    rect(draw, (41, 19, 45, 22), (78, 55, 38, 255))


def morgan(draw, p):
    # Barber apron and scissors separated from the hand.
    rect(draw, (18, 12, 30, 23), (46, 89, 59, 255))
    rect(draw, (19, 13, 29, 15), (117, 151, 107, 255))
    rect(draw, (39, 18, 40, 25), (232, 232, 210, 255))
    rect(draw, (41, 17, 43, 19), (190, 190, 176, 255))
    rect(draw, (41, 23, 43, 25), (190, 190, 176, 255))


def detective(draw, p):
    # Noir detective: trench coat, hat brim, magnifier, and blue tie.
    rect(draw, (14, 0, 33, 2), p["outline"])
    rect(draw, (15, 1, 32, 2), (65, 55, 42, 255))
    rect(draw, (18, 2, 29, 5), (76, 65, 49, 255))
    rect(draw, (22, 12, 25, 20), (42, 78, 112, 255))
    rect(draw, (21, 20, 26, 21), (42, 78, 112, 255))
    rect(draw, (38, 16, 41, 19), (206, 220, 210, 255))
    rect(draw, (39, 17, 40, 18), (60, 90, 105, 255))
    rect(draw, (41, 20, 44, 23), p["outline"])
    rect(draw, (42, 21, 43, 22), (92, 66, 44, 255))


PALETTES = {
    "detective": {
        "skin": (199, 150, 99, 255),
        "hair": (38, 31, 25, 255),
        "coat": (132, 111, 72, 255),
        "coat_dark": (91, 76, 52, 255),
        "shirt": (222, 213, 181, 255),
        "sleeve": (111, 92, 62, 255),
        "pants": (47, 50, 61, 255),
        "shoes": (22, 21, 20, 255),
        "eye": (20, 18, 16, 255),
        "mouth": (83, 45, 39, 255),
        "outline": (16, 14, 12, 255),
    },
    "jordan": {
        "skin": (198, 145, 94, 255),
        "hair": (44, 30, 22, 255),
        "coat": (116, 82, 49, 255),
        "coat_dark": (76, 54, 35, 255),
        "shirt": (230, 214, 176, 255),
        "sleeve": (92, 65, 42, 255),
        "pants": (57, 54, 66, 255),
        "shoes": (23, 21, 21, 255),
        "eye": (23, 21, 18, 255),
        "mouth": (91, 45, 39, 255),
        "outline": (16, 14, 12, 255),
    },
    "lucas": {
        "skin": (191, 133, 80, 255),
        "hair": (29, 27, 25, 255),
        "coat": (126, 56, 46, 255),
        "coat_dark": (86, 41, 36, 255),
        "shirt": (238, 226, 198, 255),
        "sleeve": (126, 56, 46, 255),
        "pants": (55, 43, 35, 255),
        "shoes": (24, 20, 17, 255),
        "eye": (20, 18, 15, 255),
        "mouth": (96, 42, 38, 255),
        "outline": (16, 14, 12, 255),
    },
    "morgan": {
        "skin": (181, 126, 82, 255),
        "hair": (34, 34, 34, 255),
        "coat": (57, 85, 62, 255),
        "coat_dark": (38, 59, 45, 255),
        "shirt": (178, 160, 119, 255),
        "sleeve": (57, 85, 62, 255),
        "pants": (43, 45, 48, 255),
        "shoes": (22, 22, 22, 255),
        "eye": (17, 17, 17, 255),
        "mouth": (86, 48, 43, 255),
        "outline": (16, 14, 12, 255),
    },
}

EXTRAS = {
    "detective": detective,
    "jordan": jordan,
    "lucas": lucas,
    "morgan": morgan,
}


def generate(name):
    image = Image.new("RGBA", (48, 36), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    palette = PALETTES[name]
    draw_base(draw, palette)
    EXTRAS[name](draw, palette)
    image = image.resize((192, 144), Image.Resampling.NEAREST)
    image.save(OUT_DIR / f"{name}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("detective", "jordan", "lucas", "morgan"):
        generate(name)
        print(OUT_DIR / f"{name}.png")


if __name__ == "__main__":
    main()
