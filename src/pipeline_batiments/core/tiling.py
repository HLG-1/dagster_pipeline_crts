"""
Calcul de la grille de tuiles chevauchantes sur une image de hauteur H,
largeur W. 
Utilise par : assets/tuilage.py (etape 02)
"""
from __future__ import annotations


def tile_positions(
    height: int,
    width: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Renvoie la liste des fenetres (row, col, tile_h, tile_w) couvrant
    toute l'image, avec chevauchement `overlap` px entre tuiles voisines.

    Garantit :
        - couverture complete de l'image (union des tuiles = image)
        - overlap respecte sur les bords internes (necessaire a la fusion
          des batiments coupes, etape 05)
    """
    if height <= tile_size and width <= tile_size:
        return [(0, 0, height, width)]


    stride = max(1, tile_size - overlap)
    rows = list(range(0, max(height - tile_size, 0) + 1, stride))
    cols = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not rows or rows[-1] + tile_size < height:
        rows.append(max(0, height - tile_size))
    if not cols or cols[-1] + tile_size < width:
        cols.append(max(0, width - tile_size))
    rows, cols = sorted(set(rows)), sorted(set(cols))

    out = []
    for r in rows:
        for c in cols:
            out.append((r, c, min(tile_size, height - r), min(tile_size, width - c)))
    return out
