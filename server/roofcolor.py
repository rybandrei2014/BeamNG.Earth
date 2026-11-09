"""Roof colours from the aerial photos (earthd --roof-colors; the user, 2026-11-08: "apply realistic colors to
them from the Earth view").

A building's roof is the one part of it the aerial photos actually see, and the server already downloads those
photos for the ground. So with --roof-colors on (and --buildings on), blockworker._tile_parts hands this
module's caller-side helper to buildings.tile_buildings(roof_im=...), which collects the tile's buildings that
have no mapped roof colour and asks tile_estimate() once for all of them: the tile's imagery rendered at about
TARG_PX_M a texel through the same source chain the ground's photo uses (the same tone curve and season
grading, so the colour matches the photo the player sees beside the building), sampled per footprint by the
earthfast kernel roof_sample (the ring eroded ERODE_PX off its edge; canopy green and hard shadow dropped;
median colour, coverage, spread - in Rust, not here).

Precedence: an OSM/Overture building:roof_colour / roof_color always wins (a building is only ever a candidate
when nothing is mapped), and the colour enters through the existing roof_color -> custom_look path: the mod
tints the roof sheet by vertex colour, so nothing on the game side changes. A roof the photo cannot show
clearly - under trees, built after the flight, too small, half yard - keeps its predicted material: a wrong
real colour is worse than a plausible guessed one, hence the gates below. Off (default): nothing here runs and
tiles are byte-identical to before.
"""
import logging

import numpy as np

import fast

log = logging.getLogger(__name__)

GUTTER_M = 24.0        # m of imagery fetched past the tile's box: a footprint overhanging its edge stays whole
TARG_PX_M = 0.75       # m a texel (~z17 at mid latitudes; each source's own resolution cap and the overzoom apply)
MIN_PX, MAX_PX = 256, 704
ERODE_PX = 1.0         # texels off the footprint edge: the wall top and parapets stand inside the ring
GREEN_GAP = 40.0       # 0..255 over both other channels: a texel that green is a tree top, not a roof
SHADOW_MAX = 45.0      # 0..255: at most this bright the photo sits in hard shadow: not the roof's colour
MIN_KEEP = 8           # texels left after the drops: under this the "roof" is a photo speck, not a roof
MIN_COVERAGE = 0.4     # and this share of the ring's inside texels must survive (a roof lost to trees stays guessed)
MAX_SPREAD = 32.0      # percentile width of the kept texels: above it the patch mixes two materials, or roof and yard


def tile_estimate(im, frame, x0, y0, x1, y1, cands):
    """cands: [(attr index, key (the building's id, str), ring (k, 2) frame metres)] of the tile's buildings with
    no mapped roof colour (buildings.tile_buildings; indices unused here). One imagery render for the tile, one
    roof_sample call for its rings; {key: "#rrggbb"} of the buildings whose sample passed the gates.

    The ring -> pixel mapping comes from the tile's own box, so the same building samples to the same colour
    under every anchor (a geotile's content keeps one colour across rebases). Raises (ImageryError, network):
    the caller keeps the predicted roofs."""
    if not cands:
        return {}
    side = max(x1 - x0, y1 - y0)
    rx0, ry0 = x0 - GUTTER_M, y0 - GUTTER_M
    rsize = side + 2.0 * GUTTER_M
    px = min(MAX_PX, max(MIN_PX, int(round(rsize / TARG_PX_M))))
    rgb, _used = im.render(frame, rx0, ry0, rsize, px)
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return {}
    s = min(rgb.shape[0], rgb.shape[1]) / rsize     # (the raster's own size, as Imagery.render's px)
    seen, off, pts, keys = set(), [0], [], []
    for _i, key, ring in cands:
        if key in seen:
            continue        # (a building drawn in parts asks once: the parts share the outline's colour)
        seen.add(key)
        rr = np.asarray(ring, np.float64)[:, :2]
        pts.append(np.stack([(rr[:, 0] - rx0) * s, (ry0 + rsize - rr[:, 1]) * s], 1))   # render row 0 = north
        off.append(off[-1] + len(pts[-1]))
        keys.append(key)
    if not keys:
        return {}
    rows = np.asarray(fast.roof_sample(rgb[:, :, :3], off, np.concatenate(pts),
                                       np.array([ERODE_PX, GREEN_GAP, SHADOW_MAX, MIN_KEEP], np.float64)))
    out = {}
    for key, (r, g, b, cov, spread) in zip(keys, rows):
        if not np.isfinite(cov) or cov < MIN_COVERAGE or spread > MAX_SPREAD:
            continue
        out[key] = "#%02x%02x%02x" % tuple(min(255, max(0, int(round(c)))) for c in (r, g, b))
    return out
