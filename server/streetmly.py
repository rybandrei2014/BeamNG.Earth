"""Facade colours from the street photos (earthd --facade-colours; PLAN.md M2, the user 2026-11-09: "apply
realistic textures onto buildings ... or at least realistic colors").

Overhead photos can never colour a facade - that is physics, not licensing - so the walls come from Mapillary:
the free, contributor-licensed street imagery whose API the benchmark tool already speaks (tools/streetview.py;
Google Street View stays out, its terms forbid this). For one geotile the job asks the Graph API for the newest
perspective photos in the tile's box, downloads each one's 1024 px thumbnail, and hands the tile's buildings'
wall quads to the earthfast kernel facade_look: the wall's four corners are projected with the photo's own
pinhole model (level camera, so only walls near the image centre are trusted, and none closer than MIN_DIST_M),
and the wall's pixels are clustered in Lab - the largest mode that is neither window-dark nor ground-dark is the
painted wall. A building's colour is the median of its confident wall rows; rows that disagree (two facades,
or a bad projection) leave it predicted.

Precedence: a mapped OSM/Overture facade_color always wins (only unmapped buildings are candidates), and the
colour enters through the existing facade_color -> custom_look path, so nothing on the game side changes.
The token is the player's own Mapillary client token (settings page, $BNGE_MAPILLARY_TOKEN, or
~/.config/beamng-earth/mapillary.token): a shared token behind every player would be a rate-limit and abuse
problem. Requests carry the project's USER_AGENT through the netfetch session like every other source;
photos are read-only fuel for colours - only the extracted hexes are kept (the tile's colour file), never
image bytes in the world cache. The photos are CC BY-SA 4.0 by their contributors (credits.py, SOURCES.md).
Off (default) and in --offline: nothing here runs.
"""
import json
import logging
import math
import os

import numpy as np

import fast

log = logging.getLogger(__name__)

API = "https://graph.mapillary.com/images"
TOKEN_ENV = "BNGE_MAPILLARY_TOKEN"
TOKEN_FILE = os.path.expanduser("~/.config/beamng-earth/mapillary.token")
FIELDS = ("id,computed_geometry,captured_at,camera_type,camera_parameters,is_pano,thumb_1024_url,"
          "width,height,sequence")
LANE = "streetmly"        # (netfetch lane: Mapillary's own bucket, so a photo outage never parks the map data)

AGL_M = 2.0            # m: street photos sit on a car roof or a dashboard (tools/streetview.py:46)
MIN_H_PX = 40.0        # px a wall's projection must stand tall: a speck is not a facade
MAX_OFF_FRAC = 0.3     # of the half-frame a wall's centre may sit off: the unknown pitch bends the edges
MIN_DIST_M = 8.0       # m nearer than this a photo is too oblique for its wall's pixels to be usable
L_DARK = 25.0          # Lab L under this is window glass and ground shadow, not the painted wall
MIN_SHARE = 0.3        # of the wall's lit pixels one colour must wear to be the wall's own
MIN_KEEP = 40.0        # pixels a mode must hold to count
L_TOL, AB_TOL = 4.0, 8.0
PARAMS = np.array([MIN_H_PX, MAX_OFF_FRAC, MIN_DIST_M, L_DARK, MIN_SHARE, MIN_KEEP, L_TOL, AB_TOL])

MIN_PHOTOS = 2         # confident photos per building: one photo can be a reflection, two agree
MAX_PHOTOS = 12        # thumbnails one tile's job reads at most (API etiquette, and the tile has work to do)
MAX_SPREAD = 24.0      # 0..255 across the agreed rows: above it the building shows two facades, not one
DEFAULT_F_WIDTH = 0.714  # px per width unit for a photo without camera parameters (~70 deg, streetview.py:92)
M_PER_DEG = 111320.0


def token():
    """The player's own Mapillary client token, or None: the env wins over the settings file."""
    t = (os.environ.get(TOKEN_ENV) or "").strip()
    if not t:
        try:
            with open(TOKEN_FILE) as fh:
                t = fh.read().strip()
        except OSError:
            t = ""
    return t if t.startswith("MLY|") else None


def _get(net, url, tok):
    """One authorised GET as bytes; its own function so a test can feed canned photos and metadata."""
    return net.get(url, kind="background", headers={"Authorization": f"OAuth {tok}"}, lane=LANE).body


def photos(net, tok, lat, lon, half_m):
    """The tile's usable photos, newest first, at most MAX_PHOTOS: perspective cameras with a position, a
    heading and a thumbnail (panoramas and fisheyes have no usable pinhole)."""
    dlat = half_m / M_PER_DEG
    dlon = half_m / (M_PER_DEG * max(0.05, math.cos(math.radians(lat))))
    import urllib.parse
    q = urllib.parse.urlencode({"fields": FIELDS, "access_token": tok,
                                "bbox": f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}",
                                "limit": 200})
    data = json.loads(_get(net, f"{API}?{q}", tok)).get("data", [])
    ok = []
    for im in data:
        g = (im.get("computed_geometry") or {}).get("coordinates")
        hd = im.get("computed_compass_angle", im.get("compass_angle"))
        if not g or hd is None or im.get("is_pano") or im.get("camera_type") not in (None, "perspective"):
            continue
        if not im.get("thumb_1024_url"):
            continue
        ok.append(im)
    ok.sort(key=lambda im: -(im.get("captured_at") or 0))
    return ok[:MAX_PHOTOS]


def intrinsics(im, w, h):
    """(fx, fy, cx, cy) in thumbnail pixels: Mapillary's focal is relative to the full image's larger side
    and normalised to the width; the decoded thumb's own size scales it (cx, cy at the centre)."""
    p = im.get("camera_parameters") or []
    fw = DEFAULT_F_WIDTH
    if p and p[0] and p[0] > 0.1:
        fw = float(p[0]) * max(im.get("width") or w, im.get("height") or h) / max(1, im.get("width") or w)
    return fw * w, fw * w, w / 2.0, h / 2.0


def decode(url, net, tok):
    """A thumbnail URL's (h, w, 3) uint8 RGB."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(_get(net, url, tok))).convert("RGB")
    return np.asarray(im, np.uint8)


def photo_rows(net, tok, im, frame, walls):
    """(W, 5) facade rows of one photo for the tile's wall quads (frame metres, z up): cam at the photo's
    position AGL_M above the ground, level, along its heading; nan rows are walls this photo cannot show."""
    g = im["computed_geometry"]["coordinates"]
    e, n, _u = frame.geodetic_to_enu(g[1], g[0], AGL_M)
    rgb = decode(im["thumb_1024_url"], net, tok)
    fx, fy, cx, cy = intrinsics(im, rgb.shape[1], rgb.shape[0])
    yaw = math.radians(float(im.get("computed_compass_angle", im.get("compass_angle"))))
    return np.asarray(fast.facade_look(rgb, (e, n, AGL_M), yaw, (fx, fy, cx, cy), walls, PARAMS))


def colours_file(cache_dir, tx, ty):
    """The tile's finished colour file (the background job's output; the tile build reads only this). tx:
    an anchor tile id or a geotiles.GeoBox (its key)."""
    return os.path.join(cache_dir, f"bnge_facade_colours_{getattr(tx, 'key', tx)}_{ty}.json")


def colours_load(cache_dir, tx, ty):
    """The finished file's {building id: "#rrggbb"}, or None when the tile has none yet: a tile built before
    the colours arrive keeps its predicted facades, the next visit wears them."""
    try:
        with open(colours_file(cache_dir, tx, ty)) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    col = data.get("colours") if isinstance(data, dict) else None
    return col if isinstance(col, dict) else None


def colours_save(cache_dir, tx, ty, colours):
    """The background job's finished write: atomic, so a tile build never half-reads it."""
    path = colours_file(cache_dir, tx, ty)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"colours": colours}, fh, sort_keys=True)
    os.replace(tmp, path)


def run_job(net, cache_dir, tx, ty, frame, box, job):
    """The background job of one tile (blockworker._facade_job): when the tile has no colour file yet and the
    player has a token, the street photos are asked and the finished file written - an empty one means
    "looked, nothing agreed" and stops the next build asking again. The tile build itself never runs this."""
    import pacer
    tok = token()
    if tok is None or not job or colours_load(cache_dir, tx, ty) is not None:
        return
    pacer.pace()
    cands = [(0, k, np.asarray(w, np.float64)) for k, w in job]
    colours_save(cache_dir, tx, ty, tile_facade_colours(net, tok, frame, *box, cands))


def tile_facade_colours(net, tok, frame, x0, y0, x1, y1, cands):
    """cands: [(attr index, key (the building's id, str), walls (W, 4, 3) frame metres)] of the tile's
    buildings that have no mapped facade colour -> {key: "#rrggbb"} for those the street photos agree on.
    A building needs MIN_PHOTOS confident wall rows within MAX_SPREAD of each other; one photo, or walls
    that disagree, keeps its predicted look."""
    if not cands or tok is None:
        return {}
    keys = np.concatenate([np.full(len(w), j, np.int64) for j, (_i, k, w) in enumerate(cands)])
    walls = np.concatenate([w for _i, _k, w in cands])
    la, lo, _ = frame.enu_to_geodetic(0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0)
    half = 0.7 * max(x1 - x0, y1 - y0)
    out = {}
    got = {}
    try:
        ims = photos(net, tok, float(la), float(lo), half)
        import pacer
        for im in ims:
            try:
                rows = np.asarray(photo_rows(net, tok, im, frame, walls))
            except Exception as ex:
                log.debug("facade colour: %s: %s", im.get("id"), ex)
                continue
            pacer.pace()
            for wi, row in enumerate(rows):
                if not np.isfinite(row[3]):
                    continue
                got.setdefault(int(keys[wi]), []).append(row)
    except Exception as ex:
        log.info("facade colour: %s: %s", type(ex).__name__, ex)   # (the tile stays as it was: predicted roofs)
        return {}
    for j, (_i, key, _w) in enumerate(cands):
        rows = got.get(j)
        if not rows or len(rows) < MIN_PHOTOS:
            continue
        rgb = np.array([r[:3] for r in rows])
        med = np.median(rgb, axis=0)
        if float(np.max(np.median(np.abs(rgb - med), axis=0))) > MAX_SPREAD:
            continue
        out[key] = "#%02x%02x%02x" % tuple(int(round(v)) for v in med)
    return out
