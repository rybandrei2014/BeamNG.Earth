"""server/roofcolor.py and the earthfast roof_sample kernel (--roof-colors, the user 2026-11-08).

Synthetic rasters (no network, no real imagery): planted colour patches under known footprints through the
same ring -> pixel mapping tile_estimate uses."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast  # noqa: E402
import roofcolor  # noqa: E402

PARAMS = np.array([roofcolor.ERODE_PX, roofcolor.GREEN_GAP, roofcolor.SHADOW_MAX, roofcolor.MIN_KEEP])


def sample(img, ring, params=PARAMS):
    r = np.asarray(ring, np.float64)
    return np.asarray(fast.roof_sample(img, np.array([0, len(r)]), r, params))[0]


class KernelTest(unittest.TestCase):
    def test_red_roof_on_a_green_yard(self):
        img = np.zeros((40, 40, 3), np.uint8)
        img[:, :] = (60, 140, 60)
        img[10:30, 10:30] = (170, 70, 60)
        row = sample(img, [(11, 11), (29, 11), (29, 29), (11, 29)])
        self.assertTrue(np.isfinite(row[3]), str(row))
        self.assertEqual(tuple(row[:3]), (170.0, 70.0, 60.0))
        self.assertGreater(row[3], roofcolor.MIN_COVERAGE)
        self.assertLess(row[4], roofcolor.MAX_SPREAD)

    def test_canopy_is_no_roof(self):
        img = np.zeros((20, 20, 3), np.uint8)
        img[:, :] = (60, 200, 60)          # the whole patch that green: nothing kept
        self.assertTrue(np.isnan(sample(img, [(2, 2), (18, 2), (18, 18), (2, 18)])[3]))

    def test_shadow_is_no_roof(self):
        img = np.full((20, 20, 3), 10, np.uint8)
        self.assertTrue(np.isnan(sample(img, [(2, 2), (18, 2), (18, 18), (2, 18)])[3]))

    def test_two_rings_one_call(self):
        img = np.zeros((20, 40, 3), np.uint8)
        img[:, :20] = (200, 60, 60)
        img[:, 20:] = (60, 60, 200)
        r = np.array([(2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0)])
        off = np.array([0, 4, 8])
        pts = np.concatenate([r, r + [20.0, 0.0]])
        rows = np.asarray(fast.roof_sample(img, off, pts, PARAMS))
        self.assertEqual(rows.shape, (2, 5))
        self.assertEqual(tuple(rows[0, :3]), (200.0, 60.0, 60.0))
        self.assertEqual(tuple(rows[1, :3]), (60.0, 60.0, 200.0))

    def test_a_ring_of_two_pixels_is_nothing(self):
        img = np.full((40, 40, 3), 128, np.uint8)
        self.assertTrue(np.isnan(sample(img, [(19, 19), (21, 19), (21, 21), (19, 21)])[3]))


class Im:
    """A stand-in imagery.Imagery: paint() plants a colour over a footprint through tile_estimate's own
    mapping (the tile box x0, y0, x1, y1 and the raster's side), row 0 north. Raises instead when told to."""

    def __init__(self, x0, y0, x1, y1, side=None, raises=None, px=256):
        self.box = (x0, y0, x1, y1)
        self.raises = raises
        self.px = px
        self.rsize = (max(x1 - x0, y1 - y0) if side is None else side) + 2.0 * roofcolor.GUTTER_M
        self.img = np.full((px, px, 3), 120, np.uint8)     # a neutral, non-green photo
        self.calls = 0

    def paint(self, ring, rgb):
        s = self.px / self.rsize
        rx0, ry0 = self.box[0] - roofcolor.GUTTER_M, self.box[1] - roofcolor.GUTTER_M
        r = np.asarray(ring, np.float64)
        u = np.clip(((r[:, 0] - rx0) * s).astype(int), 0, self.px - 1)
        v = np.clip(((ry0 + self.rsize - r[:, 1]) * s).astype(int), 0, self.px - 1)
        self.img[v.min():v.max() + 1, u.min():u.max() + 1] = rgb

    def render(self, frame, x, y, size, px):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.img, 1.0


class EstimateTest(unittest.TestCase):
    BOX = (0.0, 0.0, 300.0, 300.0)
    RING = [(120.0, 120.0), (180.0, 120.0), (180.0, 180.0), (120.0, 180.0)]

    def est(self, im, cands=None):
        return roofcolor.tile_estimate(im, None, *self.BOX, cands or [(0, "B1", np.array(self.RING))])

    def test_planted_colour_comes_back(self):
        im = Im(*self.BOX)
        im.paint(self.RING, (170, 70, 60))
        self.assertEqual(self.est(im), {"B1": "#aa463c"})

    def test_two_materials_stay_predicted(self):
        im = Im(*self.BOX)
        im.paint([(120.0, 120.0), (180.0, 120.0), (180.0, 150.0), (120.0, 150.0)], (170, 70, 60))
        im.paint([(120.0, 150.0), (180.0, 150.0), (180.0, 180.0), (120.0, 180.0)], (20, 30, 200))
        self.assertEqual(self.est(im), {}, "half and half of two materials: no colour")

    def test_canopy_stays_predicted(self):
        im = Im(*self.BOX)
        im.paint(self.RING, (60, 200, 60))                 # under a tree
        self.assertEqual(self.est(im), {})

    def test_shadow_stays_predicted(self):
        im = Im(*self.BOX)
        im.paint(self.RING, (12, 12, 12))
        self.assertEqual(self.est(im), {})

    def test_a_failure_propagates_to_the_caller(self):
        with self.assertRaises(RuntimeError):
            self.est(Im(*self.BOX, raises=RuntimeError("imagery: no photo")))

    def test_no_candidates_no_render(self):
        im = Im(*self.BOX)
        self.assertEqual(roofcolor.tile_estimate(im, None, *self.BOX, []), {})
        self.assertEqual(im.calls, 0)

    def test_a_building_drawn_in_parts_asks_once(self):
        im = Im(*self.BOX)
        im.paint(self.RING, (170, 70, 60))
        ring = np.array(self.RING)
        self.assertEqual(self.est(im, [(0, "B1", ring), (1, "B1", ring), (2, "B2", ring)]),
                         {"B1": "#aa463c", "B2": "#aa463c"})
        ring2 = np.array([(120.0, 120.0), (180.0, 120.0), (180.0, 150.0), (120.0, 150.0)])
        self.assertEqual(roofcolor.tile_estimate(im, None, *self.BOX, [(0, "B1", ring), (5, "B1", ring2)]),
                         {"B1": "#aa463c"}, "the outline's sample, once (parts share the look)")


if __name__ == "__main__":
    unittest.main()
