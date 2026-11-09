"""server/streetmly.py (--facade-colours, PLAN.md M2): the canned-API test - two photos of one painted wall
agree, one photo does not, disagreeing photos keep the prediction, and failures leave the tile untouched.
No network: streetmly._get feeds JSON and JPEG fixtures through the same paths the live job uses."""
import io
import json
import math
import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import geodesy  # noqa: E402
import streetmly  # noqa: E402

LAT, LON = 39.2834, -76.5934
FRAME = geodesy.EnuFrame(LAT, LON, 0.0)
STUCCO = (200, 90, 70)

# the wall: 8 m wide at 20 m north of the photo, ground to an 8 m eave, facing south (the camera looks north)
WALL = np.array([[-4.0, 20.0, 0.0], [4.0, 20.0, 0.0], [4.0, 20.0, 8.0], [-4.0, 20.0, 8.0]])
CANDS = [(0, "B1", WALL[None, :, :])]


def photo_jpeg(rgb=STUCCO):
    """A 1024x768 thumbnail: the wall fills its projection (rows 15..507 of the test rig), sky above, road below."""
    img = np.zeros((768, 1024, 3), np.uint8)
    img[:, :] = (120, 160, 220)          # sky
    img[15:508, :] = rgb                 # the wall's own pixels
    img[508:, :] = (40, 40, 44)          # road and kerb: too dark to vote
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def meta(e=0.0, n=0.0, pid="p1"):
    """A Mapillary image record for a camera at frame (e, n) looking north."""
    la, lo, _ = FRAME.enu_to_geodetic(e, n, 0.0)
    return {"id": pid, "computed_geometry": {"coordinates": [float(lo), float(la)]},
            "compass_angle": 0.0, "captured_at": 1730000000000, "camera_type": "perspective",
            "camera_parameters": [1.2, 0.0, 0.0], "is_pano": False, "width": 4000, "height": 3000,
            "thumb_1024_url": f"https://a.mapillary.com/{pid}.jpg", "sequence": "s1"}


class FakeNet:
    """Answers the search with the given photos and each thumb URL with canned JPEGs."""

    def __init__(self, ims, jpegs):
        self.ims, self.jpegs, self.urls = ims, jpegs, []

    def get(self, url, **kw):
        self.urls.append(url)
        if "graph.mapillary.com" in url:
            body = json.dumps({"data": self.ims}).encode("utf-8")
        else:
            body = self.jpegs[url.split("/")[-1]]

        class R:
            pass
        r = R()
        r.body = body
        return r


class FacadeTest(unittest.TestCase):
    def run_tile(self, ims, jpegs):
        net = FakeNet(ims, jpegs)
        with mock.patch.object(streetmly, "_get", lambda net, url, tok: net.get(url).body):
            return streetmly.tile_facade_colours(net, "MLY|tok", FRAME, 0.0, 0.0, 256.0, 256.0, CANDS)

    def test_two_agreeing_photos_paint_the_wall(self):
        jpegs = {"p1.jpg": photo_jpeg(), "p2.jpg": photo_jpeg()}
        got = self.run_tile([meta(pid="p1"), meta(e=2.0, pid="p2")], jpegs)
        self.assertIn("B1", got, str(got))
        rgb = np.array([int(got["B1"][i:i + 2], 16) for i in (1, 3, 5)])
        self.assertLess(float(np.abs(rgb - np.array(STUCCO)).max()), 8.0,
                        f"the JPEG round trip may shift a level or two: {got['B1']}")

    def test_one_photo_is_not_enough(self):
        got = self.run_tile([meta()], {"p1.jpg": photo_jpeg()})
        self.assertEqual(got, {}, "one photo can be a shop window's reflection")

    def test_disagreement_stays_predicted(self):
        jpegs = {"p1.jpg": photo_jpeg(STUCCO), "p2.jpg": photo_jpeg((60, 60, 200))}
        got = self.run_tile([meta(pid="p1"), meta(e=2.0, pid="p2")], jpegs)
        self.assertEqual(got, {}, "two facades, or a bad projection: keep the predicted colour")

    def test_a_failing_search_leaves_the_tile_alone(self):
        net = FakeNet([], {})

        def boom(url, **kw):
            raise OSError("no route to graph.mapillary.com")
        net.get = boom
        with mock.patch.object(streetmly, "_get", lambda net, url, tok: net.get(url).body):
            self.assertEqual(streetmly.tile_facade_colours(net, "MLY|tok", FRAME, 0, 0, 256, 256, CANDS), {})

    def test_no_candidates_ask_for_nothing(self):
        self.assertEqual(streetmly.tile_facade_colours(None, "MLY|tok", FRAME, 0, 0, 1, 1, []), {})

    def test_no_token_asks_for_nothing(self):
        self.assertEqual(streetmly.tile_facade_colours(object(), None, FRAME, 0, 0, 1, 1, CANDS), {})


class TokenTest(unittest.TestCase):
    def test_env_wins_and_junk_is_no_token(self):
        with mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: "MLY|abc"}):
            self.assertEqual(streetmly.token(), "MLY|abc")
        with mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: "hunter2"}):
            self.assertIsNone(streetmly.token())
        with mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: ""}), mock.patch.object(
                streetmly, "TOKEN_FILE", os.devnull + "_missing"):
            self.assertIsNone(streetmly.token())


if __name__ == "__main__":
    unittest.main()


class JobTest(unittest.TestCase):
    """run_job: a tile without a colour file gets one made (or an empty one when nothing agreed), and a tile
    that already has it asks for nothing."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()

    def test_a_missing_file_is_made(self):
        net = FakeNet([meta(pid="p1"), meta(e=2.0, pid="p2")], {"p1.jpg": photo_jpeg(), "p2.jpg": photo_jpeg()})
        with mock.patch.object(streetmly, "_get", lambda net, url, tok: net.get(url).body), \
                mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: "MLY|tok"}):
            streetmly.run_job(net, self.dir, 7, 3, FRAME, (0.0, 0.0, 256.0, 256.0),
                              [("B1", np.asarray(WALL[None]))])
        self.assertEqual(streetmly.colours_load(self.dir, 7, 3), {"B1": streetmly.colours_load(self.dir, 7, 3)["B1"]})

    def test_an_existing_file_is_left_alone(self):
        streetmly.colours_save(self.dir, 7, 3, {})

        class NoNet:
            def get(self, *a, **kw):
                raise AssertionError("no fetches")
        with mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: "MLY|tok"}):
            streetmly.run_job(NoNet(), self.dir, 7, 3, FRAME, (0.0, 0.0, 1.0, 1.0), [("B1", np.asarray(WALL[None]))])

    def test_no_token_no_file(self):
        with mock.patch.dict(os.environ, {streetmly.TOKEN_ENV: ""}), mock.patch.object(
                streetmly, "TOKEN_FILE", os.devnull + "_missing"):
            streetmly.run_job(object(), self.dir, 7, 4, FRAME, (0.0, 0.0, 1.0, 1.0),
                              [("B1", np.asarray(WALL[None]))])
        self.assertIsNone(streetmly.colours_load(self.dir, 7, 4))
