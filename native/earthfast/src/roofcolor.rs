//! Roof colours from the aerial photos (server/roofcolor.py, PLAN.md M1): each building's footprint ring, in the
//! raster pixels of the tile's imagery render, samples that raster inside the ring (eroded off its edge, canopy
//! green and hard shadow dropped) into a median colour, a coverage and a spread, per building.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

/// The ray-cast point-in-polygon test; ring closed by wrapping.
fn inside_ring(ring: &[(f64, f64)], u: f64, v: f64) -> bool {
    let mut hit = false;
    let n = ring.len();
    for i in 0..n {
        let (x1, y1) = ring[i];
        let (x2, y2) = ring[(i + 1) % n];
        if (y1 > v) != (y2 > v) {
            let t = (v - y1) / (y2 - y1);
            if u < x1 + t * (x2 - x1) {
                hit = !hit;
            }
        }
    }
    hit
}

/// The squared distance from (u, v) to the ring's edges.
fn dist2_to_ring(ring: &[(f64, f64)], u: f64, v: f64) -> f64 {
    let mut best = f64::INFINITY;
    let n = ring.len();
    for i in 0..n {
        let (x1, y1) = ring[i];
        let (x2, y2) = ring[(i + 1) % n];
        let (dx, dy) = (x2 - x1, y2 - y1);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((u - x1) * dx + (v - y1) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (ex, ey) = (x1 + t * dx - u, y1 + t * dy - v);
        best = best.min(ex * ex + ey * ey);
    }
    best
}

/// The median of v (sorted in place); the mean of the two middle values for an even count.
fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// The spread of v: its 84th percentile minus its 16th (v sorted). Robust to a few stray texels, and - unlike
/// the median absolute deviation - it sees a roof that is half one material and half another, where the
/// majority alone fills the middle quantiles.
fn spread(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len();
    if m < 2 {
        return 0.0;
    }
    v[(m as f64 * 0.84) as usize] - v[(m as f64 * 0.16) as usize]
}

/// The per-ring sample rows (see roof_sample): pure, so it runs with and without the interpreter.
/// src is the raster (h * w * 3, row 0 north); offs (n + 1) into pts; a row is [nan; 5] when the ring kept
/// fewer than min_keep texels or has no texel inside the image.
fn sample_rows(src: &[u8], h: usize, w: usize, offs: &[usize], pts: &[(f64, f64)], erode: f64, gap: f64,
               shadow: f64, min_keep: usize) -> Vec<[f64; 5]> {
    let n = offs.len().saturating_sub(1);
    let er2 = erode * erode;
    let mut rows = vec![[f64::NAN; 5]; n];
    for b in 0..n {
        let (a, z) = (offs[b], offs[b + 1].min(pts.len()));
        if z.saturating_sub(a) < 3 {
            continue;
        }
        let ring = &pts[a..z];
        let (u0, u1) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |ac, &(u, _)| {
            (ac.0.min(u), ac.1.max(u))
        });
        let (v0, v1) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |ac, &(_, v)| {
            (ac.0.min(v), ac.1.max(v))
        });
        // texel centres (x + 0.5, y + 0.5) inside the ring's bbox, clamped to the image
        let (x0, x1) = ((u0 - 0.5).ceil().max(0.0) as usize, (u1 - 0.5).floor().min(w as f64 - 1.0) as usize);
        let (y0, y1) = ((v0 - 0.5).ceil().max(0.0) as usize, (v1 - 0.5).floor().min(h as f64 - 1.0) as usize);
        if u0.is_infinite() || x1 < x0 || y1 < y0 {
            continue;
        }
        let (mut kr, mut kg, mut kb) = (Vec::new(), Vec::new(), Vec::new());
        let mut inside_ct = 0usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (u, v) = (x as f64 + 0.5, y as f64 + 0.5);
                if !inside_ring(ring, u, v) {
                    continue;
                }
                inside_ct += 1;
                if dist2_to_ring(ring, u, v) <= er2 {
                    continue;     // (the wall top and the parapet read one texel inside the footprint)
                }
                let q = (y * w + x) * 3;
                let (r, g, bl) = (src[q] as f64, src[q + 1] as f64, src[q + 2] as f64);
                if g > r + gap && g > bl + gap {
                    continue;     // (canopy: the photo shows a tree top there, not the roof)
                }
                if r.max(g).max(bl) < shadow {
                    continue;     // (hard shadow: the value is the shadow's, not the roof's colour)
                }
                kr.push(r);
                kg.push(g);
                kb.push(bl);
            }
        }
        if kr.len() < min_keep || inside_ct == 0 {
            continue;
        }
        let (mr, mg, mb) = (median(&mut kr), median(&mut kg), median(&mut kb));
        let spread = (spread(&mut kr) + spread(&mut kg) + spread(&mut kb)) / 3.0;
        rows[b] = [mr, mg, mb, kr.len() as f64 / inside_ct as f64, spread];
    }
    rows
}

/// roof_sample(rgb, off, pts, params) -> (n, 5) rows: per-building median r, g, b (0..255), coverage (kept /
/// the ring's texels inside the image) and spread (the 84th minus the 16th percentile of the kept texels, averaged
/// over the channels); a nan row where too few texels survived. rgb: (h, w, 3) the tile's imagery (row 0 north);
/// off: (n + 1) offsets of each ring into pts; pts: (P, 2) the rings' corners in raster pixels (col, row);
/// params: (erode_px, green_gap, shadow_max, min_keep). server/roofcolor.py turns surviving rows into hexes.
#[pyfunction]
#[pyo3(signature = (rgb, off, pts, params))]
pub fn roof_sample<'py>(py: Python<'py>, rgb: PyReadonlyArray3<'py, u8>, off: PyReadonlyArray1<'py, i64>,
                        pts: PyReadonlyArray2<'py, f64>, params: PyReadonlyArray1<'py, f64>)
                        -> PyResult<Bound<'py, PyArray2<f64>>> {
    let img = rgb.as_array();
    if img.ndim() != 3 || img.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("roof_sample: rgb is (h, w, 3)"));
    }
    let p = params.as_array();
    if p.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "roof_sample: params are (erode_px, green_gap, shadow_max, min_keep)"));
    }
    let offs: Vec<usize> = off.as_array().iter().map(|&v| v as usize).collect();
    if offs.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("roof_sample: off is (n + 1)"));
    }
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let src: Vec<u8> = img.iter().copied().collect();
    let pv = pts.as_array();
    let ptsv: Vec<(f64, f64)> = (0..pv.shape()[0]).map(|i| (pv[[i, 0]], pv[[i, 1]])).collect();
    let (erode, gap, shadow, min_keep) = (p[0], p[1], p[2], p[3] as usize);
    // the work: the rings' bbox areas (a tile's few dozen small roofs stay under RELEASE_MIN and keep the GIL)
    let work: usize = (0..offs.len() - 1).map(|b| {
        let (a, z) = (offs[b], offs[b + 1].min(ptsv.len()));
        if z.saturating_sub(a) < 3 {
            return 0;
        }
        let ring = &ptsv[a..z];
        let (u0, u1) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |ac, p| (ac.0.min(p.0), ac.1.max(p.0)));
        let (v0, v1) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |ac, p| (ac.0.min(p.1), ac.1.max(p.1)));
        let (du, dv) = (u1 - u0, v1 - v0);
        if du.is_finite() && dv.is_finite() { (du * dv) as usize } else { 0 }
    }).sum();
    let rows = if work >= crate::RELEASE_MIN {
        py.allow_threads(move || sample_rows(&src, h, w, &offs, &ptsv, erode, gap, shadow, min_keep))
    } else {
        sample_rows(&src, h, w, &offs, &ptsv, erode, gap, shadow, min_keep)
    };
    Ok(numpy::ndarray::Array2::from_shape_vec((rows.len(), 5), rows.concat())
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(h: usize, w: usize, f: impl Fn(usize, usize) -> [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity(h * w * 3);
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        v
    }

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<(f64, f64)> {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    #[test]
    fn ring_and_distance() {
        let r = rect(2.0, 2.0, 8.0, 6.0);
        assert!(inside_ring(&r, 5.0, 4.0));
        assert!(!inside_ring(&r, 1.0, 4.0));
        assert!(!inside_ring(&r, 9.0, 4.0));
        assert_eq!(dist2_to_ring(&r, 3.0, 4.0), 1.0);      // 1 px in from the x0 edge
    }

    #[test]
    fn median_even_odd() {
        assert_eq!(median(&mut vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut vec![4.0, 1.0, 2.0, 3.0]), 2.5);
    }

    #[test]
    fn red_roof_on_a_green_yard() {
        // 16x16: a (200, 60, 50) roof over rows 4..=11, cols 4..=11, green all round it; the ring covers it.
        let src = img(16, 16, |x, y| {
            if (4..12).contains(&x) && (4..12).contains(&y) { [200, 60, 50] } else { [70, 150, 60] }
        });
        let pts = rect(4.0, 4.0, 12.0, 12.0);
        let rows = sample_rows(&src, 16, 16, &[0, 4], &pts, 1.0, 40.0, 45.0, 8);
        assert_eq!(rows.len(), 1);
        let [r, g, b, cov, spread] = rows[0];
        assert!((r - 200.0).abs() < 1e-9 && (g - 60.0).abs() < 1e-9 && (b - 50.0).abs() < 1e-9, "{rows:?}");
        // the ring's 64 inside texels; the 1 px erosion keeps the inner 6x6, all red
        assert!((cov - 36.0 / 64.0).abs() < 1e-9, "coverage {cov}");
        assert!(spread.abs() < 1e-9);
    }

    #[test]
    fn canopy_and_shadow_are_dropped() {
        // half the roof under a tree (green), the other half in shadow (dark): the sunlit, unshadowed quarter wins
        let src = img(16, 16, |x, y| {
            if !(4..12).contains(&x) || !(4..12).contains(&y) {
                [70, 150, 60]
            } else if x < 8 && y < 8 {
                [90, 170, 80]              // tree top over the roof
            } else if x >= 10 || y >= 10 {
                [20, 18, 22]               // hard shadow
            } else {
                [180, 170, 160]
            }
        });
        let pts = rect(4.0, 4.0, 12.0, 12.0);
        let rows = sample_rows(&src, 16, 16, &[0, 4], &pts, 0.0, 40.0, 45.0, 4);
        let [r, g, b, _cov, spread] = rows[0];
        assert!((r - 180.0).abs() < 1e-9 && (g - 170.0).abs() < 1e-9 && (b - 160.0).abs() < 1e-9, "{rows:?}");
        assert!(spread.abs() < 1e-9);
    }

    #[test]
    fn too_few_kept_or_nothing_inside_is_nan() {
        let green = img(16, 16, |_, _| [70, 150, 60]);
        let rows = sample_rows(&green, 16, 16, &[0, 4], &rect(4.0, 4.0, 12.0, 12.0), 0.0, 40.0, 45.0, 8);
        assert!(rows[0].iter().all(|v| v.is_nan()), "all canopy: no roof colour");
        // a ring wholly outside the raster
        let rows = sample_rows(&green, 16, 16, &[0, 4], &rect(40.0, 40.0, 60.0, 60.0), 0.0, 40.0, 45.0, 4);
        assert!(rows[0].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn half_and_half_is_a_spread() {
        // the median absolute deviation misses this: the majority alone fills the middle quantiles. The
        // percentile spread scores it the full distance, which is what the caller's MAX_SPREAD gate needs.
        let src = img(16, 16, |x, y| {
            if !(4..12).contains(&x) || !(4..12).contains(&y) {
                [70, 150, 60]
            } else if y < 8 {
                [200, 60, 50]
            } else {
                [40, 90, 200]
            }
        });
        let rows = sample_rows(&src, 16, 16, &[0, 4], &rect(4.0, 4.0, 12.0, 12.0), 0.0, 40.0, 45.0, 8);
        assert!(rows[0][4] > 100.0, "two materials must not read as one roof: {rows:?}");
    }

    #[test]
    fn several_rings_in_one_call() {
        let src = img(32, 32, |x, y| {
            if (2..10).contains(&x) && (2..10).contains(&y) { [200, 60, 50] }
            else if (18..26).contains(&x) && (18..26).contains(&y) { [40, 90, 200] }
            else { [70, 150, 60] }
        });
        let pts: Vec<(f64, f64)> = [rect(2.0, 2.0, 10.0, 10.0), rect(18.0, 18.0, 26.0, 26.0)].concat();
        let rows = sample_rows(&src, 32, 32, &[0, 4, 8], &pts, 1.0, 40.0, 45.0, 8);
        assert!((rows[0][0] - 200.0).abs() < 1e-9 && (rows[1][2] - 200.0).abs() < 1e-9, "{rows:?}");
        assert!(!rows[0][0].is_nan() && !rows[1][0].is_nan());
    }
}
