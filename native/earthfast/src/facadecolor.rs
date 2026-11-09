//! Facade colours from street-level photos (server/streetmly.py, PLAN.md M2): a building wall seen in a
//! Mapillary thumbnail is projected with the photo's pinhole model (level camera, so only pixels near the
//! image centre are trusted), and the wall's pixels are clustered in Lab into up to three modes: the largest
//! one that is not window-dark or ground-dark is the painted wall. server/streetmly.py takes the median of
//! the per-photo rows per building orientation.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::prelude::*;

/// sRGB 0..255 -> CIE Lab (D65): pure math, no tables.
fn rgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let gam = |c: f64| {
        let c = c / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let (rl, gl, bl) = (gam(r), gam(g), gam(b));
    // linear sRGB -> XYZ (D65), then the Lab transfer with the D65 white point
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
    let f = |t: f64| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
    let (fx, fy, fz) = (f(x / 0.95047), f(y), f(z / 1.08883));
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// The median of v (sorted in place).
fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 { v[m] } else { 0.5 * (v[m - 1] + v[m]) }
}

const NB: usize = 20;    // Lab bins per axis: L 0..100 by 5, a and b -110..110 by 11

/// The wall rows (see facade_look): pure, so it runs with and without the interpreter. A row is
/// [nan; 5] when the wall is not usable in this photo; else [r, g, b, share, kept].
fn wall_rows(src: &[u8], h: usize, w: usize, cam: [f64; 3], yaw: f64, intr: [f64; 4],
             walls: &[[f64; 12]], p: &[f64]) -> Vec<[f64; 5]> {
    let (min_h, max_off, min_dist, l_dark, min_share, min_kept, l_tol, ab_tol) =
        (p[0], p[1], p[2], p[3], p[4], p[5] as usize, p[6], p[7]);
    let (fx, fy, cx, cy) = (intr[0], intr[1], intr[2], intr[3]);
    let (sy, cyw) = (yaw.sin(), yaw.cos());     // yaw: radians from north, clockwise (compass)
    // level camera: right = (cos, -sin, 0), forward = (sin, cos, 0), up = (0, 0, 1)
    let mut rows = vec![[f64::NAN; 5]; walls.len()];
    for (wi, wt) in walls.iter().enumerate() {
        let (mut u0, mut u1, mut v0, mut v1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        let (mut dx, mut dy) = (0.0, 0.0);
        let mut ok = true;
        for c in 0..4 {
            let (px, py, pz) = (wt[c * 3], wt[c * 3 + 1], wt[c * 3 + 2]);
            let (ex, ey, ez) = (px - cam[0], py - cam[1], pz - cam[2]);
            dx += ex / 4.0;
            dy += ey / 4.0;
            let (xx, yy, zz) = (cyw * ex - sy * ey, ez, sy * ex + cyw * ey);
            if zz <= 0.5 {
                ok = false;
                break;
            }
            let (uu, vv) = (cx + fx * xx / zz, cy - fy * yy / zz);
            if !uu.is_finite() || !vv.is_finite() {
                ok = false;
                break;
            }
            (u0, u1) = (u0.min(uu), u1.max(uu));
            (v0, v1) = (v0.min(vv), v1.max(vv));
        }
        if !ok || (dx * dx + dy * dy) < min_dist * min_dist {
            continue;    // (a wall behind the camera or too close: too oblique to trust its pixels)
        }
        let (x0, x1) = (u0.floor().max(0.0) as usize, u1.ceil().min(w as f64 - 1.0) as usize);
        let (y0, y1) = (v0.floor().max(0.0) as usize, v1.ceil().min(h as f64 - 1.0) as usize);
        if x1 < x0 || y1 < y0 || ((y1 - y0) as f64) < min_h {
            continue;    // (out of the photo or too small: a few pixels are noise, not a facade)
        }
        if ((x0 + x1) as f64 / 2.0 - cx).abs() > max_off * w as f64
            || ((y0 + y1) as f64 / 2.0 - cy).abs() > max_off * h as f64
        {
            continue;    // (far off centre: the unknown camera pitch bends the projection there)
        }
        // pass 1: the Lab histogram (very dark pixels excluded up front: grime and night don't vote)
        let mut hist = vec![0u32; NB * NB * NB];
        let (mut ls, mut as_, mut bs) =
            (vec![0.0f64; NB * NB * NB], vec![0.0f64; NB * NB * NB], vec![0.0f64; NB * NB * NB]);
        let mut total = 0usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let q = (y * w + x) * 3;
                let (r, g, b) = (src[q] as f64, src[q + 1] as f64, src[q + 2] as f64);
                if r.max(g).max(b) < 40.0 {
                    continue;
                }
                let (l, aa, bb) = rgb_to_lab(r, g, b);
                let il = ((l / 5.0) as isize).clamp(0, NB as isize - 1) as usize;
                let ia = (((aa + 110.0) / 11.0) as isize).clamp(0, NB as isize - 1) as usize;
                let ib = (((bb + 110.0) / 11.0) as isize).clamp(0, NB as isize - 1) as usize;
                let k = (il * NB + ia) * NB + ib;
                hist[k] += 1;
                ls[k] += l;
                as_[k] += aa;
                bs[k] += bb;
                total += 1;
            }
        }
        if total < min_kept * 2 {
            continue;
        }
        // up to three modes, largest first: the 3x3x3 neighbourhood is the mode, its 3x3x3 mean its centre
        let mut picked: Vec<(f64, f64, f64, u32)> = Vec::new();
        for _ in 0..3 {
            let mut best = usize::MAX;
            for k in 0..hist.len() {
                if hist[k] > 0 && (best == usize::MAX || hist[k] > hist[best]) {
                    best = k;
                }
            }
            if best == usize::MAX {
                break;
            }
            let (bl, ba, bb) = ((best / (NB * NB)) as i64, ((best / NB) % NB) as i64, (best % NB) as i64);
            let (mut n, (mut sl, mut sa, mut sb)) = (0u32, (0.0, 0.0, 0.0));
            for dl in -1..=1 {
                for da in -1..=1 {
                    for db in -1..=1 {
                        let (l, a, b) = (bl + dl, ba + da, bb + db);
                        if l < 0 || a < 0 || b < 0 || l >= NB as i64 || a >= NB as i64 || b >= NB as i64 {
                            continue;
                        }
                        let k = ((l as usize) * NB + a as usize) * NB + b as usize;
                        n += hist[k];
                        sl += ls[k];
                        sa += as_[k];
                        sb += bs[k];
                    }
                }
            }
            // suppress the neighbourhood: the next pick is another mode, not this one's shoulder
            for dl in -1..=1 {
                for da in -1..=1 {
                    for db in -1..=1 {
                        let (l, a, b) = (bl + dl, ba + da, bb + db);
                        if l < 0 || a < 0 || b < 0 || l >= NB as i64 || a >= NB as i64 || b >= NB as i64 {
                            continue;
                        }
                        hist[((l as usize) * NB + a as usize) * NB + b as usize] = 0;
                    }
                }
            }
            if n < min_kept as u32 {
                continue;
            }
            let (ml, ma, mb) = (sl / n as f64, sa / n as f64, sb / n as f64);
            if ml < l_dark {
                continue;    // (window-dark / ground-dark: glass and shadow, not the painted wall)
            }
            picked.push((ml, ma, mb, n));
            break;           // (v1: the largest lit mode is the facade; the two smaller ones are windows and trim)
        }
        let Some(&(ml, ma, mb, _)) = picked.first() else { continue };
        // pass 2: the pixels that wear the mode's colour, and their median RGB
        let (mut kr, mut kg, mut kb) = (Vec::new(), Vec::new(), Vec::new());
        for y in y0..=y1 {
            for x in x0..=x1 {
                let q = (y * w + x) * 3;
                let (r, g, b) = (src[q] as f64, src[q + 1] as f64, src[q + 2] as f64);
                if r.max(g).max(b) < 40.0 {
                    continue;
                }
                let (l, aa, bb) = rgb_to_lab(r, g, b);
                if (l - ml).abs() <= l_tol && (aa - ma).abs() <= ab_tol && (bb - mb).abs() <= ab_tol {
                    kr.push(r);
                    kg.push(g);
                    kb.push(b);
                }
            }
        }
        if kr.len() < min_kept || kr.len() * 2 < total {
            continue;    // (too few, or a minority of the wall: the look stays the predicted one)
        }
        let share = kr.len() as f64 / total as f64;
        if share < min_share {
            continue;
        }
        rows[wi] = [median(&mut kr), median(&mut kg), median(&mut kb), share, kr.len() as f64];
    }
    rows
}

/// facade_look(rgb, cam, yaw_rad, intr, walls, params) -> (W, 5) rows: per-wall median r, g, b (0..255), share
/// (mode pixels / the wall's lit pixels) and kept count; a nan row where the wall is unusable or unsure.
/// rgb: (h, w, 3) the photo's thumbnail (row 0 top); cam: (3) the camera position in the scene's ENU metres;
/// yaw_rad: the photo's compass heading (radians from north, clockwise); intr: (fx, fy, cx, cy) pixels;
/// walls: (W, 4, 3) the wall quads' corners in the same ENU metres (two ground, two eave);
/// params: (min_h_px, max_off_frac, min_dist_m, l_dark, min_share, min_kept, l_tol, ab_tol).
#[pyfunction]
#[pyo3(signature = (rgb, cam, yaw_rad, intr, walls, params))]
pub fn facade_look<'py>(py: Python<'py>, rgb: PyReadonlyArray3<'py, u8>, cam: PyReadonlyArray1<'py, f64>,
                        yaw_rad: f64, intr: PyReadonlyArray1<'py, f64>, walls: PyReadonlyArray3<'py, f64>,
                        params: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let img = rgb.as_array();
    if img.ndim() != 3 || img.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("facade_look: rgb is (h, w, 3)"));
    }
    let cv = cam.as_array();
    if cv.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("facade_look: cam is (3)"));
    }
    let iv = intr.as_array();
    if iv.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err("facade_look: intr is (fx, fy, cx, cy)"));
    }
    let pv = params.as_array();
    if pv.len() != 8 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "facade_look: params are (min_h_px, max_off_frac, min_dist_m, l_dark, min_share, min_kept, l_tol, ab_tol)"));
    }
    let wv = walls.as_array();
    if wv.shape().len() != 3 || wv.shape()[1] != 4 || wv.shape()[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("facade_look: walls is (W, 4, 3)"));
    }
    let (h, w) = (img.shape()[0], img.shape()[1]);
    let src: Vec<u8> = img.iter().copied().collect();
    let wt: Vec<[f64; 12]> = (0..wv.shape()[0])
        .map(|i| (0..4).flat_map(|c| [wv[[i, c, 0]], wv[[i, c, 1]], wv[[i, c, 2]]]).collect::<Vec<_>>().try_into().unwrap())
        .collect();
    let camv = [cv[0], cv[1], cv[2]];
    let intrv = [iv[0], iv[1], iv[2], iv[3]];
    let par: Vec<f64> = pv.iter().copied().collect();
    let work = wt.len() * h * w / 4;
    let rows = if work >= crate::RELEASE_MIN {
        py.allow_threads(move || wall_rows(&src, h, w, camv, yaw_rad, intrv, &wt, &par))
    } else {
        wall_rows(&src, h, w, camv, yaw_rad, intrv, &wt, &par)
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

    // the test rig: camera at the origin 2 m up looking north (yaw 0), fx = fy = 400, centre (200, 150)
    const CAM: [f64; 3] = [0.0, 0.0, 2.0];
    const INTR: [f64; 4] = [400.0, 400.0, 200.0, 150.0];
    const P: [f64; 8] = [40.0, 0.3, 8.0, 25.0, 0.3, 40.0, 4.0, 8.0];

    /// A wall of x0..x1 at distance d, ground 0 to eave e: its four corners (ground a, ground b, eave b, eave a).
    fn wall(x0: f64, x1: f64, d: f64, e: f64) -> [f64; 12] {
        [x0, d, 0.0, x1, d, 0.0, x1, d, e, x0, d, e]
    }

    #[test]
    fn lab_of_known_colours() {
        let (l, ..) = rgb_to_lab(255.0, 255.0, 255.0);
        assert!((l - 100.0).abs() < 0.5, "white L {l}");
        let (l, _, _) = rgb_to_lab(0.0, 0.0, 0.0);
        assert!(l.abs() < 1e-9, "black L {l}");
    }

    #[test]
    fn a_painted_wall_reads_its_colour() {
        // 400x300: (6, 140, 170) stucco everywhere but a dark window band inside the wall's own pixels
        let src = img(300, 400, |x, _| if (120..170).contains(&x) { [22, 24, 30] } else { [6, 140, 170] });
        // a wall 20 m north, x -4..8, 0..8 m up: ground row at v = 150 + 400*2/20 = 190, eave at v = 150 - 400*6/20 = 30
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-4.0, 8.0, 20.0, 8.0)], &P);
        let [r, g, b, share, kept] = rows[0];
        assert!((r - 6.0).abs() < 1e-9 && (g - 140.0).abs() < 1e-9 && (b - 170.0).abs() < 1e-9, "{rows:?}");
        assert!(share > 0.5 && kept > 40.0, "windows excluded, wall wins: {rows:?}");
    }

    #[test]
    fn a_wall_behind_or_too_close_is_nothing() {
        let src = img(300, 400, |_, _| [120, 120, 120]);
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-4.0, 4.0, -20.0, 8.0)], &P);
        assert!(rows[0].iter().all(|v| v.is_nan()), "behind the camera");
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-4.0, 4.0, 4.0, 8.0)], &P);
        assert!(rows[0].iter().all(|v| v.is_nan()), "4 m away is under the 8 m floor");
    }

    #[test]
    fn a_wall_off_the_centre_is_nothing() {
        // looking north, a wall far to the east: its pixels sit at the image edge, where the unknown pitch lies
        let src = img(300, 400, |_, _| [120, 120, 120]);
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(30.0, 36.0, 20.0, 8.0)], &P);
        assert!(rows[0].iter().all(|v| v.is_nan()), "off centre and mostly out of frame");
    }

    #[test]
    fn a_tiny_wall_is_nothing() {
        let src = img(300, 400, |_, _| [120, 120, 120]);
        // 2 m wide at 60 m: 20 px tall (the eave sits at 400*3/60 = 20 px above the camera line)
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-1.0, 1.0, 60.0, 3.0)], &P);
        assert!(rows[0].iter().all(|v| v.is_nan()), "20 px is under the 40 px floor");
    }

    #[test]
    fn a_dark_wall_is_nothing() {
        let src = img(300, 400, |_, _| [10, 10, 12]);    // night, or a wall of pure shadow
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-4.0, 8.0, 20.0, 8.0)], &P);
        assert!(rows[0].iter().all(|v| v.is_nan()), "no lit pixels vote");
    }

    #[test]
    fn a_minority_colour_stays_predicted() {
        // the wall is mostly (90, 90, 90) with a narrow (220, 210, 60) band across its pixels: the band is
        // trim, not the facade; the largest mode wins, so the row must read the grey, not the band
        let src = img(300, 400, |x, _| if (150..190).contains(&x) { [220, 210, 60] } else { [90, 90, 90] });
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[wall(-4.0, 8.0, 20.0, 8.0)], &P);
        assert!((rows[0][0] - 90.0).abs() < 1e-9 && (rows[0][2] - 90.0).abs() < 1e-9, "{rows:?}");
    }

    #[test]
    fn two_walls_one_photo() {
        let src = img(300, 400, |x, _| if x < 200 { [200, 60, 50] } else { [60, 60, 200] });
        // two gables of the same north-facing frame, one seen left of centre, one right
        let w1 = wall(-8.0, -2.0, 20.0, 8.0);    // u 40..160, all over the red half
        let w2 = wall(2.0, 8.0, 20.0, 8.0);      // u 240..360, all over the blue half
        let rows = wall_rows(&src, 300, 400, CAM, 0.0, INTR, &[w1, w2], &P);
        assert!((rows[0][0] - 200.0).abs() < 1e-9, "left wall: {:?}", rows[0]);
        assert!((rows[1][2] - 200.0).abs() < 1e-9, "right wall: {:?}", rows[1]);
    }
}
