import numpy as np
from typing import Literal, Tuple, Optional
from scipy.ndimage import gaussian_filter
from numpy.fft import rfftn, irfftn

# -------------------------------
# Thickness-field generators
# -------------------------------

def _field_gaussian(shape: Tuple[int,int], sigma: float, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 1, size=shape).astype(np.float32)
    t = gaussian_filter(t, sigma=sigma, mode="reflect")
    t -= t.min()
    t /= (t.max() - t.min() + 1e-8)
    return t

def _field_powerlaw(shape: Tuple[int,int], beta: float, seed: Optional[int]) -> np.ndarray:
    """
    Make a 1/f^beta random field via FFT synthesis (beta≈1..3 looks good).
    """
    H, W = shape
    rng = np.random.default_rng(seed)
    # white noise in Fourier domain with random phases
    white = rng.normal(0, 1, size=(H, W)).astype(np.float32)
    # radial frequency grid
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    k[0, 0] = 1.0  # avoid divide by zero at DC
    amp = 1.0 / (k ** (beta / 2.0))  # amplitude shaping (power ∝ 1/k^beta)
    F = rfftn(white)
    F *= amp
    t = irfftn(F, s=(H, W))
    t = t.astype(np.float32)
    t -= t.min()
    t /= (t.max() - t.min() + 1e-8)
    return t

def _make_thickness(
        shape: Tuple[int,int],
        method: Literal["gaussian","powerlaw"]="powerlaw",
        smoothness: float = 64.0,     # used when method="gaussian"
        beta: float = 2.0,            # used when method="powerlaw"
        scan_bias: float = 0.0,       # 0..1, linear growth along y
        poly_xy: Tuple[float,float,float,float]=(0,0,0,0),  # a x + b y + c x^2 + d y^2
        seed: Optional[int]=None
) -> np.ndarray:
    H, W = shape
    if method == "gaussian":
        t = _field_gaussian(shape, sigma=float(smoothness), seed=seed)
    else:
        t = _field_powerlaw(shape, beta=float(beta), seed=seed)

    # optional linear growth along slow-scan (y)
    if scan_bias > 0:
        y = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        t = (1 - scan_bias) * t + scan_bias * y

    # optional low-order polynomial drift
    a, b, c, d = poly_xy
    if any(abs(v) > 0 for v in (a, b, c, d)):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        xx = (xx - W/2) / max(W, 1)
        yy = (yy - H/2) / max(H, 1)
        poly = a*xx + b*yy + c*(xx**2) + d*(yy**2)
        # normalize poly to [0,1] and blend (lightly) with t
        poly -= poly.min()
        poly /= (poly.max() - poly.min() + 1e-8)
        t = 0.9 * t + 0.1 * poly

    # final normalization
    t -= t.min()
    t /= (t.max() - t.min() + 1e-8)
    return t

# ---------------------------------------
# STEM-aware contamination (NumPy/Scipy)
# ---------------------------------------

def add_stem_contamination(
        img: np.ndarray,                            # (H,W) or (C,H,W), float in [0,1]
        mode: Literal["HAADF","BF","ABF"]="HAADF",
        strength: float = 0.6,                      # overall film thickness scale (0..1)
        haze_ratio: float = 0.10,                   # additive diffuse background scale
        nonlinearity: Literal["exp","power"]="exp", # mass–thickness law
        k: float = 1.2,                             # coefficient in exp/power law
        gamma: float = 1.0,                         # power exponent if nonlinearity="power"
        field_method: Literal["powerlaw","gaussian"]="powerlaw",
        smoothness: float = 64.0,                   # only for gaussian field
        beta: float = 2.0,                          # only for powerlaw field
        scan_bias: float = 0.0,                     # 0..1 (growth along y)
        poly_xy: Tuple[float,float,float,float]=(0,0,0,0),
        seed: Optional[int]=None
) -> np.ndarray:
    """
    STEM-informed contamination:
      - build thickness t(x,y) in [0,1] with desired spatial statistics,
      - HAADF brightens with film; BF/ABF darken with film,
      - add weak additive haze proportional to t,
      - preserve output in [0,1].
    """
    # shape handling
    if img.ndim == 2:
        H, W = img.shape
        imgC = img[np.newaxis, ...].astype(np.float32)
    elif img.ndim == 3:
        C, H, W = img.shape
        imgC = img.astype(np.float32)
    else:
        raise ValueError("img must be (H,W) or (C,H,W)")

    # thickness field
    t = _make_thickness(
        (H, W),
        method=field_method,
        smoothness=smoothness,
        beta=beta,
        scan_bias=scan_bias,
        poly_xy=poly_xy,
        seed=seed,
    )
    t = (strength * t).astype(np.float32)

    # multiplicative mass–thickness response
    m = mode.upper()
    if nonlinearity == "exp":
        g = np.exp( (+k)*t if m=="HAADF" else (-k)*t ).astype(np.float32)
    else:
        # (1 + k t)^±gamma
        base = np.maximum(1.0 + k*t, 1e-6)
        g = np.power(base, (+gamma) if m=="HAADF" else (-gamma)).astype(np.float32)

    # additive diffuse background (haze)
    bg = (haze_ratio * t).astype(np.float32)

    # apply shared field to all channels
    out = imgC * g[None, ...] + bg[None, ...]
    out = np.clip(out, 0.0, 1.0)
    return out[0] if img.ndim == 2 else out

# -----------------------
# Simple, ergonomic API
# -----------------------

_PRESETS = {
    # quick knobs with good defaults
    "haadf_fast": dict(mode="HAADF", field_method="powerlaw", beta=2.0, strength=0.55, haze_ratio=0.08, nonlinearity="power", k=1.0, gamma=1.1, scan_bias=0.0),
    "abf_fast":   dict(mode="ABF",   field_method="powerlaw", beta=2.0, strength=0.55, haze_ratio=0.08, nonlinearity="power", k=1.0, gamma=1.1, scan_bias=0.0),
    "phys_heavy": dict(mode="HAADF", field_method="powerlaw", beta=2.4, strength=0.8,  haze_ratio=0.12, nonlinearity="exp",   k=1.4, gamma=1.0, scan_bias=0.2),
}

def add_stem_contamination_easy(
        img: np.ndarray,
        preset: Literal["haadf_fast","abf_fast","phys_heavy"]="haadf_fast",
        strength: Optional[float]=None,
        smoothness: Optional[float]=None,  # if you switch field_method="gaussian" via **overrides
        haze_ratio: Optional[float]=None,
        scan_bias: Optional[float]=None,
        seed: Optional[int]=None,
        **overrides
) -> np.ndarray:
    cfg = dict(_PRESETS[preset])
    if strength   is not None: cfg["strength"]   = float(strength)
    if smoothness is not None: cfg["smoothness"] = float(smoothness)
    if haze_ratio is not None: cfg["haze_ratio"] = float(haze_ratio)
    if scan_bias  is not None: cfg["scan_bias"]  = float(scan_bias)
    if seed       is not None: cfg["seed"]       = seed
    cfg.update(overrides)  # e.g. field_method="gaussian", nonlinearity="exp"
    return add_stem_contamination(img, **cfg)
