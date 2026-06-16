"""Quick sanity check — single image feature extraction."""
import sys
sys.path.insert(0, ".")
from extract_person_features import extract_features_from_image, _make_detector
from pathlib import Path

IMG = Path(r"C:\Users\k1s1t\Desktop\Project\Styleflow\Skin\release\RGB\RGB\train\primavera\bright\10054.jpg")

det = _make_detector()
feats = extract_features_from_image(IMG, det)
det.close()

if feats is None:
    print("FAILED: no features extracted")
    sys.exit(1)

valid = {k: v for k, v in feats.items() if k.endswith("_valid")}
print("Valid regions:", valid)

keys = [
    "skin_mean_L", "skin_mean_a", "skin_mean_b", "skin_mean_C",
    "skin_warm_score", "deltaE_skin_hair", "face_contrast_L",
    "clear_muted_score", "light_dark_score",
]
for k in keys:
    val = feats.get(k, float("nan"))
    try:
        print(f"  {k:<25} = {val:.3f}")
    except Exception:
        print(f"  {k:<25} = {val}")

print(f"\nTotal feature keys: {len(feats)}")
