import joblib
from pathlib import Path

def inspect_joblib(path: str):
    obj = joblib.load(path)
    print(f"\n=== {path} ===")
    print("Type:", type(obj))

    # If it's your bundle dict
    if isinstance(obj, dict):
        print("Keys:", sorted(obj.keys()))
        for k in ["meta", "main_feature_cols", "down_feature_cols"]:
            if k in obj:
                v = obj[k]
                if k.endswith("_cols"):
                    print(f"{k}: {len(v)} cols")
                    print("  first 20:", v[:20])
                else:
                    print(f"{k}: {v}")

        for mk in ["main_model", "down_model"]:
            if mk in obj and obj[mk] is not None:
                m = obj[mk]
                print(f"\n{mk}: {type(m)}")
                if hasattr(m, "get_params"):
                    params = m.get_params()
                    # print a few common ones
                    for p in ["n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state", "n_jobs"]:
                        if p in params:
                            print(f"  {p}: {params[p]}")
                if hasattr(m, "n_features_in_"):
                    print("  n_features_in_:", m.n_features_in_)
                if hasattr(m, "feature_names_in_"):
                    print("  feature_names_in_ (first 20):", list(m.feature_names_in_)[:20])
                if hasattr(m, "classes_"):
                    print("  classes_:", m.classes_)
        return

    # If it's a plain sklearn model object
    m = obj
    if hasattr(m, "get_params"):
        print("Params:", m.get_params())
    if hasattr(m, "n_features_in_"):
        print("n_features_in_:", m.n_features_in_)
    if hasattr(m, "feature_names_in_"):
        print("feature_names_in_ (first 20):", list(m.feature_names_in_)[:20])
    if hasattr(m, "classes_"):
        print("classes_:", m.classes_)

for p in [
    "models/default_model_76.joblib",
    "models/downside_bundle.joblib",
]:
    if Path(p).exists():
        inspect_joblib(p)
    else:
        print("Missing:", p)
