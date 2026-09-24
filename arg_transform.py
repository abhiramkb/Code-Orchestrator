import math
from typing import Any, Dict

# 1. Base Python math namespace
SAFE_MATH_NAMESPACE = {
    # Constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    # Standard algebra & trig
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "exp": math.exp,
    "log": math.log,        
    "log10": math.log10,    
    "log2": math.log2,      
    "sqrt": math.sqrt,
    "pow": math.pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "radians": math.radians,
    "degrees": math.degrees,
    "gamma": math.gamma,
    "lgamma": math.lgamma,  # Natural log of absolute value of gamma
    "erf": math.erf,        # Error function
    "erfc": math.erfc,      # Complementary error function
}

# 2. Add SciPy special functions if scipy is installed
try:
    import scipy.special as sp

    SAFE_MATH_NAMESPACE.update(
        {
            # Bessel functions of the first kind (order 0, 1, n)
            "j0": sp.j0,
            "j1": sp.j1,
            "jn": sp.jn,
            # Bessel functions of the second kind (order 0, 1, n)
            "y0": sp.y0,
            "y1": sp.y1,
            "yn": sp.yn,
            # Modified Bessel functions
            "i0": sp.i0,
            "i1": sp.i1,
            "iv": sp.iv,
            "k0": sp.k0,
            "k1": sp.k1,
            "kv": sp.kv,
            # Spherical Bessel functions
            "spherical_jn": sp.spherical_jn,
            "spherical_yn": sp.spherical_yn,
            # Airy functions
            "airy": sp.airy,
            # Other common special functions
            "beta": sp.beta,
            "digamma": sp.digamma,
            "psi": sp.digamma,  # Common alias for digamma
            "zeta": sp.zeta,
            "polygamma": sp.polygamma,
            "hyp2f1": sp.hyp2f1,  # Gauss hypergeometric function
        }
    )
except ImportError:
    pass  # Fallback to standard math module if scipy isn't installed


def _apply_transform(raw_val: str, transform: str) -> str:
    """Evaluates arbitrary mathematical expressions (including special functions)
    on a raw string value.
    """
    num = float(raw_val)

    # Convert caret '^' to Python's '**' exponentiation
    expr = transform.replace("^", "**")

    eval_namespace = {
        "__builtins__": None,
        "val": num,
        "x": num,
        **SAFE_MATH_NAMESPACE,
    }

    try:
        res = eval(expr, eval_namespace)
        # SciPy special functions might return 0D numpy arrays or tuples
        if hasattr(res, "item"):
            res = res.item()
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate transform expression '{transform}' on value '{raw_val}': {e}"
        ) from e

    if isinstance(res, (float, int)):
        return f"{res:.10g}"
    return str(res)


def _coerce_numeric(val: Any) -> Any:
    """Converts a raw string value to float when possible, otherwise leaves it as-is.

    Used to let cut expressions compare numeric columns numerically without requiring
    a 'transform' on every column just to cast it.
    """
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def _evaluate_cut(expr: str, row: Dict[str, Any]) -> bool:
    """Evaluates a boolean cut expression against one row's named column values.

    Shares SAFE_MATH_NAMESPACE with _apply_transform, but binds every column in `row`
    by name instead of a single 'val'/'x', and does not force values to float, so a
    cut may compare a string-valued column directly (e.g. "status == 'VALID'").
    Numeric-looking values are coerced with _coerce_numeric first, so a cut can also
    write plain numeric comparisons (e.g. "Q < 10") without an explicit cast.
    """
    eval_namespace = {
        "__builtins__": None,
        **SAFE_MATH_NAMESPACE,
        **{k: _coerce_numeric(v) for k, v in row.items()},
    }

    try:
        result = eval(expr, eval_namespace)
    except Exception as e:
        raise ValueError(f"Failed to evaluate cut expression '{expr}' on row {row}: {e}") from e

    return bool(result)