# optimizer.py
import numpy as np
import cvxpy as cp

def compute_dynamic_prices(predicted_loads, base_price=6.0, price_bounds=(3.0, 10.0)):
    """
    predicted_loads: array-like of forecasted Total_MW for horizon (e.g., next 12 hours)
    returns: list of dynamic prices (same length)
    """
    predicted_loads = np.array(predicted_loads).astype(float)
    n = len(predicted_loads)
    # decision variables: prices for each hour
    p = cp.Variable(n)

    # We want to incentivize charging in low-load hours: set target price inversely proportional to load
    # Normalize load to [0,1]
    L = (predicted_loads - predicted_loads.min()) / (predicted_loads.max() - predicted_loads.min() + 1e-6)
    target = (1 - L) * base_price  # lower load -> lower target price

    # objective: keep p close to target while smoothing price changes
    lam = 1.0  # smoothing weight
    obj = cp.Minimize(cp.sum_squares(p - target) + lam * cp.sum_squares(cp.diff(p)))
    constraints = [p >= price_bounds[0], p <= price_bounds[1]]
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)


    if p.value is None:
        # fallback simple rule
        return (target.clip(price_bounds[0], price_bounds[1])).tolist()
    return p.value.tolist()

# quick test
if __name__ == "__main__":
    import numpy as np
    pred = np.linspace(30000, 35000, 12) + np.random.randn(12)*200
    print(compute_dynamic_prices(pred))
