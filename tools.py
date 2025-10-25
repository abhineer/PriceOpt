
import pulp
from langchain.tools import tool
from data import black_friday_profile

@tool
def optimize_price_with_lp(
    item_id: str,
    cost_price: float,
    current_price: float,
    competitor_prices: list,
    target_margin_percent: float,
    stock_level: int,
    hourly_sales: list,
    price_elasticity: float
) -> str:
    """
    Optimize price using LP to maximize revenue under margin, elasticity, and stock constraints.
    """
    if not hourly_sales or len(hourly_sales) < 1:
        return f"Item: {item_id} — Invalid hourly_sales data."

    recent_sales = sum(hourly_sales[-6:])
    profile_tail = black_friday_profile[-6:].sum()
    scale_factor = recent_sales / profile_tail
    estimated_total_demand = scale_factor * black_friday_profile.sum()

    pct_changes = [-0.10, -0.05, 0.0, 0.05, 0.10]
    price_demand_revenue = []

    min_price = cost_price * (1 + target_margin_percent / 100)
    max_price = max(competitor_prices) * 1.10
    current_demand = estimated_total_demand * (1 + price_elasticity * 0)
    current_revenue = current_price * min(current_demand, stock_level)

    for pct in pct_changes:
        price = round(current_price * (1 + pct), 2)
        if price < min_price or price > max_price:
            continue
        simulated_demand = estimated_total_demand * (1 + price_elasticity * pct)
        demand_capped = min(simulated_demand, stock_level)
        revenue = price * demand_capped
        price_demand_revenue.append((price, revenue))

    if not price_demand_revenue:
        return f"Item: {item_id} — No feasible price within margin and competitor guardrails."

    prob = pulp.LpProblem("Maximize_Revenue", pulp.LpMaximize)
    price_vars = {
        i: pulp.LpVariable(f"price_option_{i}", cat="Binary")
        for i in range(len(price_demand_revenue))
    }
    prob += pulp.lpSum(price_vars[i] * price_demand_revenue[i][1] for i in price_vars)
    prob += pulp.lpSum(price_vars[i] for i in price_vars) == 1
    prob.solve()

    for i, var in price_vars.items():
        if var.value() == 1:
            selected_price, selected_revenue = price_demand_revenue[i]
            revenue_diff = selected_revenue - current_revenue
            return (
                f"Item: {item_id} — Current Price: ${current_price:.2f}, "
                f"Recommended Price: ${selected_price:.2f}, "
                f"Estimated Revenue Increase: ${revenue_diff:.2f}."
            )
    return f"Item: {item_id} — LP model failed to find a valid price."
