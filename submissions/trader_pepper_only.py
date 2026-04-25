from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

PEPPER = "INTARIAN_PEPPER_ROOT"

POSITION_LIMITS = {
    PEPPER: 80,
}

# Pepper Root: near-deterministic upward trend
PEPPER_SLOPE = 0.001
PEPPER_ANCHOR_ALPHA = 0.20
PEPPER_TREND_BIAS = 4.0
PEPPER_INV_PENALTY = 0.05
PEPPER_TAKE_EDGE = 1.0
PEPPER_PASSIVE_OFFSET = 3.0
PEPPER_PASSIVE_BID_SIZE = 12
PEPPER_PASSIVE_ASK_SIZE = 6


class Trader:
    def load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {
                "last_mid": {},
                "pepper_anchor": None,
            }
        try:
            data = json.loads(trader_data)
            if "last_mid" not in data:
                data["last_mid"] = {}
            if "pepper_anchor" not in data:
                data["pepper_anchor"] = None
            return data
        except Exception:
            return {
                "last_mid": {},
                "pepper_anchor": None,
            }

    def dump_state(self, data: dict) -> str:
        return json.dumps(data, separators=(",", ":"))

    def best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, order_depth: OrderDepth, fallback: Optional[float] = None) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return fallback

    def try_buy(
        self,
        orders: List[Order],
        product: str,
        price: int,
        qty: int,
        buy_capacity: int,
        buy_used: int,
    ) -> int:
        qty = min(qty, max(0, buy_capacity - buy_used))
        if qty > 0:
            orders.append(Order(product, price, qty))
            buy_used += qty
        return buy_used

    def try_sell(
        self,
        orders: List[Order],
        product: str,
        price: int,
        qty: int,
        sell_capacity: int,
        sell_used: int,
    ) -> int:
        qty = min(qty, max(0, sell_capacity - sell_used))
        if qty > 0:
            orders.append(Order(product, price, -qty))
            sell_used += qty
        return sell_used

    def trade_pepper(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        cache: dict,
    ) -> List[Order]:
        orders: List[Order] = []
        product = PEPPER
        position = state.position.get(product, 0)
        limit = POSITION_LIMITS[product]

        buy_capacity = limit - position
        sell_capacity = limit + position
        buy_used = 0
        sell_used = 0

        best_bid, best_ask = self.best_bid_ask(order_depth)
        last_mid = cache["last_mid"].get(product)
        mid = self.mid_price(order_depth, fallback=last_mid)

        if mid is None:
            return orders

        observed_anchor = mid - PEPPER_SLOPE * state.timestamp
        prev_anchor = cache.get("pepper_anchor")
        if prev_anchor is None:
            pepper_anchor = observed_anchor
        else:
            pepper_anchor = (1 - PEPPER_ANCHOR_ALPHA) * prev_anchor + PEPPER_ANCHOR_ALPHA * observed_anchor
        cache["pepper_anchor"] = pepper_anchor

        fair = pepper_anchor + PEPPER_SLOPE * state.timestamp
        fair += PEPPER_TREND_BIAS
        fair -= PEPPER_INV_PENALTY * position

        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_qty = -order_depth.sell_orders[ask_price]
            if ask_price <= fair - PEPPER_TAKE_EDGE:
                buy_used = self.try_buy(
                    orders, product, ask_price, ask_qty, buy_capacity, buy_used
                )

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_qty = order_depth.buy_orders[bid_price]
            if bid_price >= fair + PEPPER_TAKE_EDGE:
                sell_used = self.try_sell(
                    orders, product, bid_price, bid_qty, sell_capacity, sell_used
                )

        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            passive_bid = min(best_bid + 1, math.floor(fair - PEPPER_PASSIVE_OFFSET))
            passive_ask = max(best_ask - 1, math.ceil(fair + PEPPER_PASSIVE_OFFSET))

            if passive_bid < best_ask:
                buy_used = self.try_buy(
                    orders,
                    product,
                    int(passive_bid),
                    PEPPER_PASSIVE_BID_SIZE,
                    buy_capacity,
                    buy_used,
                )

            if passive_ask > best_bid:
                sell_used = self.try_sell(
                    orders,
                    product,
                    int(passive_ask),
                    PEPPER_PASSIVE_ASK_SIZE,
                    sell_capacity,
                    sell_used,
                )

        cache["last_mid"][product] = mid
        return orders

    def run(self, state: TradingState):
        cache = self.load_state(state.traderData)

        result: Dict[str, List[Order]] = {
            product: [] for product in state.order_depths
        }

        if PEPPER in state.order_depths:
            result[PEPPER] = self.trade_pepper(
                state, state.order_depths[PEPPER], cache
            )

        trader_data = self.dump_state(cache)
        conversions = 0
        return result, conversions, trader_data