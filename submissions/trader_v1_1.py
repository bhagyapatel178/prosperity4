from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

# =========================
# Constants
# =========================

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMITS = {
    PEPPER: 80,
    OSMIUM: 80,
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

# v1.1 small Pepper tweaks
PEPPER_EARLY_TS = 30000
PEPPER_EARLY_TARGET = 24          # modest early long target
PEPPER_EARLY_EXTRA_BUY = 1.0      # allow slightly earlier buying
PEPPER_EARLY_BID_OFFSET = 2.0     # tighter passive bid early
PEPPER_EARLY_BID_SIZE = 16        # larger passive bid early
PEPPER_EARLY_ASK_SIZE = 4         # smaller passive ask early

# Osmium: strong short-horizon mean reversion
OSMIUM_MEANREV_ALPHA = 0.90
OSMIUM_INV_PENALTY = 0.10
OSMIUM_TAKE_EDGE = 1.0
OSMIUM_PASSIVE_OFFSET = 2.0
OSMIUM_PASSIVE_SIZE = 10


class Trader:
    def __init__(self):
        pass

    # -------------------------
    # State helpers
    # -------------------------
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

    # -------------------------
    # Market helpers
    # -------------------------
    def best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, order_depth: OrderDepth, fallback: Optional[float] = None) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return fallback

    # -------------------------
    # Order placement helpers
    # -------------------------
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

    # -------------------------
    # Pepper strategy
    # -------------------------
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

        # Estimate anchor: mid - slope * timestamp
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

        is_early = state.timestamp < PEPPER_EARLY_TS
        under_early_target = position < PEPPER_EARLY_TARGET

        # Early in the run, be slightly more willing to buy if we still have not built
        # a modest long position.
        pepper_take_threshold = fair - PEPPER_TAKE_EDGE
        if is_early and under_early_target:
            pepper_take_threshold = fair - PEPPER_TAKE_EDGE + PEPPER_EARLY_EXTRA_BUY

        # Aggressively take clearly good asks
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_qty = -order_depth.sell_orders[ask_price]
            if ask_price <= pepper_take_threshold:
                buy_used = self.try_buy(
                    orders, product, ask_price, ask_qty, buy_capacity, buy_used
                )

        # Aggressively hit clearly rich bids
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_qty = order_depth.buy_orders[bid_price]
            if bid_price >= fair + PEPPER_TAKE_EDGE:
                sell_used = self.try_sell(
                    orders, product, bid_price, bid_qty, sell_capacity, sell_used
                )

        # Passive quoting if both sides exist
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            passive_offset = PEPPER_PASSIVE_OFFSET
            passive_bid_size = PEPPER_PASSIVE_BID_SIZE
            passive_ask_size = PEPPER_PASSIVE_ASK_SIZE

            if is_early and under_early_target:
                passive_offset = PEPPER_EARLY_BID_OFFSET
                passive_bid_size = PEPPER_EARLY_BID_SIZE
                passive_ask_size = PEPPER_EARLY_ASK_SIZE

            passive_bid = min(best_bid + 1, math.floor(fair - passive_offset))
            passive_ask = max(best_ask - 1, math.ceil(fair + PEPPER_PASSIVE_OFFSET))

            # Make sure passive quotes do not cross
            if passive_bid < best_ask:
                buy_used = self.try_buy(
                    orders,
                    product,
                    int(passive_bid),
                    passive_bid_size,
                    buy_capacity,
                    buy_used,
                )

            if passive_ask > best_bid:
                sell_used = self.try_sell(
                    orders,
                    product,
                    int(passive_ask),
                    passive_ask_size,
                    sell_capacity,
                    sell_used,
                )

        cache["last_mid"][product] = mid
        return orders

    # -------------------------
    # Osmium strategy
    # -------------------------
    def trade_osmium(
        self,
        state: TradingState,
        order_depth: OrderDepth,
        cache: dict,
    ) -> List[Order]:
        orders: List[Order] = []
        product = OSMIUM
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

        # Mean-reversion prediction
        if last_mid is not None:
            last_move = mid - last_mid
            predicted_next_mid = mid - OSMIUM_MEANREV_ALPHA * last_move
        else:
            predicted_next_mid = mid

        fair = predicted_next_mid - OSMIUM_INV_PENALTY * position

        # Aggressive taking when top-of-book is mispriced vs predicted fair
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_qty = -order_depth.sell_orders[ask_price]
            if ask_price <= fair - OSMIUM_TAKE_EDGE:
                buy_used = self.try_buy(
                    orders, product, ask_price, ask_qty, buy_capacity, buy_used
                )

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_qty = order_depth.buy_orders[bid_price]
            if bid_price >= fair + OSMIUM_TAKE_EDGE:
                sell_used = self.try_sell(
                    orders, product, bid_price, bid_qty, sell_capacity, sell_used
                )

        # Passive market making around fair
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            passive_bid = min(best_bid + 1, math.floor(fair - OSMIUM_PASSIVE_OFFSET))
            passive_ask = max(best_ask - 1, math.ceil(fair + OSMIUM_PASSIVE_OFFSET))

            if passive_bid < best_ask:
                buy_used = self.try_buy(
                    orders,
                    product,
                    int(passive_bid),
                    OSMIUM_PASSIVE_SIZE,
                    buy_capacity,
                    buy_used,
                )

            if passive_ask > best_bid:
                sell_used = self.try_sell(
                    orders,
                    product,
                    int(passive_ask),
                    OSMIUM_PASSIVE_SIZE,
                    sell_capacity,
                    sell_used,
                )

        cache["last_mid"][product] = mid
        return orders

    # -------------------------
    # Main entry point
    # -------------------------
    def run(self, state: TradingState):
        cache = self.load_state(state.traderData)

        result: Dict[str, List[Order]] = {
            PEPPER: [],
            OSMIUM: [],
        }

        if PEPPER in state.order_depths:
            result[PEPPER] = self.trade_pepper(
                state, state.order_depths[PEPPER], cache
            )

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self.trade_osmium(
                state, state.order_depths[OSMIUM], cache
            )

        trader_data = self.dump_state(cache)
        conversions = 0
        return result, conversions, trader_data