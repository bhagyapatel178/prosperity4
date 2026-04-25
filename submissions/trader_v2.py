from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

# =========================================================
# Products / limits
# =========================================================

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMITS = {
    PEPPER: 80,
    OSMIUM: 80,
}

# =========================================================
# Pepper Root parameters
# =========================================================
# Empirically, Pepper follows a very clean upward trend:
# fair ~= anchor + 0.001 * timestamp
#
# v2 change:
# Instead of treating Pepper mainly as a market-making product,
# we explicitly try to build/hold a long position while the
# drift is unfolding, then relax later.

PEPPER_SLOPE = 0.001
PEPPER_ANCHOR_ALPHA = 0.20
PEPPER_INV_PENALTY = 0.03

# How far above current fair we are willing to buy while below target
PEPPER_BUY_CUSHION_EARLY = 3.0
PEPPER_BUY_CUSHION_MID = 2.0
PEPPER_BUY_CUSHION_LATE = 0.5

# How rich bids need to be before we happily sell
PEPPER_RICH_SELL_EDGE = 3.0

# If above target, allow lighter unloading near fair
PEPPER_UNWIND_SELL_EDGE = 0.5

# Passive quoting
PEPPER_PASSIVE_BID_OFFSET_EARLY = 1.0
PEPPER_PASSIVE_BID_OFFSET_MID = 2.0
PEPPER_PASSIVE_BID_OFFSET_LATE = 3.0

PEPPER_PASSIVE_ASK_OFFSET_EARLY = 4.0
PEPPER_PASSIVE_ASK_OFFSET_MID = 3.0
PEPPER_PASSIVE_ASK_OFFSET_LATE = 2.0

# =========================================================
# Osmium parameters
# =========================================================
# Leave the core v1 structure intact since it was already
# contributing positively in both backtest and official run.

OSMIUM_MEANREV_ALPHA = 0.90
OSMIUM_INV_PENALTY = 0.10
OSMIUM_TAKE_EDGE = 1.0
OSMIUM_PASSIVE_OFFSET = 2.0
OSMIUM_PASSIVE_SIZE = 10


class Trader:
    # -----------------------------------------------------
    # State helpers
    # -----------------------------------------------------
    def load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {
                "last_mid": {},
                "pepper_anchor": None,
                "pepper_start_anchor": None,
            }
        try:
            data = json.loads(trader_data)
            if "last_mid" not in data:
                data["last_mid"] = {}
            if "pepper_anchor" not in data:
                data["pepper_anchor"] = None
            if "pepper_start_anchor" not in data:
                data["pepper_start_anchor"] = None
            return data
        except Exception:
            return {
                "last_mid": {},
                "pepper_anchor": None,
                "pepper_start_anchor": None,
            }

    def dump_state(self, data: dict) -> str:
        return json.dumps(data, separators=(",", ":"))

    # -----------------------------------------------------
    # Market helpers
    # -----------------------------------------------------
    def best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def mid_price(self, order_depth: OrderDepth, fallback: Optional[float] = None) -> Optional[float]:
        best_bid, best_ask = self.best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return fallback

    # -----------------------------------------------------
    # Order placement helpers
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Pepper target schedule
    # -----------------------------------------------------
    def pepper_target_params(self, drift_progress: float):
        """
        drift_progress is roughly:
            current_fair - starting_anchor

        On hidden official data it looked like the total Pepper rise
        was materially smaller than in the sample days, so using drift
        rather than absolute timestamp is a little more robust.

        Returns:
            target_position,
            buy_cushion,
            passive_bid_offset,
            passive_ask_offset,
            passive_bid_size,
            passive_ask_size
        """
        if drift_progress < 25:
            # Early phase: build a meaningful long
            return 55, PEPPER_BUY_CUSHION_EARLY, PEPPER_PASSIVE_BID_OFFSET_EARLY, PEPPER_PASSIVE_ASK_OFFSET_EARLY, 20, 2
        elif drift_progress < 60:
            # Mid phase: still long, but less aggressive
            return 35, PEPPER_BUY_CUSHION_MID, PEPPER_PASSIVE_BID_OFFSET_MID, PEPPER_PASSIVE_ASK_OFFSET_MID, 14, 5
        else:
            # Late phase: keep only a modest long / begin light reduction
            return 12, PEPPER_BUY_CUSHION_LATE, PEPPER_PASSIVE_BID_OFFSET_LATE, PEPPER_PASSIVE_ASK_OFFSET_LATE, 8, 10

    # -----------------------------------------------------
    # Pepper strategy
    # -----------------------------------------------------
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

        # Estimate anchor from fair = anchor + slope * t
        observed_anchor = mid - PEPPER_SLOPE * state.timestamp
        prev_anchor = cache.get("pepper_anchor")
        if prev_anchor is None:
            pepper_anchor = observed_anchor
        else:
            pepper_anchor = (1 - PEPPER_ANCHOR_ALPHA) * prev_anchor + PEPPER_ANCHOR_ALPHA * observed_anchor
        cache["pepper_anchor"] = pepper_anchor

        if cache.get("pepper_start_anchor") is None:
            cache["pepper_start_anchor"] = pepper_anchor

        start_anchor = cache["pepper_start_anchor"]
        fair = pepper_anchor + PEPPER_SLOPE * state.timestamp
        fair -= PEPPER_INV_PENALTY * position

        drift_progress = fair - start_anchor

        (
            target_position,
            buy_cushion,
            passive_bid_offset,
            passive_ask_offset,
            passive_bid_size,
            passive_ask_size,
        ) = self.pepper_target_params(drift_progress)

        # -----------------------------
        # Aggressive buys: while below target, allow buying even slightly
        # above current fair because the trend is so strong.
        # -----------------------------
        for ask_price in sorted(order_depth.sell_orders.keys()):
            ask_qty = -order_depth.sell_orders[ask_price]
            effective_pos = position + buy_used - sell_used
            remaining_to_target = max(0, target_position - effective_pos)

            if remaining_to_target > 0 and ask_price <= fair + buy_cushion:
                qty = min(ask_qty, remaining_to_target + 8)
                buy_used = self.try_buy(
                    orders, product, ask_price, qty, buy_capacity, buy_used
                )
            elif ask_price <= fair - 1.0:
                # Clear value buy even if already near target
                buy_used = self.try_buy(
                    orders, product, ask_price, ask_qty, buy_capacity, buy_used
                )

        # -----------------------------
        # Sells:
        # 1) always sell if bid is genuinely rich
        # 2) if above target, allow gentle unloading near fair
        # -----------------------------
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            bid_qty = order_depth.buy_orders[bid_price]
            effective_pos = position + buy_used - sell_used
            excess_over_target = max(0, effective_pos - target_position)

            if bid_price >= fair + PEPPER_RICH_SELL_EDGE:
                sell_used = self.try_sell(
                    orders, product, bid_price, bid_qty, sell_capacity, sell_used
                )
            elif excess_over_target > 0 and bid_price >= fair + PEPPER_UNWIND_SELL_EDGE:
                qty = min(bid_qty, excess_over_target + 4)
                sell_used = self.try_sell(
                    orders, product, bid_price, qty, sell_capacity, sell_used
                )

        # -----------------------------
        # Passive quoting:
        # Below target -> stronger bid, tiny ask
        # Above target -> more willing ask
        # -----------------------------
        if best_bid is not None and best_ask is not None and best_bid < best_ask:
            effective_pos = position + buy_used - sell_used

            passive_bid = min(best_bid + 1, math.floor(fair + buy_cushion - passive_bid_offset))
            passive_ask = max(best_ask - 1, math.ceil(fair + passive_ask_offset))

            # Bid size depends on how far below target we are
            remaining_to_target = max(0, target_position - effective_pos)
            bid_size = min(passive_bid_size, remaining_to_target + 4) if remaining_to_target > 0 else max(2, passive_bid_size // 2)

            # Ask size depends on whether we are above target
            if effective_pos > target_position:
                ask_size = min(passive_ask_size + 6, effective_pos - target_position + 4)
            else:
                ask_size = passive_ask_size

            if passive_bid < best_ask:
                buy_used = self.try_buy(
                    orders,
                    product,
                    int(passive_bid),
                    int(bid_size),
                    buy_capacity,
                    buy_used,
                )

            if passive_ask > best_bid and effective_pos > 0:
                sell_used = self.try_sell(
                    orders,
                    product,
                    int(passive_ask),
                    int(ask_size),
                    sell_capacity,
                    sell_used,
                )

        cache["last_mid"][product] = mid
        return orders

    # -----------------------------------------------------
    # Osmium strategy
    # -----------------------------------------------------
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

        if last_mid is not None:
            last_move = mid - last_mid
            predicted_next_mid = mid - OSMIUM_MEANREV_ALPHA * last_move
        else:
            predicted_next_mid = mid

        fair = predicted_next_mid - OSMIUM_INV_PENALTY * position

        # Aggressive taking
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

        # Passive market making
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

    # -----------------------------------------------------
    # Main entry point
    # -----------------------------------------------------
    def run(self, state: TradingState):
        cache = self.load_state(state.traderData)

        result: Dict[str, List[Order]] = {
            product: [] for product in state.order_depths
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