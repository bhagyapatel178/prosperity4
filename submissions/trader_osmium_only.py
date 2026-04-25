from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math

OSMIUM = "ASH_COATED_OSMIUM"

POSITION_LIMITS = {
    OSMIUM: 80,
}

# Osmium: strong short-horizon mean reversion
OSMIUM_MEANREV_ALPHA = 0.90
OSMIUM_INV_PENALTY = 0.10
OSMIUM_TAKE_EDGE = 1.0
OSMIUM_PASSIVE_OFFSET = 2.0
OSMIUM_PASSIVE_SIZE = 10


class Trader:
    def load_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {
                "last_mid": {},
            }
        try:
            data = json.loads(trader_data)
            if "last_mid" not in data:
                data["last_mid"] = {}
            return data
        except Exception:
            return {
                "last_mid": {},
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

    def run(self, state: TradingState):
        cache = self.load_state(state.traderData)

        result: Dict[str, List[Order]] = {
            product: [] for product in state.order_depths
        }

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self.trade_osmium(
                state, state.order_depths[OSMIUM], cache
            )

        trader_data = self.dump_state(cache)
        conversions = 0
        return result, conversions, trader_data