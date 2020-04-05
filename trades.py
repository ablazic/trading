
from bittrex.bittrex import Bittrex, API_V2_0, API_V1_1, TIMEINEFFECT_IMMEDIATE_OR_CANCEL, ORDERTYPE_MARKET, ORDERTYPE_LIMIT, TIMEINEFFECT_FILL_OR_KILL
from pandas.io.json import json_normalize

# Enter your API key & secret within ''
API_KEY = ''

API_SECRET = ''

# API version being used - V2_0 doesn't have good documentation yet
API_VERSION = API_V1_1


class BTCClient:
    
    # Change market here
    market = "USDT-BTC"

    def __init__(self):
        self.client = Bittrex(API_KEY, API_SECRET, api_version=API_VERSION)
    
    # get market data
    def get_markets(self):
        return self.client.get_markets()["result"]
    
    
    def get_balance(self):
        pass

    # Place a buy order
    def buy(self, quantity, rate):
        return self.parse_response(self.client.trade_buy(market=self.market, quantity=quantity,
                              order_type=ORDERTYPE_LIMIT,
                              time_in_effect=TIMEINEFFECT_IMMEDIATE_OR_CANCEL,
                              rate=rate
                              ))
    # Place a sell order
    def sell(self, quantity, rate):
        return self.parse_response(self.client.trade_sell(market=self.market, quantity=quantity,
                               order_type=ORDERTYPE_LIMIT,
                               time_in_effect=TIMEINEFFECT_IMMEDIATE_OR_CANCEL,
                               rate=rate))
    
    # Get list of open orders
    def openorders(self):
        self.client.get_open_orders(market=self.market)

    def parse_response(self, response):
        result = response["result"]
        success = response["success"]
        if not success:
            raise ValueError(response["message"])
        return result
   
    # Get balances for different assets on bittrex
    def get_balances(self):
        return self.parse_response(self.client.get_balances())
    
    # get orderbook
    def get_orderbook(self):
        return self.parse_response(self.client.get_order_history(market=self.market))

    # Get current bid/ ask data
    def get_bid_ask(self):
        return self.parse_response(self.client.get_market_summary(self.market))

if __name__ == '__main__':
    client = BTCClient()

    for bal in client.get_balances():
        print(bal)
    # #
    for order in client.get_orderbook():
        print(order)
    #
    # r = client.get_bid_ask()[0]
    # print(r)

