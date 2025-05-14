# this would contain the broker wrapper of angelone 
# the sample use for connection is there in the end
from SmartApi import SmartConnect
import pandas as pd
import pyotp
from logzero import logger
import creds
import requests

class AngelBrokerWrapper:
    def __init__(self, api_key: str, username: str, password: str, token: str, correlation_id: str = "abcde"):
        """
        Initialize the Angel Broker Wrapper.

        :param api_key: API key for authentication.
        :param username: Username for login.
        :param password: Password for login.
        :param token: TOTP token for authentication.
        :param correlation_id: Correlation ID for the session (optional).
        """
        self.api_key = api_key
        self.username = username
        self.password = password
        self.token = token
        self.correlation_id = correlation_id
        self.smart_api = SmartConnect(api_key=self.api_key)
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        self.connect()

    def connect(self):
        """
        Establish a connection and authenticate the user.
        """
        try:
            totp = pyotp.TOTP(self.token).now()
            print("token is:")
            print(self.token)
            data = self.smart_api.generateSession(self.username, self.password, totp)

            if data['status'] == False:
                logger.error(data)
                raise Exception("Authentication failed.")

            self.auth_token = data['data']['jwtToken']
            print("auth token is:")
            print(self.auth_token)
            self.refresh_token = data['data']['refreshToken']
            print("refresh token is")
            print(self.refresh_token)
            self.feed_token = self.smart_api.getfeedToken()
            print("feed token is")
            print(self.feed_token)
            logger.info("Connection established and authenticated.")
        except Exception as e:
            logger.error("Failed to connect: %s", e)
            raise e

    def disconnect(self):
        """
        Disconnect the session.
        """
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None
        logger.info("Disconnected from the broker.")

    def authenticate(self):
        """
        Refresh the authentication token if necessary.
        """
        try:
            self.smart_api.generateToken(self.refresh_token)
            logger.info("Token refreshed successfully.")
        except Exception as e:
            logger.error("Failed to refresh token: %s", e)
            raise e

    def place_order(self, order_params: dict):
        """
        Place an order.

        :param order_params: A dictionary containing order parameters.
        :return: Order ID or response details.
        """
        try:
            order_id = self.smart_api.placeOrder(order_params)
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error("Failed to place order: %s", e)
            raise e

    def get_orders(self):
        try:
            order_details = self.smart_api.position()
            return order_details
        except Exception as e:
            logger.error("Failer to get orders")
            raise e
        
    def get_orderbook(self):
        try:
            orderbook_details = self.smart_api.orderBook()
            return orderbook_details
        except Exception as e:
            logger.error("Failer to get orderbook")
            
    def cancel_position(self,direction,quantity):
        
        if direction=="BUY":
            main_order_response = self.smart_api.placeOrder({ # AngelBrokerWrapper.place_order() to be used here
                    "variety": "NORMAL",
                    "tradingsymbol": creds.tradingsymbol,
                    "symboltoken": creds.token_id,
                    "transactiontype": "SELL",
                    "exchange": creds.exchange,
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price":"0",
                    "quantity": quantity,
                })
            print(main_order_response)
            
        else:
           main_order_response = self.smart_api.placeOrder({
                    "variety": "NORMAL",
                    "tradingsymbol": creds.tradingsymbol,
                    "symboltoken": creds.token_id,
                    "transactiontype": "BUY",
                    "exchange": creds.exchange,
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "price":"0",
                    "quantity": quantity,
                }) 
           print(main_order_response)

    def check_order_status(self, order_id: str):
        """
        Check the status of an order.

        :param order_id: The ID of the order.
        :return: Order status details.
        """
        try:
            # Fetch all orders using the `orderBook` method
            order_book = self.smart_api.orderBook()

            if isinstance(order_book, dict) and order_book.get("data"):
                orders = order_book["data"]
                # Filter the order book for the specific order ID
                order_status = next((order for order in orders if order.get("orderid") == order_id), None)

                if order_status:
                    logger.info(f"Order status for {order_id}: {order_status}")
                    return order_status
                else:
                    raise Exception(f"Order with ID {order_id} not found in the order book.")
            else:
                raise Exception("Unexpected response format for order book.")
        except Exception as e:
            logger.error("Failed to check order status: %s", e)
            raise e

    def modify_order(self, order_id: str, new_params: dict):
        """
        Modify an existing order.

        :param order_id: The ID of the order.
        :param new_params: New parameters for the order.
        :return: Modification confirmation details.
        """
        try:
            response = self.smart_api.modifyOrder(order_id, new_params)
            logger.info(f"Order {order_id} modified successfully: {response}")
            return response
        except Exception as e:
            logger.error("Failed to modify order: %s", e)
            raise e

    def cancel_order(self, order_id: str,variety="NORMAL"):
        """
        Cancel an existing order.

        :param order_id: The ID of the order.
        :return: Cancellation confirmation details.
        """
        try:
            response = self.smart_api.cancelOrder(order_id=order_id,variety=variety)
            # response = self.smart_api.cancelOrder(order_id,variety=variety)
            logger.info(f"Order {order_id} canceled successfully: {response}")
            return response
        except Exception as e:
            logger.error("Failed to cancel order: %s", e)
            raise e

    def get_account_balance(self):
        """
        Retrieve account balance details.

        :return: Account balance information.
        """
        try:
            # balance = self.smart_api.getBalance()
            # balance = self.smart_api.getMarginApi()
            balance = self.smart_api.rmsLimit()
            balance = balance['data']['availablecash']
            logger.info(f"Account balance: {balance}")
            return balance
        except Exception as e:
            logger.error("Failed to retrieve account balance: %s", e)
            raise e

        
    def get_profile(self):
        val = self.smart_api.getProfile(refreshToken=creds.token)
        return val

    def get_market_data(self, symbol: str):
        """
        Retrieve market data for a specific symbol.

        :param symbol: The trading symbol.
        :return: Market data details.
        """
        try:
            data = self.smart_api.getMarketData("FULL",{ "NSE": ["3045","881"], })
            print(f"Market data for {symbol}: {data}")
            return data
        except Exception as e:
            print("Failed to retrieve market data: %s", e)
            raise e
        
    def last_trade_prc(self,symbol):
        ltp_data = self.smart_api.ltpData(exchange=creds.exchange,tradingsymbol=creds.tradingsymbol,symboltoken=creds.token_id)
        data = ltp_data['data']
        close = data['close']
        return close
    
    def get_candle_data(self,exchange, symboltoken,interval,fromdate,todate):
        data=self.smart_api.getCandleData({
     "exchange": exchange,
     "symboltoken": symboltoken,
     "interval": interval,
     "fromdate": fromdate,
     "todate": todate
    })
        df=pd.DataFrame(data["data"])
        
        df.columns=["datetime","open","high","low","close","volume"]
        print(df)
        return df
    
    def download_new_data(self):
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        d = requests.get(url).json()
        token_df = pd.DataFrame.from_dict(d)
        token_df["expiry"] = pd.to_datetime(token_df["expiry"])
        token_df = token_df.astype({"strike":float})
        token_df.to_excel("alltokens.xlsx")
        df=token_df
        creds.token_info = df[
        (df['exch_seg'].isin(["NFO", "BFO"])) & 
        (df['instrumenttype'].isin(["OPTIDX", "OPTSTK"])) & 
        (df['name'].isin(["BANKNIFTY", "MIDCPNIFTY", "FINNIFTY", "NIFTY", "SENSEX"]))]
        print("token_info saved")
        print(creds.token_info)
        creds.token_info.to_excel("tokens.xlsx")

    def main(self):
        self.df=wrapper.get_candle_data("NSE","99926000","ONE_MINUTE","2025-01-01 11:15","2025-01-10 12:00")
        self.calculate_indicators()

    def new_main(self):
        # self.details = self.get_orderbook()
        # self.details2 = self.get_orders()
        # print(self.details)
        # print(self.details2)
        self.connect()

# Example usage:
if __name__ == "__main__":
    api_key = 'AYq866MN'
    username = 'N60066209'
    password = '1010'
    token = "TJYZILIYM56MEDGUZXXKL7QOKE"

    wrapper = AngelBrokerWrapper(api_key=api_key, username=username, password=password, token=token)
    # wrapper.main()
    wrapper.new_main()
    wrapper.disconnect()

    # order_params = {
    #     "variety": "NORMAL",
    #     "tradingsymbol": "SBIN-EQ",
    #     "symboltoken": "3045",
    #     "transactiontype": "BUY",
    #     "exchange": "NSE",
    #     "ordertype": "LIMIT",
    #     "producttype": "INTRADAY",
    #     "duration": "DAY",
    #     "price": "19500",
    #     "squareoff": "0",
    #     "stoploss": "0",
    #     "quantity": "1"
    # }
    #ohlc = wrapper.get_market_data("SBIN-EQ")
    
    # self.df=wrapper.get_candle_data("NSE","99926000","ONE_MINUTE","2025-01-01 11:15","2025-01-10 12:00")
    # order_id = wrapper.place_order(order_params)
    # wrapper.check_order_status(order_id)
    # wrapper.calculate_indicators()
    # #wrapper.download_new_data()
    