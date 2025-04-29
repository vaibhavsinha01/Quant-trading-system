"""api credentials for login"""
api_key = 'YHcOO9O3'
username = 'P479823'
password = '1118'
token = 'AKHRYAVFX36WALYNW2YIJ27ADI'

""" paths to files """
xlsx_path = r"C:\Users\vaibh\OneDrive\Desktop\ribbon_1\t3_ribbon_3rd_march (3)\t3_ribbon_3rd_march\t3-ribbon\alltokens.xlsx"
current_orders_path = r"C:\Users\vaibh\OneDrive\Desktop\ribbon_1\t3_ribbon_3rd_march (3)\t3_ribbon_3rd_march\t3-ribbon\current_orders.csv"

""" information about the stocks that would be traded """
leverage = 1 # set the leverage the default to 5
equity_choice = 0 # changed from 0 
exchange = "NSE"
option_exchange = "NFO"
token_id = "3045" # "99926000" # "2885" # "3045" # token_id = "11536" # token_id = "99926000" # token_id="12815"# token_id="9931"
tradingsymbol = "SBIN-EQ" # "NIFTY50" # "SBIN-EQ" # changed from SBIN-EQ # tradingsymbol = "SBIN-EQ" # tradingsymbol = "TCS-EQ" # tradingsymbol = "NIFTY50" #tradingsymbol="TATAINVEST-BL" # tradingsymbol="VIKASLIFE-EQ"
timeframe="FIFTEEN_MINUTE" # "ONE_MINUTE","THREE_MINUTE","FIVE_MINUTE","TEN_MINUTE","FIFTEEN_MINUTE","THIRTY_MINUTE","ONE_HOUR","ONE_DAY"
profit_percentage = 0.005 # 0.5 percentage tp
max_mul = 2
x_factor = 3 # make this more than one ruppee # suppose it is 50 points take min of price-50 / price*(1-0.01)
y_factor = 20 # this is the maximum loss due to fake entries
use_data_for_x_days = 380 # 100
min_max_choice = 1
heiken_ashi_choice = 1
initial_quantity = 1 
fake_entry_max_tries = 3
IS_BO=0
option_trading_symbol = "SBIN24APR25730CE"
option_trading_id = "121680"

"""
leverage - if you set the default leverage to 5 then the quantity would be 5 for trading if the initial_quantity is 5
day_range - 3 - no of days ahead we look for options
equity_choice 0 - equity markets
equity_choice 1 - options markets
exchange - NSE/BSE (equity) NFO/BFO (options)
token_id/trading_symbol - For equity trading (in case of stock trading you would need to put both (get them from the alltoken.xlsx file))
option_symbol - For option trading
timeframe - EX - "ONE_MINUTE"
take_profit_pips - the amount that you want to set for your profit (ex - 0.1 for Vodafone-Idea , 50 - NIFTY50)
profit_percentage - sets the profit percent
max_mul - the maximum amount of times the trade can be taken for martingle (eg max_mul=3 takes the martingle trade only once(2x) max_mul = 7 takes the martingle trade only twice(4x))
x_factor - ex - 50 points for nifty
y_factor - ex - maximum loss for the fake entry before the multiplier increases by 2
use_data_for_x_days - 100(keep it default - used in calculation of indicators) keep it default 100 ***don't change
option_range - this enables you to set a range for getting options nearby like if the ltp is 718.5 -> 720(round of) then we get the pe/ce for 690-750 if we set the option_range to 30
option_range_increment - increments the option by this range so in the above example we will get the options for 690,700,710,720,730,740,750 for ce and pe
min_max_choice 0 - if the choice is set to zero then minima is chosen for taking the stoploss b/w normal_stop_loss and x_factor stop loss
min_max_choice 1 - if the choice is set to one then maxima is chosen for taking the stoploss b/w normal_stop_loss and x_factor stop loss
heiken_ashi_choice 0 - if choice is 0 then calculates indicators using normal values
heiken_ashi_choice 1 - if choice is 1 then calculates indicators using heiken-ashi values
initial_quanity - the initial quantity to trade with default is 1
fake_entry_max_tries - set the max tries for checking if a fake trade is taking place
"""