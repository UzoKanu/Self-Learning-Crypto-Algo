import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import joblib
from collections import deque
import os
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Setup logging
logging.basicConfig(filename='solana_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# MT5 connection parameters
MT5_SERVER = 'Deriv-Demo'
MT5_LOGIN = 25681005
MT5_PASSWORD = 'Slizzer1$'

# Initialize MT5 with retry logic
def initialize_mt5(retries=3, delay=5):
    for attempt in range(retries):
        if mt5.initialize(server=MT5_SERVER, login=MT5_LOGIN, password=MT5_PASSWORD):
            logging.info("Connected to MetaTrader5")
            print("Connected to MetaTrader5")
            return True
        logging.warning(f"MT5 initialization failed, attempt {attempt + 1}/{retries}: {mt5.last_error()}")
        time.sleep(delay)
    logging.error("MT5 initialization failed after retries")
    print("MetaTrader5 initialization failed")
    return False

# Strategy configuration
SYMBOL = 'SOLUSD.0'
BASE_LOT_SIZE = 20  # Fixed lot size as requested
DATA_HISTORY_FILE = "solana_data_history.pkl"
DATA_HISTORY = deque(maxlen=500)
MODEL_FILE = "solana_rl_trading_model.weights.h5"
SCALER_FILE = "solana_scaler.pkl"
MIN_DATA_POINTS = 50
DAILY_TRADE_LIMIT = 10  # Suitable for intraday focus
TRADES_TODAY = 0
TRAILING_THRESHOLD = 0.8  # Adjusted for decent moves
RRR_OPTIONS = [1.0, 2.0, 3.0]  # Flexible RRR for decent moves
MIN_PERCENT_CHANGE = 0.011  # 1.1% minimum move

# RL parameters
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 0.02  # Reduced to 0.02 for less random HOLD
BASE_EPSILON = 0.02
REWARD_WINDOW = 50
REWARD_THRESHOLD = -50  # Adjusted for decent move rewards
REWARDS = deque(maxlen=REWARD_WINDOW)

# Load or initialize model, scaler, and data history
def initialize_model():
    return Sequential([
        Dense(128, input_dim=8, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='linear')  # 3 actions x 3 RRRs
    ])

def load_or_init_data():
    global DATA_HISTORY, MODEL, SCALER
    if os.path.exists(DATA_HISTORY_FILE):
        try:
            DATA_HISTORY = deque(joblib.load(DATA_HISTORY_FILE), maxlen=500)
            logging.info(f"Loaded {len(DATA_HISTORY)} data points from history")
            print(f"Loaded {len(DATA_HISTORY)} data points from history")
        except Exception as e:
            logging.error(f"Failed to load data history: {e}. Starting with empty history.")
            print(f"Failed to load data history: {e}. Starting with empty history.")
            DATA_HISTORY = deque(maxlen=500)
    else:
        DATA_HISTORY = deque(maxlen=500)
        logging.info("Initialized empty data history")
        print("Initialized empty data history")

    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            MODEL = initialize_model()
            MODEL.load_weights(MODEL_FILE)
            SCALER = joblib.load(SCALER_FILE)
            logging.info("Loaded pre-trained model and scaler")
            print("Loaded pre-trained model and scaler")
        except Exception as e:
            logging.error(f"Failed to load model/scaler: {e}. Initializing new.")
            print(f"Failed to load model/scaler: {e}. Initializing new.")
            MODEL = initialize_model()
            MODEL.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
            SCALER = StandardScaler()
    else:
        MODEL = initialize_model()
        MODEL.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        SCALER = StandardScaler()
        logging.info("Initialized new RL model and scaler")
        print("Initialized new RL model and scaler")

# Check and select symbol
def check_and_select_symbol(symbol):
    if not mt5.symbol_select(symbol, True):
        logging.error(f"Failed to select symbol: {symbol}. Error: {mt5.last_error()}")
        print(f"Failed to select symbol: {symbol}. Error: {mt5.last_error()}")
        return False
    return True

# Fetch market data
def fetch_market_data(symbol, timeframe, bars=500):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None:
            logging.error(f"Failed to get market data for {symbol}: {mt5.last_error()}")
            print(f"Failed to get market data: {mt5.last_error()}")
            return None
        return pd.DataFrame(rates)
    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        print(f"Error fetching market data: {e}")
        return None

# Extract features
def extract_features(df):
    try:
        df['atr'] = df['high'] - df['low']
        df['atr'] = df['atr'].rolling(14).mean()
        df['ema_fast'] = df['close'].ewm(span=10).mean()
        df['ema_slow'] = df['close'].ewm(span=50).mean()
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).mean()))
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        df.dropna(inplace=True)
        return df[['atr', 'ema_fast', 'ema_slow', 'rsi', 'bb_upper', 'bb_lower', 'volatility', 'trend']].values
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        print(f"Error extracting features: {e}")
        return None

# Collect and preprocess data
def collect_data(symbol):
    df = fetch_market_data(symbol, mt5.TIMEFRAME_M15, 500)
    if df is None:
        return None, None, None, None
    features = extract_features(df)
    if features is None or len(features) == 0:
        logging.error("No valid features extracted")
        return None, None, None, None
    if len(DATA_HISTORY) == 0:
        try:
            SCALER.fit(features)
        except Exception as e:
            logging.error(f"Scaler fit failed: {e}")
            return None, None, None, None
    try:
        scaled_features = SCALER.transform(features)
        DATA_HISTORY.append(scaled_features[-1])
        return scaled_features[-1], df['close'].iloc[-1], df['atr'].iloc[-1], df
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        return None, None, None, None

# Detect key levels (support and resistance)
def detect_key_levels(df):
    try:
        # Use a moderate window for decent moves
        highs = df['high'].rolling(window=10, min_periods=1).max()
        lows = df['low'].rolling(window=10, min_periods=1).min()
        key_levels = []
        for i in range(10, len(df) - 10):
            if df['high'].iloc[i] == highs.iloc[i]:
                key_levels.append(('resistance', df['high'].iloc[i]))
            if df['low'].iloc[i] == lows.iloc[i]:
                key_levels.append(('support', df['low'].iloc[i]))
        unique_levels = sorted(set([price for _, price in key_levels]), reverse=True)
        logging.info(f"Detected key levels: {unique_levels}")
        print(f"Detected key levels: {unique_levels}")
        return unique_levels if len(unique_levels) >= 2 else None
    except Exception as e:
        logging.error(f"Error detecting key levels: {e}")
        return None

# RL decision-making with key level logic for decent moves
def decide_action(state, current_price, df):
    global EPSILON
    if len(DATA_HISTORY) < MIN_DATA_POINTS or np.random.rand() < EPSILON:
        return 'HOLD', None

    try:
        key_levels = detect_key_levels(df)
        if key_levels is None:
            return 'HOLD', None

        levels_above = [level for level in key_levels if level > current_price]
        levels_below = [level for level in key_levels if level < current_price]

        if not levels_above or not levels_below:
            return 'HOLD', None

        next_level_above = min(levels_above)
        next_level_below = max(levels_below)
        atr = state[0]  # Extracted from state (first feature is ATR)
        threshold = atr * 2  # Dynamic threshold for decent moves
        rsi = state[3] * 100  # Unscaled RSI

        q_values = MODEL.predict(np.array([state]), verbose=0)[0]
        action_idx = np.argmax(q_values)
        action = ['BUY', 'SELL', 'HOLD'][action_idx // 3]
        rrr = RRR_OPTIONS[action_idx % 3] if action != 'HOLD' else None

        # Override with key level logic for decent moves
        move_above = next_level_above - current_price
        move_below = current_price - next_level_below
        logging.info(f"RSI: {rsi}, Move above: {move_above}, Move below: {move_below}")
        print(f"RSI: {rsi}, Move above: {move_above}, Move below: {move_below}")

        if action == 'BUY' and abs(current_price - next_level_below) < threshold and 25 < rsi < 65:
            target = next_level_above
            return 'BUY', rrr
        elif action == 'SELL' and abs(current_price - next_level_above) < threshold and 35 < rsi < 75:
            target = next_level_below
            return 'SELL', rrr
        return 'HOLD', None
    except Exception as e:
        logging.error(f"Error in RL decision: {e}")
        print(f"Error in RL decision: {e}")
        return 'HOLD', None

# Dynamic position sizing
def calculate_lot_size(atr, base_lot=BASE_LOT_SIZE):
    try:
        risk_factor = max(0.5, min(2.0, 1.0 / atr))  # Adjusted for decent moves
        return round(base_lot * risk_factor, 2)
    except Exception as e:
        logging.error(f"Error calculating lot size: {e}")
        return base_lot

# Place trade
def place_order(symbol, trade_type, lot_size, current_price, atr, rrr, target):
    global TRADES_TODAY
    if TRADES_TODAY >= DAILY_TRADE_LIMIT:
        logging.info("Daily trade limit reached")
        print("Daily trade limit reached")
        return None, None, None, None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick info for {symbol}: {mt5.last_error()}")
        print(f"Failed to get tick info for {symbol}")
        return None, None, None, None

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
        print(f"Failed to get symbol info for {symbol}")
        return None, None, None, None

    try:
        price = tick.ask if trade_type == 'BUY' else tick.bid
        sl_points = price * MIN_PERCENT_CHANGE * 2  # 2.2% SL based on 1.1% minimum
        tp_points = abs(target - price)

        stop_loss = price - sl_points if trade_type == 'BUY' else price + sl_points
        take_profit = target

        order_request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(lot_size),
            'price': float(price),
            'sl': float(stop_loss),
            'tp': float(take_profit),
            'deviation': 20,
            'type': mt5.ORDER_TYPE_BUY if trade_type == 'BUY' else mt5.ORDER_TYPE_SELL,
            'magic': 999997,
            'comment': 'Decent Move Trade',
            'type_filling': mt5.ORDER_FILLING_FOK,
            'type_time': mt5.ORDER_TIME_GTC
        }

        result = mt5.order_send(order_request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Trade placed: {trade_type} at {price}, SL: {stop_loss}, TP: {take_profit}, RRR: {rrr}")
            print(f"Trade placed: {trade_type} at {price}, SL: {stop_loss}, TP: {take_profit}, RRR: {rrr}")
            TRADES_TODAY += 1
            return result.order, price, stop_loss, take_profit
        else:
            logging.error(f"Trade failed: {result.retcode} - {mt5.last_error()}")
            print(f"Trade failed: {result.retcode}")
            return None, None, None, None
    except Exception as e:
        logging.error(f"Error placing order: {e}")
        print(f"Error placing order: {e}")
        return None, None, None, None

# Close trade manually
def close_trade(symbol, order_id, lot_size, trade_type, current_price):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick info for {symbol}: {mt5.last_error()}")
        print(f"Failed to get tick info for {symbol}")
        return False

    try:
        close_price = tick.bid if trade_type == 'BUY' else tick.ask
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': float(lot_size),
            'type': mt5.ORDER_TYPE_SELL if trade_type == 'BUY' else mt5.ORDER_TYPE_BUY,
            'position': order_id,
            'price': float(close_price),
            'deviation': 20,
            'magic': 999997,
            'comment': 'Trade Invalidated',
            'type_filling': mt5.ORDER_FILLING_FOK,
            'type_time': mt5.ORDER_TIME_GTC
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Trade closed at {close_price}")
            print(f"Trade closed at {close_price}")
            return True
        else:
            logging.error(f"Close failed: {result.retcode} - {mt5.last_error()}")
            print(f"Close failed: {result.retcode}")
            return False
    except Exception as e:
        logging.error(f"Error closing trade: {e}")
        print(f"Error closing trade: {e}")
        return False

# Update trailing stop
def update_trailing_stop(order_id, trade_type, entry_price, current_price, sl, atr, tick):
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {SYMBOL}: {mt5.last_error()}")
        print(f"Failed to get symbol info for {SYMBOL}")
        return sl

    try:
        profit_points = (current_price - entry_price) if trade_type == 'BUY' else (entry_price - current_price)
        sl_points = price * MIN_PERCENT_CHANGE * 2

        if profit_points >= sl_points * TRAILING_THRESHOLD:
            new_sl = current_price - sl_points if trade_type == 'BUY' else current_price + sl_points
            if (trade_type == 'BUY' and new_sl > sl) or (trade_type == 'SELL' and new_sl < sl):
                request = {
                    'action': mt5.TRADE_ACTION_SLTP,
                    'position': order_id,
                    'sl': float(new_sl),
                    'tp': mt5.positions_get(ticket=order_id)[0].tp if mt5.positions_get(ticket=order_id) else sl
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logging.info(f"Trailing SL updated to {new_sl}")
                    print(f"Trailing SL updated to {new_sl}")
                    return new_sl
                else:
                    logging.error(f"Trailing stop update failed: {result.retcode}")
                    print(f"Trailing stop update failed: {result.retcode}")
        return sl
    except Exception as e:
        logging.error(f"Error updating trailing stop: {e}")
        print(f"Error updating trailing stop: {e}")
        return sl

# Check trade invalidation
def is_trade_invalid(trade_type, current_price, df, entry_price, atr):
    try:
        ema_fast = df['ema_fast'].iloc[-1]
        ema_slow = df['ema_slow'].iloc[-1]
        volatility = df['volatility'].iloc[-1]
        invalidation_threshold = entry_price * 0.007  # 0.7% drawdown

        if trade_type == 'BUY':
            return current_price < ema_fast or (entry_price - current_price) > invalidation_threshold
        elif trade_type == 'SELL':
            return current_price > ema_fast or (current_price - entry_price) > invalidation_threshold
        return False
    except Exception as e:
        logging.error(f"Error checking trade invalidation: {e}")
        print(f"Error checking trade invalidation: {e}")
        return False

# Reward function with key level bonus
def calculate_reward(trade_type, entry_price, current_price, drawdown, trade_closed=False, invalidated=False, target=None):
    try:
        if trade_type == 'HOLD':
            return 0
        profit = (current_price - entry_price) if trade_type == 'BUY' else (entry_price - current_price)
        reward = profit * 5 - drawdown * 10
        if invalidated:
            reward -= 200
        if trade_closed and target:
            target_distance = abs(target - entry_price)
            if (trade_type == 'BUY' and current_price >= target) or (trade_type == 'SELL' and current_price <= target):
                reward += target_distance * 5
        return reward * 2 if trade_closed else reward
    except Exception as e:
        logging.error(f"Error calculating reward: {e}")
        return 0

# Train RL model
def train_rl_model(state, action_idx, reward, next_state):
    try:
        target = reward + GAMMA * np.max(MODEL.predict(np.array([next_state]), verbose=0)[0])
        target_vec = MODEL.predict(np.array([state]), verbose=0)[0]
        target_vec[action_idx] = target
        MODEL.fit(np.array([state]), np.array([target_vec]), epochs=1, verbose=0)
    except Exception as e:
        logging.error(f"Error training RL model: {e}")
        print(f"Error training RL model: {e}")

# Check for market shift and retrain
def check_market_shift(reward):
    global EPSILON, MODEL
    REWARDS.append(reward)
    if len(REWARDS) == REWARD_WINDOW:
        avg_reward = np.mean(REWARDS)
        logging.info(f"Average reward over last {REWARD_WINDOW} trades: {avg_reward}")
        print(f"Average reward over last {REWARD_WINDOW} trades: {avg_reward}")
        if avg_reward < REWARD_THRESHOLD:
            logging.info("Market shift detected. Retraining model...")
            print("Market shift detected. Retraining model...")
            MODEL = initialize_model()
            MODEL.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
            EPSILON = 0.2
            DATA_HISTORY.clear()
            logging.info("Model reset and exploration increased")
            print("Model reset and exploration increased")
        else:
            EPSILON = max(BASE_EPSILON, EPSILON * 0.95)

# Save state safely
def save_state():
    try:
        MODEL.save_weights(MODEL_FILE)
        joblib.dump(SCALER, SCALER_FILE)
        joblib.dump(list(DATA_HISTORY), DATA_HISTORY_FILE)
        logging.info("Model, scaler, and data history saved")
        print("Model, scaler, and data history saved")
    except Exception as e:
        logging.error(f"Failed to save state: {e}")
        print(f"Failed to save state: {e}")

# Main trading loop
def run_strategy():
    global TRADES_TODAY, EPSILON
    if not initialize_mt5():
        return

    if not check_and_select_symbol(SYMBOL):
        mt5.shutdown()
        return

    load_or_init_data()

    last_day = None
    active_order = None
    entry_price = None
    trade_type = None
    stop_loss = None
    take_profit = None
    target_price = None
    max_drawdown = 0
    lot_size = BASE_LOT_SIZE
    chosen_rrr = None

    try:
        while True:
            current_time = time.localtime()
            current_day = current_time.tm_mday

            if last_day != current_day:
                TRADES_TODAY = 0
                last_day = current_day

            state, current_price, atr, df = collect_data(SYMBOL)
            if state is None or df is None:
                logging.warning("Data collection failed, retrying in 60s")
                print("Data collection failed, retrying in 60s")
                time.sleep(60)
                continue

            if len(DATA_HISTORY) < MIN_DATA_POINTS:
                logging.info(f"Collecting initial data... {len(DATA_HISTORY)}/{MIN_DATA_POINTS} points collected")
                print(f"Collecting initial data... {len(DATA_HISTORY)}/{MIN_DATA_POINTS} points collected")
                time.sleep(60)
                continue

            action, rrr = decide_action(state, current_price, df)
            logging.info(f"Action decided: {action}, RRR: {rrr}")
            print(f"Action decided: {action}, RRR: {rrr}")

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                logging.warning("Tick info unavailable, skipping trade")
                time.sleep(60)
                continue

            lot_size = calculate_lot_size(atr)

            if action != 'HOLD' and active_order is None:
                key_levels = detect_key_levels(df)
                if key_levels:
                    levels_above = [level for level in key_levels if level > current_price]
                    levels_below = [level for level in key_levels if level < current_price]
                    if levels_above and levels_below:
                        target = next_level_above if action == 'BUY' else next_level_below
                        order_id, entry_price, stop_loss, take_profit = place_order(SYMBOL, action, lot_size, current_price, atr, rrr, target)
                        if order_id:
                            active_order = order_id
                            trade_type = action
                            chosen_rrr = rrr
                            target_price = target

            if active_order:
                positions = mt5.positions_get(ticket=active_order)
                if positions is None or len(positions) == 0:
                    reward = calculate_reward(trade_type, entry_price, current_price, max_drawdown, trade_closed=True, target=target_price)
                    next_state, _, _, _ = collect_data(SYMBOL)
                    if next_state is not None:
                        action_idx = ['BUY', 'SELL', 'HOLD'].index(trade_type) * 3 + RRR_OPTIONS.index(chosen_rrr)
                        train_rl_model(state, action_idx, reward, next_state)
                        check_market_shift(reward)
                    active_order = None
                    entry_price = None
                    stop_loss = None
                    take_profit = None
                    target_price = None
                    max_drawdown = 0
                else:
                    if is_trade_invalid(trade_type, current_price, df, entry_price, atr):
                        if close_trade(SYMBOL, active_order, lot_size, trade_type, current_price):
                            reward = calculate_reward(trade_type, entry_price, current_price, max_drawdown, trade_closed=True, invalidated=True, target=target_price)
                            next_state, _, _, _ = collect_data(SYMBOL)
                            if next_state is not None:
                                action_idx = ['BUY', 'SELL', 'HOLD'].index(trade_type) * 3 + RRR_OPTIONS.index(chosen_rrr)
                                train_rl_model(state, action_idx, reward, next_state)
                                check_market_shift(reward)
                            active_order = None
                            entry_price = None
                            stop_loss = None
                            take_profit = None
                            target_price = None
                            max_drawdown = 0
                    else:
                        stop_loss = update_trailing_stop(active_order, trade_type, entry_price, current_price, stop_loss, atr, tick)
                        drawdown = max(0, (entry_price - current_price) if trade_type == 'BUY' else (current_price - entry_price))
                        max_drawdown = max(max_drawdown, drawdown)
                        reward = calculate_reward(trade_type, entry_price, current_price, max_drawdown, target=target_price)
                        next_state, _, _, _ = collect_data(SYMBOL)
                        if next_state is not None:
                            action_idx = ['BUY', 'SELL', 'HOLD'].index(trade_type) * 3 + RRR_OPTIONS.index(chosen_rrr)
                            train_rl_model(state, action_idx, reward, next_state)
                            check_market_shift(reward)

            if len(DATA_HISTORY) % 100 == 0:
                save_state()

            time.sleep(60)

    except KeyboardInterrupt:
        print("Script stopped by user.")
        save_state()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        save_state()
    finally:
        mt5.shutdown()
        logging.info("Disconnected from MetaTrader5")
        print("Disconnected from MetaTrader5")

if __name__ == "__main__":
    run_strategy()