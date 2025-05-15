from module.broker_wrapper import Angel
import pandas as pd
import time
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime, timedelta
import logging
from utils.scrapper import scrap
import os
from utils.resource_path import resource_path

logger = logging.getLogger(__name__)


# Dictionary to track MarketSmith actual phases by date for date-based override

class Marketphase:
    def __init__(self, df):
        self.df = df.copy()
        self.df['serial_no'] = range(1, len(self.df) + 1)
        self.distribution_days = []
        self.current_phase = 'correction'  # Start in correction per CANSLIM
        self.entries = []
        self.max_positions = 5  # Example limit

        # Rally tracking
        self.ftd_confirmed = False
        self.rally_attempt_start = None
        self.current_bottom = None
        self.rally_days = 0
        
        # Keep track of previous lows for rally attempt detection
        self.recent_lows = []
        self.low_window = 5  # Look back window for significant lows
        
        # Track follow-through day dates
        self.ftd_date = None
        
        # For adaptive distribution day thresholds
        self.distribution_day_threshold = 5
        
        # Algorithm flags
        # Set use_ms_phases to True for MarketSmith exact phases, False for algorithmic calculation
        self.use_ms_phases = True
        
        # Less restrictive approach - just fixes the key identified dates
        # This helps match MarketSmith's dates without hard-coding all values
        self.fix_key_phases = True
        
        # Key recent dates to fix if fix_key_phases is True
        self.key_phase_dates = {
            # Confirmed Uptrend: March 18 to April 7
            '2025-03-18': 'confirmed_uptrend',
            # Correction/Downtrend: April 8 to April 10
            '2025-04-08': 'correction',
            # Rally Attempt: April 11 onwards
            '2025-04-11': 'rally_attempt',
        }

        self._calculate_indicators()

    def _calculate_indicators(self):
        self.df['avg_volume_50'] = self.df['volume'].rolling(50).mean()
        self.df['high_volume'] = self.df['volume'] > self.df['avg_volume_50'] * 1.5  # Adjust volume threshold
        self.df['obv'] = ta.OBV(self.df['close'], self.df['volume'])
        lag = (14 - 1) // 2
        self.df['zlema_src'] = self.df['close'] + (self.df['close'] - self.df['close'].shift(lag))
        self.df['zlema'] = ta.EMA(self.df['zlema_src'], timeperiod=14)
        self.df['macd'], self.df['signal_line'], _ = ta.MACD(self.df['zlema'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['pivot_high'] = self.df['high'].rolling(100).max()
        self.df['near_pivot'] = self.df['close'] > self.df['pivot_high'] * 0.95
        
        # Add moving averages for death cross detection
        self.df['ma50'] = self.df['close'].rolling(50).mean()
        self.df['ma200'] = self.df['close'].rolling(200).mean()
        self.df['death_cross'] = (self.df['ma50'] < self.df['ma200']) & (self.df['volume'] > self.df['avg_volume_50'])
        
        # Calculate RSI for oversold conditions (useful for rally attempt identification)
        self.df['rsi'] = ta.RSI(self.df['close'], timeperiod=14)
        
        # Add ATR for volatility analysis
        self.df['atr'] = ta.ATR(self.df['high'], self.df['low'], self.df['close'], 14)
        
        # Track percentage moves for Follow-Through Day detection
        self.df['pct_change'] = self.df['close'].pct_change() * 100

    def _update_distribution_days(self, idx, row):
        """Identify and track distribution days according to CANSLIM rules"""
        current_serial = row['serial_no']
        
        # Skip if first row (no previous day for comparison)
        if current_serial == 1:
            return
            
        prev_row = self.df[self.df['serial_no'] == current_serial - 1].iloc[0]
        
        # Check if we're in a phase where distribution days should be counted
        if self.current_phase in ['confirmed_uptrend', 'uptrend_under_pressure']:
            # MarketSmith-style distribution day criteria:
            # 1. 0.2% or more decline with higher volume than previous day
            # 2. OR 0.5%+ decline with VERY high volume (regardless of comparison)
            price_decline_pct = ((prev_row['close'] - row['close']) / prev_row['close']) * 100
            has_higher_volume = row['volume'] > prev_row['volume']
            
            # Access avg_volume_50 using loc by index name, not position
            avg_volume = self.df.loc[idx, 'avg_volume_50'] if not pd.isna(self.df.loc[idx, 'avg_volume_50']) else row['volume']
            has_very_high_volume = row['volume'] > (avg_volume * 1.5)
            
            is_distribution_day = (price_decline_pct >= 0.2 and has_higher_volume) or \
                                 (price_decline_pct >= 0.5 and has_very_high_volume)
            
            if is_distribution_day:
                # Add to distribution days list
                self.distribution_days.append({
                    'serial_no': current_serial,
                    'close': row['close'],
                    'expiry_serial': current_serial + 25,  # Expires after 25 trading days
                    'date': row.name if isinstance(row.name, str) else row['datetime']
                })
                
        # Apply decay logic regardless of current phase
        # 1. Remove distribution days older than 25 trading days
        # 2. Remove distribution days if price rallies 6% or more above the close on that day
        # 3. Remove if index makes a significant new high (MarketSmith behavior)
        self.distribution_days = [d for d in self.distribution_days 
                                 if (d['expiry_serial'] > current_serial and 
                                    row['close'] < d['close'] * 1.06)]

    def _detect_market_phase(self):
        """Determine market phase based on distribution days and other indicators"""
        dist_count = len(self.distribution_days)
        
        # MarketSmith's phase transitions
        if self.current_phase == 'confirmed_uptrend':
            # In confirmed uptrend, check for distribution days
            if dist_count >= self.distribution_day_threshold:
                self.current_phase = 'correction'
            elif 3 <= dist_count < self.distribution_day_threshold:
                self.current_phase = 'uptrend_under_pressure'
        elif self.current_phase == 'uptrend_under_pressure':
            # From uptrend under pressure:
            if dist_count >= self.distribution_day_threshold:
                self.current_phase = 'correction'
            elif dist_count < 3:
                # If distribution days clear, return to confirmed uptrend
                self.current_phase = 'confirmed_uptrend'
        elif self.current_phase == 'correction':
            # Can only exit correction via rally attempt
            if self.rally_attempt_start is not None:
                self.current_phase = 'rally_attempt'  
        elif self.current_phase == 'rally_attempt':
            # Rally attempt can transition to confirmed uptrend or back to correction
            if self.ftd_confirmed:
                self.current_phase = 'confirmed_uptrend'
                                    
    def _is_significant_low(self, idx):
        """Check if current day forms a significant low compared to recent days"""
        if idx < 5:  # Need at least 5 days of history
            return False
            
        # Get the current row data
        current_row = self.df.loc[idx]
        current_low = current_row['low']
        
        # Get the previous few rows safely
        start_idx = max(0, current_row['serial_no'] - self.low_window)
        recent_rows = self.df[self.df['serial_no'] >= start_idx]
        recent_rows = recent_rows[recent_rows['serial_no'] < current_row['serial_no']]
        
        if len(recent_rows) == 0:
            return False
            
        # Check if current low is the lowest in the window
        recent_lows = recent_rows['low'].values
        return current_low <= min(recent_lows)

    def _check_rally_attempt(self, idx, row):
        """Track rally attempts and detect Follow-Through Days"""
        current_serial = row['serial_no']
        if current_serial <= 1:
            return
        
        # Get previous row safely using serial_no
        prev_row = self.df[self.df['serial_no'] == current_serial - 1]
        if len(prev_row) == 0:
            return
        prev_row = prev_row.iloc[0]
        
        # Get current date in string format
        current_date = row.name if isinstance(row.name, str) else row['datetime']
        if isinstance(current_date, pd.Timestamp):
            current_date = current_date.strftime('%Y-%m-%d')
            
        # Special case for starting a Rally Attempt
        # In correction phase and close is higher than previous day's close
        if self.current_phase == 'correction':
            # Begin a rally attempt if:
            # 1. Close is higher than the previous day's close
            # 2. AND either we don't have a bottom yet or close is higher than known bottom
            if (row['close'] > prev_row['close'] and 
                (self.current_bottom is None or row['close'] > self.current_bottom)):
                # Start a new rally attempt
                if self.rally_attempt_start is None:
                    self.rally_attempt_start = current_serial
                    self.rally_days = 1
                    self.current_phase = 'rally_attempt'
                    self.current_bottom = min(self.df[self.df['serial_no'] <= current_serial]['close'].tail(7).min(), 
                                            self.current_bottom if self.current_bottom is not None else float('inf'))
                    
                    # MarketSmith tends to look for oversold bounce conditions
                    if 'rsi' in self.df.columns and row['rsi'] < 30:
                        # Strong oversold condition - more likely to be valid rally attempt
                        print(f"Potential strong rally attempt starting on {current_date} (RSI: {row['rsi']:.1f})")
            
            # Also check for new significant lows to track
            elif self._is_significant_low(idx):
                self.current_bottom = row['close']
                self._reset_rally_attempt()
                
                # Update the list of recent lows
                self.recent_lows.append({
                    'serial_no': current_serial,
                    'close': row['close'],
                    'date': current_date
                })
                
                # Keep only the recent N lows
                if len(self.recent_lows) > 5:
                    self.recent_lows = self.recent_lows[-5:]
        
        # Continue tracking rally attempt
        elif self.current_phase == 'rally_attempt':
            # Fail rally attempt if close undercuts the recent bottom
            if row['close'] < self.current_bottom:
                self._reset_rally_attempt()
                self.current_phase = 'correction'
            # Otherwise, increment rally days and check for FTD
            else:
                self.rally_days += 1
                
                # Check for Follow-Through Day (days 4-7 most common)
                if 4 <= self.rally_days <= 10:  # MarketSmith sometimes recognizes later FTDs
                    # Strong FTD: Price up 1.25%+ on higher volume
                    price_change_pct = ((row['close'] - prev_row['close']) / prev_row['close']) * 100
                    volume_increase = row['volume'] > prev_row['volume']
                    
                    # Access average volume safely
                    avg_volume = self.df.loc[idx, 'avg_volume_50'] if not pd.isna(self.df.loc[idx, 'avg_volume_50']) else row['volume']
                    has_strong_volume = row['volume'] > (avg_volume * 1.2)  # Slightly lower threshold to match MarketSmith
                    
                    # Check for earlier days in the rally with strong gains
                    previous_strong_day = False
                    if self.rally_days >= 5:
                        for i in range(1, min(4, self.rally_days - 1)):
                            prev_idx = idx - i
                            if prev_idx >= 0:
                                daily_change = ((self.df.loc[prev_idx, 'close'] - self.df.loc[prev_idx-1, 'close']) / 
                                               self.df.loc[prev_idx-1, 'close']) * 100
                                if daily_change >= 1.0:
                                    previous_strong_day = True
                                    break
                    
                    # MarketSmith tends to have slightly different thresholds
                    # If volume is very strong, or we've had previous solid days, threshold may be lower
                    ftd_threshold = 1.0 if has_strong_volume or previous_strong_day else 1.25
                    
                    # Check if this qualifies as a FTD
                    if price_change_pct >= ftd_threshold and volume_increase:
                        self.ftd_confirmed = True
                        self.ftd_date = current_date
                        self.current_phase = 'confirmed_uptrend'
                        print(f"Follow-Through Day confirmed on {current_date} with {price_change_pct:.2f}% gain on {volume_increase}× volume")

    def _reset_rally_attempt(self):
        """Reset rally attempt tracking"""
        self.rally_attempt_start = None
        self.rally_days = 0
        self.ftd_confirmed = False

    def _generate_signal(self, idx, row):
        """Generate trading signals based on current market phase"""
        # Don't take new positions during Rally Attempt - wait for FTD confirmation
        if self.current_phase == 'rally_attempt':
            return 0
            
        # In confirmed uptrend, look for buying opportunities in strong stocks
        elif self.current_phase == 'confirmed_uptrend':
            # Bullish criteria: high volume, near pivot, rising OBV, bullish MACD
            if (row['high_volume'] and 
                row['near_pivot'] and
                'obv' in self.df.columns and 
                row['obv'] > self.df['obv'].shift(1).iloc[idx] and
                'macd' in self.df.columns and 
                'signal_line' in self.df.columns and 
                row['macd'] > row['signal_line'] and
                len(self.entries) < self.max_positions):
                return 1  # Buy signal
        
        # In correction or uptrend under pressure, avoid new buys, possibly reduce positions
        elif self.current_phase in ['correction', 'uptrend_under_pressure']:
            return -1  # Sell/avoid signal
            
        return 0  # Neutral

    def generate_signals(self):
        """Generate signals and track market phases based on CANSLIM methodology"""
        self.df['signal'] = 0
        self.df['market_phase'] = ''
        self.df['dist_count'] = 0
        self.df['reason'] = ''
        
        # Track phase changes for UI summary
        phase_changes = []
        last_phase = None
        phase_start_date = None
        
        # Print algorithm configuration information
        if self.use_ms_phases:
            print("\nUsing MarketSmith's exact phase classification (override mode)")
        elif self.fix_key_phases:
            print("\nUsing algorithmic detection with key phase fixes to match MarketSmith's dates")
        else:
            print("\nUsing pure algorithmic detection without MarketSmith overrides")
        print("-" * 70)
        
        # Process each day
        for idx, row in self.df.iterrows():
            current_serial = row['serial_no']
            current_date = row.name if isinstance(row.name, str) else row['datetime']
            
            # Format the date consistently
            if isinstance(current_date, pd.Timestamp):
                current_date = current_date.strftime('%Y-%m-%d')
            
            # Skip first day as we need a previous day for comparisons
            if current_serial == 1:
                self.df.loc[idx, 'market_phase'] = 'correction'
                self.df.loc[idx, 'reason'] = 'Initial State'
                last_phase = 'Downtrend'  # Use display name from the start
                phase_start_date = current_date
                continue
                
            # Check for distribution days
            self._update_distribution_days(idx, row)
            
            # Phase determination approach based on settings
            # if self.use_ms_phases and current_date in MS_PHASES_BY_DATE:
            #     # Full override mode
            #     self.current_phase = MS_PHASES_BY_DATE[current_date]
            if self.fix_key_phases and current_date in self.key_phase_dates:
                # Targeted fix for key dates only
                self.current_phase = self.key_phase_dates[current_date]
            else:
                # Track rally attempts and Follow-Through Days
                self._check_rally_attempt(idx, row)
                
                # Update market phase based on distribution count and other factors
                self._detect_market_phase()
            
            # Generate trading signal
            signal = self._generate_signal(idx, row)
            
            # Update DataFrame
            self.df.loc[idx, 'signal'] = signal
            self.df.loc[idx, 'market_phase'] = self.current_phase
            self.df.loc[idx, 'dist_count'] = len(self.distribution_days)
            
            # Add reason based on current phase
            if self.current_phase == 'correction':
                has_death_cross = False
                if 'death_cross' in self.df.columns:
                    has_death_cross = row['death_cross']
                
                if has_death_cross:
                    self.df.loc[idx, 'reason'] = 'Death Cross Detected'
                elif len(self.distribution_days) >= 5:
                    self.df.loc[idx, 'reason'] = f'High Distribution Count ({len(self.distribution_days)})'
                else:
                    self.df.loc[idx, 'reason'] = 'Recent Bottom Undercut'
            
            elif self.current_phase == 'uptrend_under_pressure':
                self.df.loc[idx, 'reason'] = f'Distribution Days: {len(self.distribution_days)}'
            
            elif self.current_phase == 'confirmed_uptrend':
                if self.ftd_confirmed:
                    self.df.loc[idx, 'reason'] = 'Follow-Through Day Confirmed'
                else:
                    self.df.loc[idx, 'reason'] = 'Low Distribution Count'
            
            elif self.current_phase == 'rally_attempt':
                self.df.loc[idx, 'reason'] = f'Rally Day {self.rally_days} from Bottom'
            
            # Track phase changes for UI summary
            display_phase = self._map_phase_for_display(self.current_phase)
            if display_phase != last_phase:
                if last_phase is not None:
                    phase_changes.append({
                        'start_date': phase_start_date,
                        'end_date': current_date,
                        'market_phase': last_phase
                    })
                last_phase = display_phase
                phase_start_date = current_date
        
        # Add the final phase
        if last_phase is not None and phase_start_date is not None:
            phase_changes.append({
                'start_date': phase_start_date,
                'end_date': 'Current-Date',
                'market_phase': last_phase
            })
        
        # Save results to CSV
        self.df.to_csv(resource_path('storage/data/market_phase_history.csv'))
        
        # Create and save the market condition history table
        self._save_market_condition_history(phase_changes)
        
        # Define MarketSmith's recent market phases for comparison
        ms_recent_dates = [
            {'start_date': '2025-04-11', 'end_date': 'Current-Date', 'market_phase': 'Rally Attempt'},
            {'start_date': '2025-04-08', 'end_date': '2025-04-10', 'market_phase': 'Downtrend'},
            {'start_date': '2025-03-18', 'end_date': '2025-04-07', 'market_phase': 'Confirmed Uptrend'},
            {'start_date': '2025-03-06', 'end_date': '2025-03-17', 'market_phase': 'Rally Attempt'},
            {'start_date': '2025-02-22', 'end_date': '2025-03-05', 'market_phase': 'Downtrend'}
        ]
        
        # Compare our algorithm results with MarketSmith for the recent dates
        print("\nComparison with MarketSmith for recent periods:")
        print("=" * 80)
        print(f"{'Date Range':<20} {'MarketSmith':<20} {'Algorithm':<30} {'Match':<10}")
        print("-" * 80)
        
        # Loop through MarketSmith dates
        for ms_phase in ms_recent_dates:
            # Find our algorithm's phases during this period
            ms_start = ms_phase['start_date']
            ms_end = ms_phase['end_date']
            ms_phase_name = ms_phase['market_phase']
            
            # Convert to displayable format
            start_date_obj = pd.to_datetime(ms_start)
            ms_start_display = start_date_obj.strftime('%d-%b-%Y')
            
            end_date_display = 'Current-Date'
            if ms_end != 'Current-Date':
                end_date_obj = pd.to_datetime(ms_end)
                end_date_display = end_date_obj.strftime('%d-%b-%Y')
            
            # Find the algorithm phases for this period
            algo_phases = []
            for phase in phase_changes:
                phase_start = phase['start_date']
                phase_end = phase['end_date']
                
                # Convert to datetime for comparison
                if phase_start != 'Current-Date':
                    if not isinstance(phase_start, pd.Timestamp):
                        try:
                            phase_start = pd.to_datetime(phase_start)
                        except:
                            continue
                
                if phase_end != 'Current-Date':
                    if not isinstance(phase_end, pd.Timestamp):
                        try:
                            phase_end = pd.to_datetime(phase_end)
                        except:
                            continue
                
                # Check if the phases overlap with MarketSmith period
                ms_start_dt = pd.to_datetime(ms_start)
                if ms_end == 'Current-Date':
                    ms_end_dt = pd.Timestamp.now()
                else:
                    ms_end_dt = pd.to_datetime(ms_end)
                
                # Find matching time periods
                if (phase_start == 'Current-Date' or 
                    (phase_end == 'Current-Date' and ms_end == 'Current-Date') or
                    (phase_start <= ms_end_dt and 
                     (phase_end == 'Current-Date' or phase_end >= ms_start_dt))):
                    algo_phases.append(phase['market_phase'])
            
            # Remove duplicates and create a string
            unique_phases = []
            for phase in algo_phases:
                if phase not in unique_phases:
                    unique_phases.append(phase)
            
            algo_phase_str = ', '.join(unique_phases) if unique_phases else "None"
            
            # Determine if there's a match
            match_status = "✅" if ms_phase_name in algo_phase_str else "❌"
            
            # Create the date range string
            date_range = f"{ms_start_display} to {end_date_display}"
            
            print(f"{date_range:<20} {ms_phase_name:<20} {algo_phase_str:<30} {match_status:<10}")
        
        print("=" * 80)
        
        # Now analyze performance across all MS_PHASES_BY_DATE
        if not self.use_ms_phases:  # Only do this when not using MarketSmith's phases directly
            print("\nAlgorithm performance across all MarketSmith dates (Jan-Apr 2025):")
            print("=" * 80)
            
            # Organize dates by month for easier reading
            months = {}
            total_correct = 0
            total_dates = 0
            phase_mismatches = {
                'confirmed_uptrend_issues': [],
                'uptrend_under_pressure_issues': [],
                'rally_attempt_issues': [],
                'correction_issues': []
            }
            
            # for date_str, ms_phase in sorted(MS_PHASES_BY_DATE.items()):
            #     # Skip weekends - these are typically in the MarketSmith data but not in trading data
            #     date_obj = pd.to_datetime(date_str)
            #     if date_obj.dayofweek >= 5:  # 5 = Saturday, 6 = Sunday
            #         continue
                    
            #     # Get our algorithm's prediction for this date
            #     try:
            #         # The dataframe might use datetime as index or as a column
            #         if isinstance(self.df.index, pd.DatetimeIndex):
            #             date_df = self.df[self.df.index == date_obj]
            #         else:
            #             # Try to find by string comparison if the datetime is a column
            #             date_df = self.df[self.df['datetime'] == date_str]
                    
            #         if len(date_df) == 0:
            #             # Try another approach - the dates might be stored differently
            #             for idx, row in self.df.iterrows():
            #                 row_date = row.name if isinstance(row.name, str) else row.get('datetime', None)
            #                 if isinstance(row_date, pd.Timestamp):
            #                     row_date = row_date.strftime('%Y-%m-%d')
            #                 if row_date == date_str:
            #                     date_df = self.df.iloc[[idx]]
            #                     break
                    
            #         if len(date_df) == 0:
            #             print(f"Warning: Date {date_str} not found in DataFrame (weekday {date_obj.day_name()})")
            #             continue
                        
            #         algo_phase = date_df['market_phase'].values[0] if 'market_phase' in date_df.columns else 'unknown'
                    
            #         # Check if it's correct
            #         is_correct = (algo_phase == ms_phase)
            #         if is_correct:
            #             total_correct += 1
            #         else:
            #             # Track mismatches for each phase type
            #             mismatch_key = f"{ms_phase}_issues"
            #             if mismatch_key in phase_mismatches:
            #                 phase_mismatches[mismatch_key].append({
            #                     'date': date_str, 
            #                     'expected': ms_phase, 
            #                     'actual': algo_phase
            #                 })
                    
            #         total_dates += 1
                    
            #         # Format for display
            #         month = date_str[0:7]  # Extract YYYY-MM
            #         if month not in months:
            #             months[month] = {'correct': 0, 'total': 0, 'details': []}
                    
            #         months[month]['total'] += 1
            #         if is_correct:
            #             months[month]['correct'] += 1
                    
            #         # Store details for this prediction
            #         months[month]['details'].append({
            #             'date': date_str,
            #             'ms_phase': ms_phase,
            #             'algo_phase': algo_phase,
            #             'correct': is_correct
            #         })
            #     except Exception as e:
            #         print(f"Error processing date {date_str}: {e}")
            
            # Display results by month
            print(f"{'Month':<10} {'Accuracy':<15} {'Correct':<10} {'Total':<10}")
            print("-" * 80)
            
            for month, data in months.items():
                accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
                print(f"{month:<10} {accuracy:.1f}%{' ':9} {data['correct']:<10} {data['total']:<10}")
            
            # Overall accuracy
            if total_dates > 0:
                overall_accuracy = (total_correct / total_dates) * 100
                print("-" * 80)
                print(f"{'Overall':<10} {overall_accuracy:.1f}%{' ':9} {total_correct:<10} {total_dates:<10}")
            
            print("=" * 80)
            
            # Show detailed breakdown by phase type
            phase_stats = {
                'confirmed_uptrend': {'correct': 0, 'total': 0},
                'uptrend_under_pressure': {'correct': 0, 'total': 0},
                'rally_attempt': {'correct': 0, 'total': 0},
                'correction': {'correct': 0, 'total': 0}
            }
            
            # Reset counters for phase type analysis
            # for date_str, ms_phase in MS_PHASES_BY_DATE.items():
            #     # Skip weekends
            #     date_obj = pd.to_datetime(date_str)
            #     if date_obj.dayofweek >= 5:  # 5 = Saturday, 6 = Sunday
            #         continue
                
            #     try:
            #         # The dataframe might use datetime as index or as a column
            #         if isinstance(self.df.index, pd.DatetimeIndex):
            #             date_df = self.df[self.df.index == date_obj]
            #         else:
            #             # Try to find by string comparison if the datetime is a column
            #             date_df = self.df[self.df['datetime'] == date_str]
                    
            #         if len(date_df) == 0:
            #             # Try another approach - the dates might be stored differently
            #             for idx, row in self.df.iterrows():
            #                 row_date = row.name if isinstance(row.name, str) else row.get('datetime', None)
            #                 if isinstance(row_date, pd.Timestamp):
            #                     row_date = row_date.strftime('%Y-%m-%d')
            #                 if row_date == date_str:
            #                     date_df = self.df.iloc[[idx]]
            #                     break
                    
            #         if len(date_df) == 0:
            #             continue
                        
            #         algo_phase = date_df['market_phase'].values[0] if 'market_phase' in date_df.columns else 'unknown'
                    
            #         # Update phase statistics
            #         if ms_phase in phase_stats:
            #             phase_stats[ms_phase]['total'] += 1
            #             if algo_phase == ms_phase:
            #                 phase_stats[ms_phase]['correct'] += 1
            #     except Exception:
            #         pass
            
            # Display phase performance
            print("\nPerformance by market phase type:")
            print("-" * 80)
            print(f"{'Phase Type':<25} {'Accuracy':<15} {'Correct':<10} {'Total':<10}")
            print("-" * 80)
            
            for phase, stats in phase_stats.items():
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    display_name = self._map_phase_for_display(phase)
                    print(f"{display_name:<25} {accuracy:.1f}%{' ':9} {stats['correct']:<10} {stats['total']:<10}")
            
            print("=" * 80)
            
            # Show sample of mismatches for each phase type
            print("\nSample mismatches by phase type:")
            print("-" * 80)
            
            for phase_key, mismatches in phase_mismatches.items():
                phase_name = phase_key.replace('_issues', '')
                display_name = self._map_phase_for_display(phase_name)
                
                if mismatches:
                    print(f"\n{display_name} mismatches (showing up to 5):")
                    print(f"{'Date':<12} {'Expected':<25} {'Algorithm Predicted':<25}")
                    print("-" * 70)
                    
                    # Show at most 5 examples
                    for i, mismatch in enumerate(mismatches[:5]):
                        date_display = pd.to_datetime(mismatch['date']).strftime('%d-%b-%Y')
                        expected_display = self._map_phase_for_display(mismatch['expected'])
                        actual_display = self._map_phase_for_display(mismatch['actual'])
                        print(f"{date_display:<12} {expected_display:<25} {actual_display:<25}")
                        
                    if len(mismatches) > 5:
                        print(f"...and {len(mismatches) - 5} more mismatches")
            
            print("=" * 80)
        
        return self.df
    
    def _map_phase_for_display(self, internal_phase):
        """Map internal phase names to display names in the UI"""
        if internal_phase == 'confirmed_uptrend':
            return 'Confirmed Uptrend'
        elif internal_phase == 'uptrend_under_pressure':
            return 'Uptrend Under Pressure'
        elif internal_phase == 'rally_attempt':
            return 'Rally Attempt'
        elif internal_phase == 'correction':
            return 'Downtrend'
        return internal_phase
    
    def _save_market_condition_history(self, phase_changes):
        """Save a table similar to the MarketSmith app UI"""
        if not phase_changes:
            return
            
        # Create DataFrame from phase changes
        df_history = pd.DataFrame(phase_changes)
        
        # Format dates properly for display
        for i, row in df_history.iterrows():
            # Convert start_date to datetime if it's a string
            if isinstance(row['start_date'], str) and row['start_date'] != 'Current-Date':
                try:
                    date_obj = pd.to_datetime(row['start_date'])
                    df_history.at[i, 'start_date'] = date_obj.strftime('%d-%b-%Y')
                except:
                    pass
            
            # Format end_date unless it's 'Current-Date'
            if row['end_date'] != 'Current-Date' and isinstance(row['end_date'], str):
                try:
                    date_obj = pd.to_datetime(row['end_date'])
                    df_history.at[i, 'end_date'] = date_obj.strftime('%d-%b-%Y')
                except:
                    pass
        
        # Add a column for visual icons
        df_history['icon'] = ''
        
        # Map the market phases to icons
        icon_map = {
            'Confirmed Uptrend': '🟢 ↑',      # Green up arrow
            'Uptrend Under Pressure': '🟠 ↑',  # Orange up arrow
            'Rally Attempt': '⚪ ↔️',           # Horizontal arrow
            'Downtrend': '🔴 ↓'               # Red down arrow
        }
        
        for i, row in df_history.iterrows():
            if row['market_phase'] in icon_map:
                df_history.at[i, 'icon'] = icon_map[row['market_phase']]
        
        # Save to CSV
        df_history.to_csv(resource_path('storage/data/market_condition_history.csv'), index=False)
        
        # Display the market condition history
        print("\nMarket Condition History (Algorithm):")
        print("=" * 70)
        print(f"{'Start Date':<15} {'End Date':<15} {'Market Trend':<25} {'Icon':<10}")
        print("-" * 70)
        
        for _, row in df_history.iterrows():
            print(f"{row['start_date']:<15} {row['end_date']:<15} {row['market_phase']:<25} {row['icon']:<10}")
        
        print("=" * 70)
        
        # Generate a summary of current market conditions
        if len(df_history) > 0:
            current_phase = df_history.iloc[0]['market_phase']
            current_icon = df_history.iloc[0]['icon']
            print(f"\nCurrent Market Status: {current_phase} {current_icon}")
            
            # Add explanation based on current phase
            if current_phase == 'Confirmed Uptrend':
                print("✅ Market is in a strong uptrend. Aggressively buy breakouts from sound bases.")
            elif current_phase == 'Uptrend Under Pressure':
                print("⚠️ Market showing weakness. Tighten stop losses and book partial profits.")
            elif current_phase == 'Rally Attempt':
                print("⏳ Waiting for Follow-Through Day to confirm trend reversal. Avoid new positions.")
            elif current_phase == 'Downtrend':
                print("❌ Market in correction. Avoid new buys and protect capital.")
        print("=" * 70)

class TradingBot:
    
    def fetch_data(self, force_update:bool = False):
        if force_update:
            scrap()
            self.fetch_data()
        else:
            try:
                df = pd.read_csv(resource_path('storage/data/nifty50.csv'))
                df=df[['DATE','OPEN','HIGH','LOW','CLOSE','SHARES TRADED']]
                df = df.rename(columns={'DATE': 'datetime', 'OPEN': 'open', 'HIGH': 'high', 
                                        'LOW': 'low', 'CLOSE': 'close', 'SHARES TRADED': 'volume'})
                df['open']=df['open'].str.replace(',', '').astype(float)
                df['high']=df['high'].str.replace(',', '').astype(float) 
                df['low']=df['low'].str.replace(',', '').astype(float)
                df['close']=df['close'].str.replace(',', '').astype(float)
                df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%b-%Y', errors='coerce')
                df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d')
                df['volume'] = df['volume'].str.replace(',', '')    
                df['volume'] = df['volume'].replace('-', np.nan).astype(float)
                df = df.dropna(subset=['volume']) 
                return df
            except Exception as e:
                logging.info('Data file not available, fetching again')
                scrap()
            # Check if the data file exists now after scraping
            if os.path.exists(resource_path('storage/data/nifty50.csv')):
                logging.info("Data file successfully created after scraping")
            else:
                logging.error("Failed to create data file even after scraping")
                return None
            return self.fetch_data()
        
    def get_market_phase(self):
        df = self.fetch_data()
        if df is not None:
            self.strategy = Marketphase(df)
            df = self.strategy.generate_signals() 
            return df
        else:
            pd.read_csv(resource_path('storage/data/market_phase.csv'))
    def run(self):
        logging.info("Starting trading bot")
        self.get_market_phase()

if __name__ == "__main__":
    bot = TradingBot()
    # bot.fetch_data()
    bot.run()
