# relative_rotation_graph.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RRG')

class RRG:
    def __init__(self, stock_file, index_file, stock_name=None):
        """
        Initialize the RRG analysis with CSV files that already contain the data.
        
        Args:
            stock_file (str): Path to CSV file containing stock price data
            index_file (str): Path to CSV file containing index price data
            stock_name (str, optional): Name of the stock being analyzed (defaults to filename if None)
        """
        self.stock_file = stock_file
        self.index_file = index_file
        self.stock_data = None
        self.index_data = None
        self.rrg_data = {}
        
        # Extract stock name from filename if not provided
        if stock_name:
            self.stock_name = stock_name
        else:
            # Extract name from file path (remove extension and path)
            import os
            base_name = os.path.basename(stock_file)
            self.stock_name = base_name.split('_')[0]  # Assumes format like "reliance_fetched_data..."
        
        logger.info(f"Analyzing stock: {self.stock_name}")
    
    def load_data(self):
        """
        Load data from CSV files.
        """
        try:
            # Load stock data
            logger.info(f"Loading stock data from {self.stock_file}")
            stock_df = pd.read_csv(self.stock_file)
            
            # Load index data
            logger.info(f"Loading index data from {self.index_file}")
            index_df = pd.read_csv(self.index_file)
            
            # Ensure datetime format is correct
            if 'datetime' in stock_df.columns:
                stock_df['date'] = pd.to_datetime(stock_df['datetime'])
                stock_df.set_index('date', inplace=True)
                self.stock_data = stock_df['close']
            else:
                # Try to find a datetime column (might be named differently)
                date_cols = [col for col in stock_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    stock_df['date'] = pd.to_datetime(stock_df[date_cols[0]])
                    stock_df.set_index('date', inplace=True)
                    self.stock_data = stock_df['close']
                else:
                    raise ValueError("No datetime column found in stock data")
            
            # Process index data similarly
            if 'datetime' in index_df.columns:
                index_df['date'] = pd.to_datetime(index_df['datetime'])
                index_df.set_index('date', inplace=True)
                self.index_data = index_df['close']
            else:
                # Try to find a datetime column
                date_cols = [col for col in index_df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    index_df['date'] = pd.to_datetime(index_df[date_cols[0]])
                    index_df.set_index('date', inplace=True)
                    self.index_data = index_df['close']
                else:
                    raise ValueError("No datetime column found in index data")
            
            # Check that we have sufficient data
            if len(self.stock_data) < 10 or len(self.index_data) < 10:
                logger.warning("Not enough data points for reliable RRG analysis")
                
            logger.info(f"Successfully loaded data: {len(self.stock_data)} stock records, {len(self.index_data)} index records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def compute_rrg_features(self, period=12):
        """
        Compute the RRG features (Relative Strength and Momentum).
        
        Args:
            period (int): The period to use for momentum calculation (default: 12)
        """
        if self.index_data is None or self.stock_data is None:
            logger.error("No data available. Run load_data() first.")
            return False
            
        try:
            # Align dates to ensure we're comparing the same time periods
            aligned_df = pd.DataFrame({
                "stock": self.stock_data,
                "index": self.index_data
            })
            
            # Handle any missing values by forward filling then backward filling
            aligned_df = aligned_df.dropna()
            
            if aligned_df.empty:
                logger.warning("No aligned data available after removing NaN values")
                return False
                
            # Calculate Relative Strength (RS) - normalized ratio of stock to index
            aligned_df['rs'] = (aligned_df['stock'] / aligned_df['stock'].iloc[0]) / (aligned_df['index'] / aligned_df['index'].iloc[0])
            aligned_df['rs_log'] = np.log(aligned_df['rs'])
            
            # Calculate RS-Momentum (rate of change in RS)
            if len(aligned_df) > period:
                aligned_df['rs_momentum'] = aligned_df['rs_log'] - aligned_df['rs_log'].shift(period)
            else:
                # If we don't have enough data for the requested period, use half the data length
                fallback_period = max(1, len(aligned_df) // 2)
                logger.warning(f"Not enough data for {period} period, using {fallback_period} instead")
                aligned_df['rs_momentum'] = aligned_df['rs_log'] - aligned_df['rs_log'].shift(fallback_period)
            
            # Store the latest values that aren't NaN
            latest = aligned_df.dropna().iloc[-1]
            self.rrg_data = {
                "RS": latest['rs_log'],
                "Momentum": latest['rs_momentum'],
                "Data": aligned_df  # Store full dataset for trailing analysis
            }
            
            logger.info(f"Computed RRG features for {self.stock_name}")
            logger.info(f"RS: {self.rrg_data['RS']:.6f}, Momentum: {self.rrg_data['Momentum']:.6f}")
            return True
                
        except Exception as e:
            logger.error(f"Error computing RRG features: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def plot_rrg(self, trail_periods=0):
        """
        Plot the RRG chart.
        
        Args:
            trail_periods (int): Number of periods to draw as a trail (default: 0)
        """
        if not self.rrg_data:
            logger.error("No RRG data available. Run compute_rrg_features() first.")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Draw quadrant lines
        plt.axhline(0, color='grey', linestyle='--', alpha=0.7)
        plt.axvline(0, color='grey', linestyle='--', alpha=0.7)
        
        # Add title with stock name
        plt.title(f"Relative Rotation Graph (RRG): {self.stock_name} vs NIFTY", fontsize=14)
        
        # Define quadrant labels - positioning them within the quadrants
        # Use coordinates relative to the axis limits
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Add quadrant labels (fixed positions relative to chart center)
        plt.text(-0.02, 0.01, "Improving", fontsize=12, ha='right', va='bottom')
        plt.text(0.02, 0.01, "Leading", fontsize=12, ha='left', va='bottom')
        plt.text(-0.02, -0.01, "Lagging", fontsize=12, ha='right', va='top')
        plt.text(0.02, -0.01, "Weakening", fontsize=12, ha='left', va='top')
        
        # Plot the stock position
        rs = self.rrg_data['RS']
        mom = self.rrg_data['Momentum']
        
        # Plot current position
        plt.scatter(rs, mom, s=100, color='blue')
        plt.text(rs + 0.0005, mom + 0.0005, self.stock_name, fontsize=10)
        
        # Draw trail if requested
        if trail_periods > 0 and 'Data' in self.rrg_data:
            data = self.rrg_data['Data'].dropna().tail(trail_periods+1)
            if len(data) > 1:
                plt.plot(data['rs_log'].values, data['rs_momentum'].values, 
                         'b-', alpha=0.5, linewidth=1)
        
        plt.xlabel("Relative Strength (log-ratio)")
        plt.ylabel("Momentum")
        plt.grid(True, alpha=0.3)
        
        # Add RRG quadrant interpretation
        quadrant, interpretation = self.get_quadrant_info(rs, mom)
        plt.figtext(0.02, 0.02, f"{quadrant} Quadrant: {interpretation}", wrap=True, fontsize=9)
        
        # Make sure the axis ranges are appropriately set and symmetric
        # This ensures the chart is properly centered and scaled
        max_range = max(abs(rs) * 1.2, abs(mom) * 1.2, 0.02)  # Minimum range to ensure visibility
        plt.xlim(-max_range, max_range)
        plt.ylim(-max_range, max_range)
        
        plt.tight_layout()
        
        # Save the chart
        output_filename = f"rrg_chart_{self.stock_name}.png"
        try:
            plt.savefig(output_filename, dpi=300)
            logger.info(f"RRG chart saved to {output_filename}")
        except Exception as e:
            logger.error(f"Error saving chart: {str(e)}")
            
        plt.show()
        
    def get_quadrant_info(self, rs, mom):
        """
        Get interpretation based on which quadrant the stock is in.
        
        Args:
            rs (float): Relative Strength value
            mom (float): Momentum value
            
        Returns:
            tuple: (quadrant_name, interpretation)
        """
        if rs > 0 and mom > 0:
            return "Leading", "Strong and improving relative to benchmark"
        elif rs > 0 and mom < 0:
            return "Weakening", "Still strong but losing momentum"
        elif rs < 0 and mom < 0:
            return "Lagging", "Weak and deteriorating relative to benchmark"
        else:  # rs < 0 and mom > 0
            return "Improving", "Still weak but gaining momentum"
            
    def analyze_trend(self):
        """
        Provide trend analysis based on RRG position.
        
        Returns:
            tuple: (quadrant, interpretation)
        """
        if not self.rrg_data:
            logger.error("No RRG data available. Run compute_rrg_features() first.")
            return None, None
            
        rs = self.rrg_data['RS']
        mom = self.rrg_data['Momentum']
        
        # Determine quadrant and get interpretation
        quadrant, short_interp = self.get_quadrant_info(rs, mom)
        
        # Extended interpretation
        if quadrant == "Leading":
            interpretation = (
                "The stock is in the Leading quadrant, showing both strong performance "
                "and positive momentum relative to the benchmark. This is typically "
                "considered favorable for maintaining positions or potential additions."
            )
        elif quadrant == "Weakening":
            interpretation = (
                "The stock is in the Weakening quadrant. While still outperforming the benchmark, "
                "momentum is decreasing, which could signal potential deterioration in relative performance. "
                "Consider monitoring closely or reducing exposure."
            )
        elif quadrant == "Lagging":
            interpretation = (
                "The stock is in the Lagging quadrant, underperforming the benchmark with "
                "negative momentum. This typically signals weakness relative to the market."
            )
        else:  # Improving
            interpretation = (
                "The stock is in the Improving quadrant. While still underperforming the benchmark, "
                "momentum is improving, which could signal potential opportunities if the trend continues."
            )
        
        # Print analysis report
        print("\n===== RRG Analysis Report =====")
        print(f"Stock: {self.stock_name}")
        print(f"Relative Strength (RS): {rs:.6f}")
        print(f"RS-Momentum: {mom:.6f}")
        print(f"Quadrant: {quadrant}")
        print(f"Interpretation: {interpretation}")
        print("=============================\n")
        
        return quadrant, interpretation

if __name__ == "__main__":
    # Using the CSV files you already have
    stock_file = input("Enter stock data file name (e.g., reliance_fetched_data_1year_15m.csv): ")
    index_file = input("Enter index data file name (e.g., nifty_fetched_data_1year_15m.csv): ")
    
    # Get stock name from user or extract from filename
    stock_name = input("Enter stock name (press Enter to use filename): ")
    if not stock_name:
        # Extract from filename
        import os
        base_name = os.path.basename(stock_file)
        stock_name = base_name.split('_')[0].upper()  # Convert to uppercase for display
    
    try:
        logger.info(f"Initializing RRG analysis for {stock_name}")
        print(f"Initializing RRG analysis for {stock_name}")
        
        # Create RRG instance with explicit stock name
        rrg = RRG(stock_file=stock_file, index_file=index_file, stock_name=stock_name)
        
        # Load data from CSV files
        if rrg.load_data():
            logger.info("Computing RRG features")
            print("Computing RRG features")
            
            # Compute with 12-period momentum (adjust as needed)
            if rrg.compute_rrg_features(period=12):
                # Analyze the trend
                rrg.analyze_trend()
                
                # Plot RRG with a trail of the last 10 periods
                logger.info("Plotting RRG chart")
                print("Plotting RRG chart")
                rrg.plot_rrg(trail_periods=10)
            else:
                logger.error("Failed to compute RRG features")
                print("Failed to compute RRG features")
        else:
            logger.error("Failed to load data")
            print("Failed to load data")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()