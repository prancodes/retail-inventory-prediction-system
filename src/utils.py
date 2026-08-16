import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging() -> logging.Logger:
    """
    Sets up a standard, professional logging configuration.
    Instead of using raw 'print' statements, real-world systems use logging
    to track execution info, warnings, and errors systematically.
    """
    logger = logging.getLogger("RetailInventorySystem")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Output to console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def set_premium_plot_style():
    """
    Applies a clean, modern, and high-quality visual style to all matplotlib
    and seaborn plots, avoiding default clunky looks.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#F8F9FA',
        'axes.edgecolor': '#E9ECEF',
        'grid.color': '#E9ECEF',
        'text.color': '#212529',
        'axes.labelcolor': '#495057',
        'xtick.color': '#495057',
        'ytick.color': '#495057',
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.frameon': True,
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#E9ECEF'
    })
