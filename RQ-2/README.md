# RQ-2: Comparative Analysis of RL Algorithms

This directory contains the implementation and analysis scripts for comparing different Reinforcement Learning algorithms' performance.

## Directory Structure

- `run_analysis.py`: Main script for running the comparative analysis
- `logs/`: Directory containing execution logs and results
- `plots/`: Directory containing generated plots and visualizations

## Requirements

```bash
numpy>=1.19.2
pandas>=1.2.0
matplotlib>=3.3.2
seaborn>=0.11.0
scipy>=1.6.0
```

## Running the Analysis

1. Make sure you have all the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis script:
   ```bash
   python run_analysis.py
   ```

## Output

The analysis will generate:
- Performance comparison plots in the `plots/` directory
- Detailed logs in the `logs/` directory
- Statistical analysis results in the console output

## Analysis Parameters

You can modify the following parameters in `run_analysis.py`:
- Algorithm selection
- Environment parameters
- Analysis metrics
- Plot configurations

## Troubleshooting

If you encounter any issues:
1. Check that all dependencies are correctly installed
2. Ensure you have sufficient disk space for logs and plots
3. Verify that the input data files are present in the correct format

For additional help, please create an issue in the main repository. 