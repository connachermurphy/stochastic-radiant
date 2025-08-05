# stochastic-radiant: LLM-generated forecasts
I implement a barebones tool for LLM-generated forecasts. The tool only supports the Anthropic API at present.

## Example usage
See `examples/single_forecast.py` for sample usage on a single forecasting question. The script requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY={your_anthropic_api_key}
```

## Contributions
Please contact me if you're interested in contributing. These tools are being actively developed for use in a submission to [ForecastBench](https://www.forecastbench.org/).