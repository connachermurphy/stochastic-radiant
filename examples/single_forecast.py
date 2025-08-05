import os

import dotenv

from stochastic_radiant.forecaster import Forecaster

# Load the API key from the .env file
dotenv.load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Specify the model to use for forecasting
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Initialize the forecaster from the Forecaster class
# Pass in the model and the API key
forecaster = Forecaster(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)

# Create a question to forecast
question = """
I am about to flip a coin. What is the probability that it will land on heads?
"""

# The forecast method will return a tuple of the forecast and the rationale
forecast, rationale = forecaster.forecast(question)

# Print the forecast and the rationale
print(f"Forecast: {forecast}")
print(f"Rationale: {rationale}")
