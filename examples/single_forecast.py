import os

import dotenv

from stochastic_radiant.forecaster import Forecaster

dotenv.load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

forecaster = Forecaster(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY)

question = """
I am about to flip a coin. What is the probability that it will land on heads?
"""

forecast, rationale = forecaster.forecast(question)

print(f"Forecast: {forecast}")
print(f"Rationale: {rationale}")
