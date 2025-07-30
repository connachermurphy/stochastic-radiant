import anthropic


class Forecaster:
    def __init__(self, model, api_key):
        """
        Initialize the forecaster class.

        Args:
            model: The model to use for forecasting.
        """
        self.client = anthropic.Anthropic(api_key=api_key)

        self.model = model
        self.api_key = api_key

        self.system_prompt = """
You are a forecasting agent. You are given a prompt and you need to forecast the outcome.

Please state your forecast in the following format:

Forecast: [forecast]

Rationale: [rationale]

Your forecast should be a single number between 0 and 1.

Please limit your rationale to 100 words.
"""

    def forecast(self, question: str) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
                max_tokens=1000,
                temperature=0.0,
            )

            response = message.content[0].text

            forecast = float(response.split("Forecast: ")[1].split("\n")[0])
            rationale = response.split("Rationale: ")[1]

            return forecast, rationale
        except Exception as e:
            print(f"Error forecasting: {e}")
            return None
