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

Your forecast should be a single number between 0 and 1, where 0 means the event will definitely not happen and 1 means it will definitely happen.

Please limit your rationale to 100 words and provide clear reasoning for your forecast.

Use the record_forecast tool to provide your structured forecast and rationale.
"""

    def forecast(self, question: str) -> tuple[float, str]:
        try:
            tools = [
                {
                    "name": "record_forecast",
                    "description": "Return a probability forecast and short rationale.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "forecast": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Probability between 0 and 1",
                            },
                            "rationale": {
                                "type": "string",
                                "maxLength": 200,
                                "description": "Brief explanation for the forecast (max 100 words)",
                            },
                        },
                        "required": ["forecast", "rationale"],
                        "additionalProperties": False,
                    },
                }
            ]

            message = self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}],
                tools=tools,
                tool_choice={"type": "tool", "name": "record_forecast"},
                max_tokens=1000,
                temperature=0.0,
            )

            # Extract the tool use result
            for block in message.content:
                if block.type == "tool_use" and block.name == "record_forecast":
                    args = block.input
                    forecast_value = float(args["forecast"])
                    rationale_text = args["rationale"]

                    # Validate the forecast is in the correct range
                    if not (0.0 <= forecast_value <= 1.0):
                        raise ValueError(
                            f"Forecast value {forecast_value} is not between 0 and 1"
                        )

                    return forecast_value, rationale_text

            raise RuntimeError("No tool_use block found in response")

        except Exception as e:
            print(f"Error forecasting: {e}")
            return None
