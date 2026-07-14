from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    access_key_or_token: str
    base_url: str = "https://api.openai.com/v1"
