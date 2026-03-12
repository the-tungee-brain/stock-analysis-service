import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4.1"
    MAX_OUTPUT_TOKENS: int = 500

    def validate(self):
        if not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")


settings = Settings()
settings.validate()
