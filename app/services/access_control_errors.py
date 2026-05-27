class WaitlistRequiredError(Exception):
    def __init__(
        self,
        message: str = "We're at capacity right now. You've been added to the waitlist.",
    ):
        self.message = message
        super().__init__(message)
