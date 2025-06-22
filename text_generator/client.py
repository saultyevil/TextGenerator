from text_generator.clients.gemini_client import GeminiClient
from text_generator.clients.openai_client import OpenAIClient
from text_generator.models import TextGenerationInput, TextGenerationResponse


class TextGenerator:
    """Text generator class.

    This class is used to generate text using different LLM models. It is a
    wrapper around multiple LLM clients, initialised using `set_model()`.
    """

    SUPPORTED_OPENAI_MODELS = OpenAIClient.SUPPORTED_MODELS
    SUPPORTED_GOOGLE_MODELS = GeminiClient.SUPPORTED_MODELS
    SUPPORTED_MODELS = SUPPORTED_OPENAI_MODELS + SUPPORTED_GOOGLE_MODELS
    VISION_MODELS = (*OpenAIClient.VISION_MODELS, *GeminiClient.VISION_MODELS)
    SEARCH_MODELS = (*OpenAIClient.SEARCH_MODELS, *GeminiClient.SEARCH_MODELS)
    AUDIO_MODELS = (*OpenAIClient.AUDIO_MODELS, *GeminiClient.AUDIO_MODELS)
    VIDEO_MODELS = (*OpenAIClient.VIDEO_MODELS, *GeminiClient.VIDEO_MODELS)

    def __init__(self, model_name: str, api_key: str) -> None:
        """Initialise a TextGeneratorLLM with default values.

        Parameters
        ----------
        model_name : str | None
            The name of the LLM model to use
        api_key : str
            The secret key of the LLM API.

        """
        self.set_model(model_name, api_key)

    # --------------------------------------------------------------------------

    @property
    def model(self) -> str:
        """Get the current LLM model name."""
        return self._client.model_name

    @property
    def system_prompt(self) -> str:
        """Get the system prompt of the client."""
        return self._client.system_prompt

    @property
    def system_prompt_name(self) -> str:
        """Get the name of the system prompt of the client."""
        return self._client.system_prompt_name

    @property
    def size_messages(self) -> int:
        """Get the size of the context, in messages."""
        return len(self._client.context)

    @property
    def size_tokens(self) -> int:
        """Get the size of the context, in tokens."""
        return self._client.token_size

    @property
    def history(self) -> list[dict]:
        """Get the context history."""
        return self._client.context

    # --------------------------------------------------------------------------

    def count_tokens_for_message(self, message: dict | list[dict[str, str]] | str) -> int:
        """Get the token count for a given message for the current LLM model.

        Parameters
        ----------
        message : list[str] | str
            The message for which the token count needs to be computed.

        Returns
        -------
        int
            The count of tokens in the given message for the current model.

        """
        return self._client.count_tokens_for_message(message)

    def create_request_json(
        self, messages: TextGenerationInput | list[TextGenerationInput], *, system_prompt: str | None = None
    ) -> dict | list:
        """Create a request JSON for the current LLM model.

        Parameters
        ----------
        messages : ContextMessage | list[ContextMessage]
            Input message(s), from the user, including attached images and
            videos.
        system_prompt : str | None
            The system prompt to use. If None, the current system prompt is
            used.

        Returns
        -------
        dict | list
            The request JSON for the current LLM model.

        """
        return self._client.create_request_json(messages, system_prompt=system_prompt)

    async def generate_response_with_context(
        self, messages: TextGenerationInput | list[TextGenerationInput]
    ) -> TextGenerationResponse:
        """Generate text from the current LLM model.

        Parameters
        ----------
        messages : ContextMessage | list[ContextMessage]
            Input message(s), from the user, including attached images and
            videos.

        """
        return await self._client.generate_response_with_context(messages)

    async def send_response_request(self, content: list[dict] | dict) -> TextGenerationResponse:
        """Send a request to the API client.

        Parameters
        ----------
        content : list[dict]
            The (correctly) formatted content to send to the API.

        """
        return await self._client.send_response_request(content)

    def set_model(self, model: str, api_key: str) -> None:
        """Set the current LLM model.

        Parameters
        ----------
        model : str
            The name of the model to use.
        api_key : str
            The secret key of the LLM API.

        """
        if model in self.SUPPORTED_OPENAI_MODELS:
            self._client = OpenAIClient(model, api_key)
        elif model in self.SUPPORTED_GOOGLE_MODELS:
            self._client = GeminiClient(model, api_key)
        else:
            msg = f"{model} is not available"
            raise NotImplementedError(msg)

    def set_system_prompt(self, prompt: str, *, prompt_name: str = "unset name") -> None:
        """Set the system prompt.

        Parameters
        ----------
        prompt : str
            The system prompt to set.
        prompt_name : str
            The name of the system prompt.

        """
        self._client.set_system_prompt(prompt, prompt_name=prompt_name)
