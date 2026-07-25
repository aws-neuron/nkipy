from http import HTTPStatus

from fastapi.responses import JSONResponse

from nkipy_serving.entrypoints.openai.protocol import ErrorResponse
from nkipy_serving.managers.tokenizer_manager import TokenizerManager


class OpenAIServingBase:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def _create_error_response(self, message: str, code: int = 400) -> JSONResponse:
        err = ErrorResponse(
            message=message,
            type=HTTPStatus(code).phrase,
            code=code,
        )
        return JSONResponse(status_code=code, content=err.model_dump())
