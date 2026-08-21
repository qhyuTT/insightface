from __future__ import annotations


class PersonSearchError(Exception):
    code = "person_search_error"
    status_code = 500

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class ModelUnavailableError(PersonSearchError):
    code = "model_unavailable"
    status_code = 503


class EnrollmentError(PersonSearchError):
    status_code = 422


class SourceError(PersonSearchError):
    code = "source_unavailable"
    status_code = 503


class SearchStopTimeoutError(PersonSearchError):
    code = "search_stop_timeout"
    status_code = 503
