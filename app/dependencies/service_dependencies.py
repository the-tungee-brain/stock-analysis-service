from fastapi import Request


def get_llm_service(request: Request):
    return request.app.state.llm_service


def get_portfolio_service(request: Request):
    return request.app.state.portfolio_service


def get_schwab_auth_service(request: Request):
    return request.app.state.schwab_auth_service


def get_user_service(request: Request):
    return request.app.state.user_service
