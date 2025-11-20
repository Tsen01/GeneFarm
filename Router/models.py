from pydantic import BaseModel, Field

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = Field(..., regex="^(Farmer|GeneticResearcher)$")

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str
    role: str
    user_id: str
    username: str | None = None
    farmname: str | None = None
