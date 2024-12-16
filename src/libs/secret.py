from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jwt.exceptions import InvalidTokenError
import jwt
from schemes import *
from base_libs import *

TOKEN_URL = "/login"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=TOKEN_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5

fake_users_db = {
	"compareface": {
		"username": "compareface",
		"full_name": "DIXSON",
		"email": "diep.xuan.son@mqsolutions.com.vn",
		"hashed_password": "$2b$12$ctx6c0gUBG8aFylmPahvE.677QfrNWcLwvOBxZ86b7lgPLqr1DdDW",
		"disabled": False,
	}
}

#-------------secret--------------
def verify_password(plain_password, hashed_password):
	return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password="aiMQ@20130516"):
	return pwd_context.hash(password)

def get_user(db, username: str):
	if username in db:
		user_dict = db[username]
		return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
	user = get_user(fake_db, username)
	if not user:
		return False
	if not verify_password(password, user.hashed_password):
		return False
	return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
	to_encode = data.copy()
	if expires_delta:
		expire = datetime.now(timezone.utc) + expires_delta
	else:
		expire = datetime.now(timezone.utc) + timedelta(minutes=15)
	to_encode.update({"exp": expire})
	encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
	return encoded_jwt

async def get_current_user(token: str = oauth2_scheme):
	credentials_exception = HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail="Could not validate credentials",
		headers={"WWW-Authenticate": "Bearer"},
	)
	try:
		payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
		username: str = payload.get("usr")
		password: str = payload.get("pass")
		if username is None:
			raise credentials_exception
		token_data = TokenData(username=username)
	except InvalidTokenError:
		raise credentials_exception
	user = get_user(fake_users_db, username=token_data.username)
	if user is None:
		raise credentials_exception
	return user

async def get_current_active_user(token: str):
	current_user: User = await get_current_user(token=token)
	if current_user.disabled:
		raise HTTPException(status_code=400, detail="Inactive user")
	return current_user
#////////////////////////////////////////////////////////////
