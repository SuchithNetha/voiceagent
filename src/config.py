import os
from dotenv import load_dotenv

load_dotenv()

class config:
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")

    STT_MODEL="base"
    TTS_MODEL="kokoro"

    Base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    Data_path=os.path.join(Base_dir,"data","properties.csv")
    Avatar_dir=os.path.join(Base_dir,"avatars")

    @staticmethod
    def validate():
        if not config.GROQ_API_KEY:
            raise ValueError("Groq api key is missing!! (add groq api key at console.groq.com)")