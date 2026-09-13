import google.generativeai as palm
import google.ai.generativelanguage as gen_lang

from .Model import Model


import os
from dotenv import load_dotenv

class PaLM2(Model):
    def __init__(self, config):
        super().__init__(config)
        load_dotenv()
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if env_key:
            self.api_key = env_key
        else:
            api_keys = config["api_key_info"]["api_keys"]
            api_pos = int(config["api_key_info"]["api_key_use"])
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            self.api_key = api_keys[api_pos]
        self.set_API_key()
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        
    def set_API_key(self):
        palm.configure(api_key=self.api_key)
        
    def query(self, msg):
        import time
        max_retries = 6
        retry_delay = 10  # start with 10 seconds delay
        
        for attempt in range(max_retries):
            try:
                if 'gemini' in self.name.lower():
                    model = palm.GenerativeModel(self.name)
                    generation_config = palm.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                    completion = model.generate_content(
                        msg,
                        generation_config=generation_config
                    )
                    return completion.text
                elif 'text' in self.name:
                    completion = palm.generate_text(
                        model=self.name,
                        prompt=msg,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        safety_settings=[
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_DEROGATORY,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_TOXICITY,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_VIOLENCE,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_SEXUAL,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_MEDICAL,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                            {
                                "category": gen_lang.HarmCategory.HARM_CATEGORY_DANGEROUS,
                                "threshold": gen_lang.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
                            },
                        ]
                    )
                    return completion.result

                elif 'chat' in self.name:
                    return palm.chat(messages=msg, candidate_count=1).last
                else:
                    model = palm.GenerativeModel(self.name)
                    generation_config = palm.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                    completion = model.generate_content(
                        msg,
                        generation_config=generation_config
                    )
                    return completion.text
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    if attempt < max_retries - 1:
                        print(f"[!] Quota exceeded/Rate limit (429). Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"[!] Max retries reached for Gemini query due to rate limiting: {e}")
                        return ""
                else:
                    print(f"Error querying Gemini API: {e}")
                    return ""
        return ""