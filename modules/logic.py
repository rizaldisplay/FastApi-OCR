import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

# 1. Define PROMPT TEMPLATE with escaped JSON format
PROMPT_TEMPLATE = """
Anda adalah asisten cerdas yang ahli dalam menganalisis dan merangkum data medis.
Berdasarkan teks hasil OCR dari surat dokter berikut, ekstrak informasi kunci dan sajikan dalam format JSON yang valid.

Teks Hasil OCR:
---
{text}
---

Tugas Anda adalah mengembalikan HANYA objek JSON, tanpa teks pembuka atau penutup.
Struktur JSON yang wajib diikuti adalah sebagai berikut:
{{
    "nama_pasien": "string atau null",
    "tanggal_surat": "string dengan format YYYY-MM-DD atau null",
    "diagnosis": "string atau null",
    "resep_obat": [
        {{
            "nama_obat": "string",
            "dosis": "string",
            "aturan_pakai": "string"
        }}
    ],
    "instruksi_tambahan": "string atau null"
}}

"""

# 2. Initialize the model from Groq with a valid model
LLM_MODEL_NAME = "llama3-70b-8192" 

# 3. Create a logic function that uses LangChain correctly
async def get_structured_summary(image_text: str) -> dict:
    """
    Mengambil teks mentah OCR, memprosesnya melalui chain LLM, dan mengembalikan dictionary.
    """
    
    parser = JsonOutputParser()
    
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["text"],
        # We no longer need partial_variables because the example format is already in the template
    )
    
    model = ChatGroq(
        temperature=0.2,
        model_name=LLM_MODEL_NAME,
        api_key=os.getenv("GROQ_API_KEY") 
    )
        
    chain = prompt | model | parser
    
    response_dict = await chain.ainvoke({"text": image_text})
    
    return response_dict