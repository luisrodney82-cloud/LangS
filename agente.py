from pathlib import Path
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
 
from langchain_core.globals import set_debug
 
set_debug(False)
 
 
from dotenv import load_dotenv
from openai import OpenAI
 
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)
 
api_key_minha = os.getenv("OPENAI_API_KEY", "").strip()
 
if not api_key_minha:
    raise RuntimeError("OPENAI_API_KEY não foi encontrado no arquivo .env.")
 
if not api_key_minha.startswith("sk-"):
    raise RuntimeError("OPENAI_API_KEY não é válido. Certifique-se de que a chave começa com 'sk-'.")
 
 
class QualProdutoinIndicado(BaseModel):
    modelo_indicado: str = Field("Qual modelo do produto é mais indicado para o ambiente")
    motivo: str = Field("Motivo pelo qual o modelo é mais indicado para o ambiente")
 
parseador = JsonOutputParser(pydantic_object=QualProdutoinIndicado)
 
class QualProdutoCrossSell(BaseModel):
    Produtos_CrossIndicados: str = Field("Qual poderia ser o produto cross-sell mais indicado para o ambiente")
    motivo_Cross: str = Field("Motivo indicação destes produtos")
 
parseador_cross = JsonOutputParser(pydantic_object=QualProdutoCrossSell)
 
 
prompt_produto = PromptTemplate(
    template= """Crie um agente de suporte a vendas para o produto {produto} O agente
           deve ser capaz de responder a perguntas sobre o produto, destacando o modelo, as características para uso na
           {ambiente}
            O agente deve ser amigável, profissional e persuasivo.
 
            {formato_de_saida}""",
    input_variables=["produto", "ambiente"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)
 
prompt_produtoCross = PromptTemplate(
    template= """Sugira produtos complementares para o produto {modelo_indicado} que sejam adequados para uso no {para_o_ambiente}.
 
            {formato_de_saida}""",
    input_variables=["modelo_indicado", "para_o_ambiente"],
    partial_variables={"formato_de_saida": parseador_cross.get_format_instructions()}
)
 
prompt_produtoUpSell = PromptTemplate(
    template= """Sugira produtos necessários para o {modelo_indicado},
    diferentes dos produtos existentes em {Produtos_CrossIndicados}.""",
    input_variables=["modelo_indicado", "Produtos_CrossIndicados"],
)
 
 
 
modelo = ChatOpenAI(
    model_name="gpt-4o-mini",
    api_key=api_key_minha,
    temperature=0.5
)
 
 
cadeia_1 = prompt_produto | modelo | parseador
cadeia_2 = prompt_produtoCross | modelo | parseador_cross
cadeia_3 = prompt_produtoUpSell | modelo | StrOutputParser()
 
indicacao_produto = cadeia_1.invoke({"produto": "TV 55 polegadas", "ambiente": "sala de estar"})
cross_sell = cadeia_2.invoke({
    "modelo_indicado": indicacao_produto["modelo_indicado"],
    "para_o_ambiente": "sala de estar",
})
 
up_sell = cadeia_3.invoke({
    "modelo_indicado": indicacao_produto["modelo_indicado"],
    "Produtos_CrossIndicados": cross_sell["Produtos_CrossIndicados"]
})
 
print("Produto indicado:", indicacao_produto)
print("Produtos vendas Cross:", cross_sell)
print("Produtos vendas UpSell:", up_sell)