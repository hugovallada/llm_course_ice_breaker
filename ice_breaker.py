import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv()
    print("Hello langchain")

    information = """
    Elon Reeve Musk FRS (Pretória, 28 de junho de 1971) é um empreendedor,
    [3] empresário e filantropo sul-africano-canadense, naturalizado estadunidense. 
    Ele é o fundador, diretor executivo e diretor técnico da SpaceX; CEO da Tesla, Inc.; 
    vice-presidente da OpenAI, fundador e CEO da Neuralink; cofundador, 
    presidente da SolarCity e proprietário do Twitter (X). 
    Em 2023, ele era a pessoa mais rica do mundo, com um patrimônio líquido estimado em US$ 225 bilhões 
    de dólares, de acordo com o Bloomberg Billionaires Index. Já a revista Forbes estimou sua fortuna em 
    US$ 234 bilhões, principalmente de suas participações acionárias nas empresas Tesla e na SpaceX.
    """

    summary_template = """
    given the {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo"
    )  # 0 -> Não vai ser criativo

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    res = chain.invoke(input={"information": information})
    print(res)
