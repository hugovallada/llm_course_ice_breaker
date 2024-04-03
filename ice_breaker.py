import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from third_parties.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()
    print("Hello langchain")

    linkedin_data = scrape_linkedin_profile("hugovallada")

    summary_template = """
    given the Linkedin {information} about a person I want you to create:
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

    res = chain.run(information=linkedin_data)
    print(res)
