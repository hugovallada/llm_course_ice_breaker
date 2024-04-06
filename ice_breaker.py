import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from agents.linkedin_lookup_agent import lookup
from output_parsers import person_intel_parser
from third_parties.linkedin import scrape_linkedin_profile


def ice_break(name: str) -> str:
    linkedin_profile_url = lookup(name=name)

    summary_template = """
        given the Linkedin {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
        \n{format_instructions}
        """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )

    llm = ChatOpenAI(
        temperature=0, model_name="gpt-3.5-turbo"
    )  # 0 -> Não vai ser criativo

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile("hugovallada")

    res = chain.run(information=linkedin_data)
    print(res)
    return res


if __name__ == "__main__":
    load_dotenv()
    print("Hello langchain")
    ice_break("Eden Marco")
