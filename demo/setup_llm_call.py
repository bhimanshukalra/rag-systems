from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


def main():

    # Test openai
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    # Test anthropic
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
