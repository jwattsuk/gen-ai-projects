from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

endpoint = "https://ms-gen-ai-jwattsuk.openai.azure.com/"
model_name = "gpt-5-mini"
deployment = "gpt-5-mini"

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

no_recipes = input("No of recipes (for example, 5): ")

ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots): ")

prompt = f"Show me {no_recipes} recipes for a dish with the following ingredients: {ingredients}. Per recipe, list all the ingredients used"

response = client.chat.completions.create(
    messages=[

        {
            "role": "user",
            "content": prompt,
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)

old_prompt_result = response.choices[0].message.content
prompt = "Produce a shopping list for the generated recipes and please don't include ingredients that I already have."

new_prompt = f"{old_prompt_result} {prompt}"
messages = [{"role": "user", "content": new_prompt}]
completion = client.chat.completions.create(
    messages=messages,
    max_completion_tokens=1200,
    model=deployment
)

# print response
print("Shopping list:")
print(completion.choices[0].message.content)