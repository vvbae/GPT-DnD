import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

from langchain.memory import (
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationKGMemory
)
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(temperature=0)
history = ChatMessageHistory()

template = """This is a conversation between the AI and a human, simulating a dungeons & dragons game. The AI will be the Dungeon Master of the game, and the human will be the player.
As an impartial Dungeon Master (DM) in a Dungeons & Dragons (D&D) game, the AI's principles are: fairness, fun, player agency, inclusivity, storytelling, flexibility, communication, challenge, Rule of Cool, and collaboration.
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY use information contained in the "Relevant Information" section and does not hallucinate.

When first start the game, the AI will introduce the goal of the game, and let user choose a character. Then it will give the human a prompt to start the game.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], template=template
)
dm = LLMChain(
    llm=llm, 
    verbose=True, 
    prompt=prompt,
    memory=ConversationKGMemory(llm=llm)
)

print("Welcome, adventurers! Gather 'round the table and prepare for an epic journey in a world of swords and sorcery. As your impartial Dungeon Master, I'll guide you through a story filled with danger, intrigue, and excitement. So grab your dice, ready your characters, and let's embark on an unforgettable adventure together!\n")

chat_turn_limit, n = 5, 0
while n < chat_turn_limit:
    user_cmd = input("Player: ")
    output = dm.predict(input=user_cmd)
    print(output)
    history.add_user_message(user_cmd)