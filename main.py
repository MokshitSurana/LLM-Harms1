from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Together

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    llm_name: str
    template: str
    message: str


def create_model(llm_name):
    return Together(
        model=llm_name,
        temperature=0.4,
        max_tokens=1024,
        top_k=50,
        together_api_key="ef98f44e5b21c4c0766d61e00538c988987e90c64d3445744e19367c563b10dc",
    )


def process_template_and_message(model, template, message):
    prompt = ChatPromptTemplate.from_template(template)
    chain = {"message": RunnablePassthrough()} | prompt | model | StrOutputParser()
    output = chain.invoke(message)
    return output


@app.post("/chat/")
async def chat_with_llm(request: ChatRequest):
    if not request.llm_name or not request.template or not request.message:
        raise HTTPException(
            status_code=400, detail="Missing one or more required fields."
        )
    model = create_model(request.llm_name)
    output = process_template_and_message(model, request.template, request.message)
    return {"responses": output}
