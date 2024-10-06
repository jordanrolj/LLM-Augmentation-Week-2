import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import openai
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.chat_models import ChatOpenAI


class RagEngine:
    def __init__(self):
        self.loader = PyPDFLoader("environmental_sci.pdf")

        # The text splitter is used to split the document into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

        self.chunks = self.loader.load_and_split(text_splitter=self.text_splitter)

        # We will now use the from_documents method to create a vectorstore from the chunks
        self.vectorstore = FAISS.from_documents(
            self.chunks, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        )
    
    def return_answer(self, question):
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            model_name="gpt-3.5-turbo"
        )
        retriever =  self.vectorstore.as_retriever(k=5)
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        return chain.invoke(question)



    