# %%
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS, InMemoryVectorStore
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional, TypedDict
import torch

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import streamlit as st

PDF_STORE = r"/Users/revathsankar/Documents/Adv Big AI/radiologyAI/Pub1564webNew-74666420.pdf"
torch.classes.__path__ = [] # add this line to manually set it to empty. 

class Pipeline:
    def __init__(self):
        embed_model = "BAAI/bge-large-en-v1.5"
        model = SentenceTransformer(embed_model)

        # Compute text embeddings
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def add_docs(self,):
        loader = PyPDFLoader(PDF_STORE,
                            mode="page",)
                            #  images_inner_format='markdown-img',
                            #  images_parser=RapidOCRBlobParser())
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        print(f"{len(all_splits)} chunks generated")

        embed_model = "BAAI/bge-large-en-v1.5"
        model = SentenceTransformer(embed_model)

        # Compute text embeddings
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=embed_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        #NOTE: Using FAISS Model
        index = faiss.IndexFlatL2(len(embeddings.embed_query("what is the weather?")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )


        doc_ids = vector_store.add_documents(documents=all_splits)


    def save_model(self):
        self.vector_store.save_local('radiology_faiss_index')


    def retrieve_context(self, query):
        vs = FAISS.load_local('radiology_faiss_index', embeddings=self.embeddings, allow_dangerous_deserialization=True)
        docs = vs.similarity_search_with_score(query)
        return docs





class DeepSeek:
    # @st.cache_resource
    def __init__(self):
        self.model_path = "deepseek-ai/deepseek-vl-7b-chat"
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cpu().eval()


    def get_initial_analysis(self, img_path):
        initial_analysis = [
            {
                "role": "User",
                "content": "<image_placeholder>Describe the tumor type in the image. Provide detailed explanation on how you arrived at your conclusion. Never provide an empty response; always provide some form of acknowledgement of the users request.",
                "images": [f"{img_path}"]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]
        return initial_analysis
    def extend_analysis(self, query:str, context:str, img_path:str):
        # print(f"[bold cyan] Image path: {img_path}")
        further_analysis = ([
            {
                "role": "User",
                "content": f"<image_placeholder>Reference this image when responding to user questions as well as the context provided, look at no other sources.\n\nBe sure to provide in-line citations in your response. Question: {query}\n\nContext: {context}",
                "images": [f"{img_path}"]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ])
        return further_analysis

    def analyze_image(self, conversation:str):
        # load images and prepare for inputs
        print("New thread spun!")
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"Response: {answer}")
        return answer



