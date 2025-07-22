from youtube_transcript_ai.text_splitter import split_text
from youtube_transcript_ai.embedding import data_embedding
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from youtube_transcript_ai.retrival import data_retriever
from youtube_transcript_ai.augmentation import final_prompt
from youtube_transcript_ai.generation import generate_output
from youtube_transcript_ai.translate_text import translate_hindi_to_english
from langdetect import detect

class YouTubeTranscriptChat:
    def __init__(self, video_id):
        self.video_id = video_id
        self.vector_store = None
        self.retriever = None
        self.chunks = None

    def load_and_embed(self):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(self.video_id)
            # Detect language of the first chunk
            first_lang = None
            if transcript_list:
                first_lang = detect(transcript_list[0]["text"])
            if first_lang == "hi":
                print("Detected Hindi transcript. Translating to English...")
                translated_chunks = [translate_hindi_to_english(chunk["text"]) for chunk in transcript_list]
            else:
                print("Detected English transcript or unknown. No translation needed.")
                translated_chunks = [chunk["text"] for chunk in transcript_list]
            transcript = " ".join(translated_chunks)
            self.chunks = split_text(transcript)
            self.vector_store = data_embedding(self.chunks)
            self.retriever = data_retriever(self.vector_store)
            print("Transcript loaded, processed, and embedded.")
        except TranscriptsDisabled:
            print("No captions available for this video.")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def ask(self, question):
        if not self.retriever:
            print("Transcript not loaded. Call load_and_embed() first.")
            return None
        docs = list(self.retriever.invoke(question)) # Retrieve relevant documents
        context = " ".join([doc.page_content for doc in docs])
        prompt = final_prompt(context, question)
        if prompt is None:
            print("Prompt is None. Skipping generation.")
            return None
        answer = generate_output(prompt)
        return answer.content if hasattr(answer, 'content') else answer
