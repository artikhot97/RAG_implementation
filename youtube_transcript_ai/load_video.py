from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from constants import VIDEO_ID
from text_splitter import split_text
from embedding import data_embedding

def load_video():
    print("Starting load_video function...")
    print(f"Using VIDEO_ID: {VIDEO_ID}")
    
    video_id = VIDEO_ID  # only the ID, not full URL
    try:
        print("Fetching transcript...")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        # print(transcript_list, "transcript_list")
        print("Transcript fetched, flattening...")
        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        
        print("Transcript: ", transcript)


        # split into chunks
        chunks = split_text(transcript)

        print(chunks)
        print(len(chunks))

        for chunk in chunks:
            print(chunk.page_content, "chunk.page_content")

            embedded_data = data_embedding([chunk])
            print(embedded_data, "embedded_data--")
            

    except TranscriptsDisabled:
        print("No captions available for this video.")
    except Exception as e:
        print(f"An error occurred: {e}")

# The YouTubeTranscriptChat class has been moved to youtube_chat.py

if __name__ == "__main__":
    load_video()


# Environment and core
python-dotenv==1.0.1
pytest==8.2.2

# Data and vector stores
chromadb==0.4.24
faiss-cpu==1.8.0

# Language models and embeddings
langchain==0.3.3
langchain-community==0.3.2
langchain-openai==0.1.11
langchain-google-genai==0.0.11

tiktoken==0.7.0
transformers==4.40.2
torch==2.3.0
langdetect==1.0.9
sentencepiece==0.2.0

youtube-transcript-api==0.6.2
