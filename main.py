from youtube_transcript_ai.constants import VIDEO_ID
from youtube_transcript_ai.youtube_chat import YouTubeTranscriptChat

if __name__ == "__main__":
    chat = YouTubeTranscriptChat(VIDEO_ID)
    chat.load_and_embed()
    while True:
        question = input("Ask a question about the video (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = chat.ask(question)
        print("Answer:", answer)
